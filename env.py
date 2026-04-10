import torch
import einops
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
# [Old - Isaac Sim 2023.1.0] import omni.isaac.orbit.sim as sim_utils
import omni.isaac.lab.sim as sim_utils  # [New - Isaac Sim 4.1.0] orbit -> lab
from omni_drones.robots.drone import MultirotorBase
# [Old - Isaac Sim 2023.1.0] from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.lab.assets import AssetBaseCfg  # [New - Isaac Sim 4.1.0] orbit -> lab
# [Old - Isaac Sim 2023.1.0] from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg  # [New - Isaac Sim 4.1.0] orbit -> lab
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
# [Old - Isaac Sim 2023.1.0] from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns  # [New - Isaac Sim 4.1.0] orbit -> lab
from omni.isaac.core.utils.viewports import set_camera_view
from tools import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
# [Old - Isaac Sim 2023.1.0] import omni.isaac.orbit.utils.math as math_utils
import omni.isaac.lab.utils.math as math_utils  # [New - Isaac Sim 4.1.0] orbit -> lab
# [Old - Isaac Sim 2023.1.0] from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg  # [New - Isaac Sim 4.1.0] orbit -> lab
import time
from omni_drones.utils.torch import quat_rotate_inverse
# [Old - Isaac Sim 2023.1.0] from omni.isaac.orbit.sensors import Camera, CameraCfg
from omni.isaac.lab.sensors import Camera, CameraCfg  # [New - Isaac Sim 4.1.0] orbit -> lab
# [Old - Isaac Sim 2023.1.0] from omni.isaac.core.prims import XFormPrim,XFormPrimView
from omni.isaac.core.prims import XFormPrim  # [New - Isaac Sim 4.1.0] XFormPrimView 已移除，平台使用预计算位置
from pxr import UsdGeom, Gf
import omni.usd

class LandingEnv(IsaacEnv):

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        self.lidar_stack_T = cfg.sensor.lidar_stack_T

        self.init_time = time.time()

        self.alpha = cfg.reward.safety_static_alpha

        super().__init__(cfg, cfg.headless)
        self.elapsed_time = 0.0
        self.platform_world_pos = self.envs_positions + self.platform_center_local.view(1, 3)
        self.platform_top_pos = self.platform_world_pos + self.platform_top_offset.view(1, 3)
        reward_cfg = getattr(cfg, "reward", None)
        self.soft_speed_xy = float(getattr(reward_cfg, "soft_speed_xy", 1.2))
        self.soft_speed_vz_up = float(getattr(reward_cfg, "soft_speed_vz_up", 0.8))
        self.soft_speed_vz_down = float(getattr(reward_cfg, "soft_speed_vz_down", 0.9))
        self.hard_speed_xy = float(getattr(reward_cfg, "hard_speed_xy", 2.5))
        self.hard_speed_vz_up = float(getattr(reward_cfg, "hard_speed_vz_up", 1.5))
        self.hard_speed_vz_down = float(getattr(reward_cfg, "hard_speed_vz_down", 2.2))
        self.contact_force_threshold = float(
            getattr(reward_cfg, "contact_force_threshold", 1.0)
        )
        self.hard_contact_force_threshold = float(
            getattr(reward_cfg, "hard_contact_force_threshold", 20.0)
        )
        self.soft_contact_dz_abs = float(getattr(reward_cfg, "soft_contact_dz_abs", 0.22))
        self.soft_contact_xy_margin = float(
            getattr(reward_cfg, "soft_contact_xy_margin", 0.12)
        )
        self.soft_contact_vxy = float(getattr(reward_cfg, "soft_contact_vxy", 0.45))
        self.soft_contact_vz_down = float(
            getattr(reward_cfg, "soft_contact_vz_down", 0.45)
        )
        self.hard_landing_vxy = float(getattr(reward_cfg, "hard_landing_vxy", 0.8))
        self.hard_landing_vz_down = float(
            getattr(reward_cfg, "hard_landing_vz_down", 0.7)
        )
        self.enable_contact_force_tracking = bool(
            getattr(reward_cfg, "enable_contact_force_tracking", False)
        )
        self.contact_force_tracking_safe_env_limit = int(
            getattr(reward_cfg, "contact_force_tracking_safe_env_limit", 256)
        )
        self.contact_proxy_dz = float(getattr(reward_cfg, "contact_proxy_dz", 0.0))
        self.descent_align_radius = float(
            getattr(reward_cfg, "descent_align_radius", 0.35)
        )
        self.descent_align_gain = float(
            getattr(reward_cfg, "descent_align_gain", 8.0)
        )
        self.offcenter_descent_speed = float(
            getattr(reward_cfg, "offcenter_descent_speed", 0.30)
        )
        self.cross_speed_soft = float(
            getattr(reward_cfg, "cross_speed_soft", 0.35)
        )
        self.height_progress_align_floor = float(
            getattr(reward_cfg, "height_progress_align_floor", 0.10)
        )
        self.flare_xy_radius = float(
            getattr(reward_cfg, "flare_xy_radius", 0.60)
        )
        self.flare_dz_abs = float(
            getattr(reward_cfg, "flare_dz_abs", 0.45)
        )
        self.flare_vxy_soft = float(
            getattr(reward_cfg, "flare_vxy_soft", 0.45)
        )
        self.flare_vz_down_soft = float(
            getattr(reward_cfg, "flare_vz_down_soft", 0.55)
        )
        self.flare_vxy_hard = float(
            getattr(reward_cfg, "flare_vxy_hard", 1.20)
        )
        self.flare_vz_down_hard = float(
            getattr(reward_cfg, "flare_vz_down_hard", 1.40)
        )
        self.hover_target_height = float(
            getattr(reward_cfg, "hover_target_height", 0.80)
        )
        self.hover_xy_radius = float(
            getattr(reward_cfg, "hover_xy_radius", self.platform_size_xy[0] * 0.35)
        )
        self.hover_height_tolerance = float(
            getattr(reward_cfg, "hover_height_tolerance", 0.20)
        )
        self.hover_vxy_soft = float(
            getattr(reward_cfg, "hover_vxy_soft", 0.22)
        )
        self.hover_vz_soft = float(
            getattr(reward_cfg, "hover_vz_soft", 0.18)
        )
        self.hover_vxy_hard = float(
            getattr(reward_cfg, "hover_vxy_hard", 0.45)
        )
        self.hover_vz_hard = float(
            getattr(reward_cfg, "hover_vz_hard", 0.35)
        )
        self.hover_hold_steps = int(
            getattr(
                reward_cfg,
                "hover_hold_steps",
                getattr(reward_cfg, "landing_hold_steps", 10),
            )
        )
        self.hover_target_offset = torch.tensor(
            [0.0, 0.0, self.hover_target_height],
            device=self.device,
            dtype=torch.float,
        )
        self.helipad_center = (
            self.platform_top_pos[0] + self.hover_target_offset
        ).view(1, 3)
        if (
            self.enable_contact_force_tracking
            and self.num_envs > self.contact_force_tracking_safe_env_limit
        ):
            print(
                "[LandingEnv] disabling contact-force tracking for "
                f"{self.num_envs} envs; using geometric contact proxy to avoid "
                "known PhysX/Fabric CUDA instability."
            )
            self.enable_contact_force_tracking = False
        # Drone Initialization
        self.drone.initialize(
            track_contact_forces=self.enable_contact_force_tracking
        )
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # start and target 
        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.target_dir = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.height_range = torch.zeros(self.num_envs, 1, 2, device=self.device)
        self.prev_drone_vel_w = torch.zeros(self.num_envs, 1, 3, device=self.device) 
        self.last_applied_motor_cmd = torch.zeros(
            self.num_envs,
            self.drone.n,
            self.drone.num_rotors,
            device=self.device,
        )
        # [New - 静态降落奖励修复]
        # 记录上一时刻与平台顶面的垂直误差，用来奖励“持续下降”的真实进展。
        # 这样策略不会因为待在平台正上方就一直拿到稳定正奖励。
        self.prev_height_error = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_horizontal_error = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_distance_error = torch.zeros(self.num_envs, 1, device=self.device)
        self.landing_hold_counter = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device
        )
        self.prev_contact = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device
        )

    def _set_drone_root_poses_gpu(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        env_ids: torch.Tensor,
    ):
        # Use the PhysX tensor API directly to avoid falling back to the old
        # setGlobalPose path when GPU direct API is enabled.
        drone_view = getattr(self.drone, "_view", None)
        physics_view = getattr(drone_view, "_physics_view", None) if drone_view is not None else None
        if drone_view is None or physics_view is None or not hasattr(drone_view, "_resolve_env_indices"):
            self.drone.set_world_poses(positions, orientations, env_ids)
            return

        indices = drone_view._resolve_env_indices(env_ids)
        poses = physics_view.get_root_transforms()
        poses[indices, :3] = positions.reshape(-1, 3).to(poses.device)
        # Project quaternion order is (w, x, y, z), while PhysX expects (x, y, z, w).
        poses[indices, 3:] = orientations.reshape(-1, 4)[:, [1, 2, 3, 0]].to(poses.device)
        physics_view.set_root_transforms(poses, indices)

    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0

        # 平台参数（必须先定义再使用）
        self.platform_size_xy = (0.8, 0.8)          # 长宽
        self.platform_height = 0.10                 # 厚度
        self.platform_clearance = 0.20              # 距地高度（平台中心到地面）
        self.platform_center_local = torch.tensor(
            [0.0, 0.0, self.platform_height * 0.5 + self.platform_clearance],
            device=self.device,
            dtype=torch.float,
        )
        self.platform_amp_xy = torch.tensor([3.0, 2.0, 0.0], device=self.device)   # 轨迹振幅
        self.platform_vel_xy = torch.tensor([0.25, 0.30, 0.0], device=self.device) # 角速度
        self.platform_phase = torch.zeros(self.num_envs, 2, device=self.device)    # 每env相位
        self.platform_top_offset = torch.tensor([0.0, 0.0, self.platform_height * 0.5], device=self.device)

        # 1. 无人机模型
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls()
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # 2. 灯光
        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # 3. 创建地面（大区域）
        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        # 只在 env_0 模板中创建平台，其他 env 由 GridCloner 复制。
        # 这样平台与无人机使用同一套 env 偏移，不会和 cloner 的位置布局打架。
        env_spacing = float(getattr(self.cfg.env, "env_spacing", 10.0))
        cols = int(np.ceil(np.sqrt(self.num_envs)))
        rows = int(np.ceil(self.num_envs / cols))
        origin_path = "/World/envs/env_0/PlatformOrigin"
        prim_utils.create_prim(
            origin_path,
            "Xform",
            translation=tuple(self.platform_center_local.cpu().numpy()),
        )
        prim_path = f"{origin_path}/LandingPlatform"
        cuboid_cfg = sim_utils.CuboidCfg(
            size=[self.platform_size_xy[0], self.platform_size_xy[1], self.platform_height],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.8), metallic=0.1),
        )
        cuboid_cfg.func(prim_path, cuboid_cfg)

        self.map_range = [env_spacing * cols, env_spacing * rows, 4.5]
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        observation_dim = 12
        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    #"lidar":UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "final_horizontal_err": UnboundedContinuousTensorSpec(1),
            "final_dz_abs": UnboundedContinuousTensorSpec(1),
            "final_vxy": UnboundedContinuousTensorSpec(1),
            "final_vz_abs": UnboundedContinuousTensorSpec(1),
            "min_horizontal_err": UnboundedContinuousTensorSpec(1),
            "min_dz_abs": UnboundedContinuousTensorSpec(1),
            "inside_platform_once": UnboundedContinuousTensorSpec(1),
            "touchdown_once": UnboundedContinuousTensorSpec(1),
            "landing_hold_max": UnboundedContinuousTensorSpec(1),
            "hard_landing": UnboundedContinuousTensorSpec(1),
            "terminal_speed_violation": UnboundedContinuousTensorSpec(1),
            "unsafe_speed_xy": UnboundedContinuousTensorSpec(1),
            "unsafe_speed_z": UnboundedContinuousTensorSpec(1),
            "fail_too_far": UnboundedContinuousTensorSpec(1),
            "fail_above_bound": UnboundedContinuousTensorSpec(1),
            "fail_below_bound": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "applied_motor_cmd": UnboundedContinuousTensorSpec(
                (self.drone.n, self.drone.num_rotors), device=self.device
            ),
            "motor_throttle": UnboundedContinuousTensorSpec(
                (self.drone.n, self.drone.num_rotors), device=self.device
            ),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def reset_target(self, env_ids: torch.Tensor):
        # 第一阶段任务：目标为平台正上方的稳定悬停点，而不是平台顶面。
        platform_top = self.platform_top_pos[env_ids]
        hover_target = platform_top + self.hover_target_offset.view(1, 3)
        self.target_pos[env_ids] = hover_target.view(-1, 1, 3)


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)

        # 为每个 env 随机平台相位
        # self.platform_phase[env_ids, 0] = torch.rand(env_ids.size(0), device=self.device) * 2*np.pi
        # self.platform_phase[env_ids, 1] = torch.rand(env_ids.size(0), device=self.device) * 2*np.pi

        self.reset_target(env_ids)

        # 初始位置：在平台附近的随机水平偏移上方，而不是固定正上方。
        radius_range = getattr(self.cfg.env, "reset_horizontal_radius_range", [0.0, 0.0])
        radius_min = float(radius_range[0]) if len(radius_range) > 0 else 0.0
        radius_max = float(radius_range[1]) if len(radius_range) > 1 else radius_min
        if radius_max < radius_min:
            radius_min, radius_max = radius_max, radius_min
        spawn_h = float(getattr(self.cfg.env, "reset_spawn_height", 2.5))
        theta = torch.rand(env_ids.size(0), device=self.device) * 2*np.pi
        if radius_max > 0.0:
            u = torch.rand(env_ids.size(0), device=self.device)
            horizontal_radius = torch.sqrt(
                u * (radius_max ** 2 - radius_min ** 2) + radius_min ** 2
            )
        else:
            horizontal_radius = torch.zeros(env_ids.size(0), device=self.device)
        dx = horizontal_radius * torch.cos(theta)
        dy = horizontal_radius * torch.sin(theta)

        platform_pos = (
            self.current_platform_pos[env_ids]
            if hasattr(self, "current_platform_pos")
            else self.platform_world_pos[env_ids]
        )  # [k,3]
        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        pos[:, 0, 0] = platform_pos[:, 0] + dx
        pos[:, 0, 1] = platform_pos[:, 1] + dy
        pos[:, 0, 2] = platform_pos[:, 2] + spawn_h

        # 朝向平台
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self._set_drone_root_poses_gpu(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        if not hasattr(self, "_debug_reset_printed"):
            self._debug_reset_printed = True
            # [New - Isaac Sim 4.1.0] 使用预计算的平台位置（无需 View）
            plat_pos_w = self.platform_world_pos
            # 只打印第一个 env id
            eid = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
            print("[debug] env", eid)
            print("[debug] cloner env position             =", self.envs_positions[eid].detach().cpu().numpy())
            print("[debug] computed platform_pos (center) =", plat_pos_w[eid].detach().cpu().numpy())
            print("[debug] platform world_pos              =", plat_pos_w[eid].detach().cpu().numpy())
            print("[debug] drone reset world_pos           =", pos[0, 0].detach().cpu().numpy())

        # 高度区间：平台顶面与初始高度之间
        self.prev_drone_vel_w[env_ids] = 0.
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        init_horizontal_err = horizontal_radius.unsqueeze(-1)
        init_dz_abs = (
            self.target_pos[env_ids, 0, 2] - pos[:, 0, 2]
        ).abs().unsqueeze(-1)
        init_distance_3d = torch.sqrt(init_horizontal_err.square() + init_dz_abs.square())
        # [New - 静态降落奖励修复]
        # reset 时把上一时刻高度误差初始化为当前 dz，后续用“误差是否变小”来奖励下降进展。
        self.prev_height_error[env_ids] = (
            self.target_pos[env_ids, 0, 2] - pos[:, 0, 2]
        ).abs().unsqueeze(-1)
        self.prev_horizontal_error[env_ids] = init_horizontal_err
        self.prev_distance_error[env_ids] = init_distance_3d
        self.landing_hold_counter[env_ids] = 0
        self.prev_contact[env_ids] = False
        self.last_applied_motor_cmd[env_ids] = 0.0

        # 清理缓冲
        self.stats[env_ids] = 0.
        self.stats["final_horizontal_err"][env_ids] = init_horizontal_err
        self.stats["final_dz_abs"][env_ids] = init_dz_abs
        self.stats["min_horizontal_err"][env_ids] = init_horizontal_err
        self.stats["min_dz_abs"][env_ids] = init_dz_abs
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")] 
        self.last_applied_motor_cmd[:] = actions.detach()
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.elapsed_time += float(self.dt)
        return
    
    # get current states/observation
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False) # (world_pos, orientation (quat), world_vel_and_angular, heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller
        self.info["applied_motor_cmd"][:] = self.last_applied_motor_cmd
        self.info["motor_throttle"][:] = self.drone.throttle

        # 相对平台状态
        target_hover = self.target_pos[:, 0, :]               # [N, 3]
        platform_top = self.platform_top_pos                  # [N, 3]
        pos_w = self.root_state[:, 0, :3]                     # [N, 3]
        vel_w = self.root_state[:, 0, 7:10]                   # [N, 3]

        rpos = target_hover - pos_w                           # [N, 3]
        platform_rpos = platform_top - pos_w                  # [N, 3]
        horizontal_err = rpos[:, :2].norm(dim=-1, keepdim=True)  # [N, 1]
        dz = rpos[:, 2].unsqueeze(-1)                         # [N, 1]
        platform_dz = platform_rpos[:, 2].unsqueeze(-1)       # [N, 1]

        # 关键：把 dx, dy 明确放进观测
        dx_dy = rpos[:, :2]                                   # [N, 2]
        rpos_xy_n = dx_dy / horizontal_err.clamp_min(1e-6)     # [N, 2]
        rpos_xy_perp = torch.cat(
            [-rpos_xy_n[:, 1:2], rpos_xy_n[:, 0:1]], dim=-1
        )

        vel_xy = vel_w[:, :2]                                  # [N,2]
        vx = vel_xy[:, 0:1]
        vy = vel_xy[:, 1:2]
        vz = vel_w[:, 2].unsqueeze(-1)                        # [N, 1]
        vxy = vel_xy.norm(dim=-1, keepdim=True)               # [N, 1]
        dz_abs = dz.abs()                                     # [N, 1]
        toward_speed = (vel_xy * rpos_xy_n).sum(dim=-1, keepdim=True)
        cross_speed = (vel_xy * rpos_xy_perp).sum(dim=-1, keepdim=True)

        distance_3d = rpos.norm(dim=-1, keepdim=True)          # [N,1]
        rpos_n = rpos / distance_3d.clamp_min(1e-6)
        toward_speed_3d = (vel_w * rpos_n).sum(dim=-1, keepdim=True)
        drone_state = torch.cat(
            [
                horizontal_err,
                dz,
                dx_dy,
                rpos_xy_n,
                vx,
                vy,
                vz,
                toward_speed,
                cross_speed,
                distance_3d,
            ],
            dim=-1,
        )  # [N, 12]

        obs = {"state": drone_state}

        # ---- reward shaping ----
        prev_vel_w = self.prev_drone_vel_w[:, 0, :]
        prev_vxy = prev_vel_w[:, :2].norm(dim=-1, keepdim=True)
        prev_vz = prev_vel_w[:, 2].unsqueeze(-1)

        # 第一阶段只学“快速飞到平台上方并稳定悬停”：
        # 1. target_pos 直接定义成平台上方的悬停点；
        # 2. 奖励以到达 hover target 的 3D progress 为主；
        # 3. 接近目标后，再明显奖励低速稳定。
        reward_xy = torch.exp(-2.2 * horizontal_err)
        reward_z = torch.exp(-3.2 * dz_abs)
        reward_dist3d = torch.exp(-0.95 * distance_3d)

        reward_xy_progress = (
            self.prev_horizontal_error - horizontal_err
        ).clamp(min=-0.20, max=0.20)
        reward_height_progress = (
            self.prev_height_error - dz_abs
        ).clamp(min=-0.16, max=0.16)
        reward_distance_progress = (
            self.prev_distance_error - distance_3d
        ).clamp(min=-0.24, max=0.24)

        target_vz = (0.90 * dz).clamp(min=-0.60, max=0.60)
        reward_vertical_track = torch.exp(-5.0 * (vz - target_vz).pow(2))

        speed_3d = vel_w.norm(dim=-1, keepdim=True)
        reward_toward = torch.tanh(1.2 * torch.relu(toward_speed_3d))
        move_gate = (speed_3d > 0.03).float()
        reward_direction = move_gate * torch.relu(
            toward_speed_3d / speed_3d.clamp_min(1e-4)
        )
        penalty_away = torch.relu(-toward_speed_3d)
        penalty_cross_speed = torch.relu(
            cross_speed.abs() - self.cross_speed_soft
        ).square()
        vertical_closing_speed = torch.sign(dz) * vz
        reward_vertical_closing = torch.tanh(1.8 * torch.relu(vertical_closing_speed))
        penalty_wrong_vertical = torch.relu(-vertical_closing_speed)
        penalty_above_target = torch.relu(-dz - self.hover_height_tolerance)

        speed_xy_excess = torch.relu(vxy - self.soft_speed_xy)
        soft_speed_vz = max(self.soft_speed_vz_up, self.soft_speed_vz_down)
        hard_speed_vz = max(self.hard_speed_vz_up, self.hard_speed_vz_down)
        speed_vz_excess = torch.relu(vz.abs() - soft_speed_vz)

        ang_vel_w = self.root_state[:, 0, 10:13]
        heading_w = self.root_state[:, 0, 13:16]
        up_w = self.root_state[:, 0, 16:19]
        ang_speed = ang_vel_w.norm(dim=-1, keepdim=True)
        reward_upright = torch.square((up_w[:, 2:3] + 1.0) * 0.5)
        throttle_change = self.drone.throttle_difference.mean(dim=-1, keepdim=True)

        # 初始 reset 时机头已经朝向平台，这里只轻微鼓励“继续朝着平台飞”。
        heading_xy = heading_w[:, :2]
        heading_xy_n = heading_xy / heading_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        heading_alignment = 0.5 * (
            (heading_xy_n * rpos_xy_n).sum(dim=-1, keepdim=True) + 1.0
        )
        heading_active = (horizontal_err > 0.05).float()
        reward_heading = heading_active * heading_alignment

        hover_slowdown_gate = torch.sigmoid(
            (self.hover_xy_radius * 1.8 - horizontal_err) * 12.0
        ) * torch.sigmoid(
            (self.hover_height_tolerance * 2.2 - dz_abs) * 14.0
        )

        penalty_dist = 0.18 * horizontal_err
        penalty_speed_xy = speed_xy_excess.square() + 0.25 * speed_xy_excess
        penalty_speed_z = (
            speed_vz_excess.square()
            + 0.25 * speed_vz_excess
        )
        hover_vxy_excess = torch.relu(vxy - self.hover_vxy_soft)
        hover_vz_excess = torch.relu(vz.abs() - self.hover_vz_soft)
        penalty_near_vxy = hover_slowdown_gate * (
            hover_vxy_excess.square() + 0.20 * hover_vxy_excess
        )
        penalty_near_vz = hover_slowdown_gate * (
            hover_vz_excess.square() + 0.20 * hover_vz_excess
        )
        penalty_ang_speed = ang_speed.square()
        penalty_throttle_change = throttle_change
        unsafe_speed_xy = vxy > self.hard_speed_xy
        unsafe_speed_z = vz.abs() > hard_speed_vz
        terminal_speed_violation = (
            (horizontal_err < self.hover_xy_radius * 1.5)
            & (dz_abs < self.hover_height_tolerance * 1.5)
            & (
                (vxy > self.hover_vxy_hard)
                | (vz.abs() > self.hover_vz_hard)
            )
        )

        if self.enable_contact_force_tracking:
            contact_forces = self.drone.base_link.get_net_contact_forces()
            contact_force_mag = contact_forces.norm(dim=-1)
            contact = contact_force_mag > self.contact_force_threshold
        else:
            contact_force_mag = torch.zeros_like(vxy)
            # 不依赖接触力时，使用真实平台顶面而不是 hover target 判断是否触碰平台。
            contact = platform_dz > self.contact_proxy_dz
        contact_started = contact & (~self.prev_contact)
        pad_contact_radius = max(self.platform_size_xy) * 0.5 + self.soft_contact_xy_margin
        pad_contact_zone = (horizontal_err < pad_contact_radius) & (
            platform_dz.abs() < self.soft_contact_dz_abs
        )
        impact_vxy = torch.maximum(vxy, prev_vxy)
        impact_vz_down = torch.maximum(torch.relu(-vz), torch.relu(-prev_vz))
        soft_contact = (
            contact_started
            & pad_contact_zone
            & (impact_vxy < self.soft_contact_vxy)
            & (impact_vz_down < self.soft_contact_vz_down)
        )
        hard_landing = contact_started & pad_contact_zone & (
            (~soft_contact) | (contact_force_mag > self.hard_contact_force_threshold)
        )
        collision = contact_started & (~pad_contact_zone)

        # ---- hover-above-platform success ----
        horiz_ok = horizontal_err < self.hover_xy_radius
        height_ok = dz_abs < self.hover_height_tolerance
        vz_ok = vz.abs() < self.hover_vz_soft
        vxy_ok = vxy < self.hover_vxy_soft
        hover_zone = horiz_ok & height_ok & vz_ok & vxy_ok
        self.landing_hold_counter = torch.where(
            hover_zone,
            self.landing_hold_counter + 1,
            torch.zeros_like(self.landing_hold_counter),
        )
        reward_hover_zone = hover_zone.float()
        reward_hold = (
            self.landing_hold_counter.float() / max(float(self.hover_hold_steps), 1.0)
        ).clamp(max=1.0)

        # 每一步都付出时间代价，避免“慢慢磨到超时”也拿很高回报。
        step_penalty = torch.full_like(reward_height_progress, 0.08)

        self.reward = (
            0.95 * reward_xy +
            1.10 * reward_z +
            0.90 * reward_dist3d +
            4.60 * reward_xy_progress +
            3.40 * reward_height_progress +
            4.10 * reward_distance_progress +
            0.90 * reward_toward +
            0.45 * reward_direction +
            1.20 * reward_vertical_track +
            0.80 * reward_vertical_closing +
            1.20 * reward_hover_zone +
            2.40 * reward_hold +
            0.10 * reward_upright +
            0.05 * reward_heading
            - 1.25 * penalty_away
            - 0.12 * penalty_cross_speed
            - 0.95 * penalty_wrong_vertical
            - 0.40 * penalty_above_target
            - 0.42 * penalty_near_vxy
            - 0.48 * penalty_near_vz
            - 0.20 * penalty_speed_xy
            - 0.18 * penalty_speed_z
            - 0.08 * penalty_ang_speed
            - 0.04 * penalty_throttle_change
            - 0.22 * penalty_dist
            - step_penalty
        )

        reached_hover = self.landing_hold_counter >= self.hover_hold_steps
        self.reward[reached_hover] += 250.0

        # ---- termination: prevent "fly away" ----
        platform_top_z = platform_top[:, 2].unsqueeze(-1)  # [N,1]
        below_bound = (pos_w[:, 2].unsqueeze(-1) < (platform_top_z - 0.8))
        above_bound = (pos_w[:, 2].unsqueeze(-1) > (platform_top_z + 4.0))

        too_far = (horizontal_err > 4.0)
        fail = (
            too_far
            | below_bound
            | above_bound
            | soft_contact
            | collision
            | hard_landing
            | terminal_speed_violation
        )

        self.reward[fail] -= 15.0
        self.reward[too_far] -= 15.0
        self.reward[above_bound] -= 30.0
        self.reward[below_bound] -= 25.0
        self.reward[soft_contact] -= 20.0
        self.reward[collision] -= 80.0
        self.reward[hard_landing] -= 60.0
        self.reward[terminal_speed_violation] -= 40.0
        self.reward[unsafe_speed_xy] -= 12.0
        self.reward[unsafe_speed_z] -= 16.0

        self.terminated = fail | reached_hover
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.prev_height_error = dz_abs.detach()
        self.prev_horizontal_error = horizontal_err.detach()
        self.prev_distance_error = distance_3d.detach()
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()
        self.prev_contact = contact.detach()

        self.stats["final_horizontal_err"] = horizontal_err
        self.stats["final_dz_abs"] = dz_abs
        self.stats["final_vxy"] = vxy
        self.stats["final_vz_abs"] = vz.abs()
        self.stats["min_horizontal_err"] = torch.minimum(
            self.stats["min_horizontal_err"], horizontal_err
        )
        self.stats["min_dz_abs"] = torch.minimum(
            self.stats["min_dz_abs"], dz_abs
        )
        self.stats["inside_platform_once"] = torch.maximum(
            self.stats["inside_platform_once"], horiz_ok.float()
        )
        self.stats["touchdown_once"] = torch.maximum(
            self.stats["touchdown_once"], hover_zone.float()
        )
        self.stats["landing_hold_max"] = torch.maximum(
            self.stats["landing_hold_max"], self.landing_hold_counter.float()
        )
        self.stats["hard_landing"] = torch.maximum(
            self.stats["hard_landing"], hard_landing.float()
        )
        self.stats["terminal_speed_violation"] = torch.maximum(
            self.stats["terminal_speed_violation"], terminal_speed_violation.float()
        )
        self.stats["unsafe_speed_xy"] = torch.maximum(
            self.stats["unsafe_speed_xy"], unsafe_speed_xy.float()
        )
        self.stats["unsafe_speed_z"] = torch.maximum(
            self.stats["unsafe_speed_z"], unsafe_speed_z.float()
        )
        self.stats["fail_too_far"] = too_far.float()
        self.stats["fail_above_bound"] = above_bound.float()
        self.stats["fail_below_bound"] = below_bound.float()
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reached_hover.float()
        self.stats["collision"] = torch.maximum(
            self.stats["collision"], collision.float()
        )
        self.stats["truncated"] = self.truncated.float()

        return TensorDict({
            "agents": TensorDict({"observation": obs}, [self.num_envs]),
            "stats": self.stats.clone(),
            "info": self.info.clone(),
        }, self.batch_size)

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
