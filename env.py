import torch
import einops
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from tools import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time
from omni_drones.utils.torch import quat_rotate_inverse
from omni.isaac.orbit.sensors import Camera, CameraCfg
from omni.isaac.core.prims import XFormPrim
from pxr import UsdGeom, Gf
import omni.usd

class LandingEnv(IsaacEnv):

    def __init__(self, cfg):
        print("[LandRL_v2]: Initializing Landing Environment...")
        # LiDAR params (kept for compatibility)
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

        # ===== 从 cfg.reward 读取着陆参数 =====
        reward_cfg = getattr(cfg, "reward", None)

        # 着陆检测阈值
        self.landing_dz_threshold = float(getattr(reward_cfg, "landing_dz_threshold", 0.08))
        self.landing_below_threshold = float(getattr(reward_cfg, "landing_below_threshold", 0.03))
        self.landing_xy_margin = float(getattr(reward_cfg, "landing_xy_margin", 0.35))
        self.landing_vxy_soft = float(getattr(reward_cfg, "landing_vxy_soft", 0.25))
        self.landing_vz_soft = float(getattr(reward_cfg, "landing_vz_soft", 0.20))
        self.landing_hold_steps = int(getattr(reward_cfg, "landing_hold_steps", 10))

        # 下降对齐
        self.descent_align_radius = float(getattr(reward_cfg, "descent_align_radius", 0.45))
        self.speed_proximity_height = float(getattr(reward_cfg, "speed_proximity_height", 1.5))

        # 近地速度限制
        self.speed_vxy_limit = float(getattr(reward_cfg, "speed_vxy_limit", 0.4))
        self.speed_vz_limit = float(getattr(reward_cfg, "speed_vz_limit", 0.5))

        # 触地区软着陆 shaping（用于压低“先猛冲再稳住”的错误策略）
        self.touchdown_zone_height = float(
            getattr(reward_cfg, "touchdown_zone_height", 0.18)
        )
        self.touchdown_vxy_target = float(
            getattr(reward_cfg, "touchdown_vxy_target", self.landing_vxy_soft)
        )
        self.touchdown_vz_target = float(
            getattr(reward_cfg, "touchdown_vz_target", self.landing_vz_soft)
        )

        # 硬着陆两级制阈值
        self.hard_landing_vxy = float(getattr(reward_cfg, "hard_landing_vxy", 1.0))
        self.hard_landing_vz_down = float(getattr(reward_cfg, "hard_landing_vz_down", 1.0))
        self.hard_landing_vxy_extreme = float(getattr(reward_cfg, "hard_landing_vxy_extreme", 2.5))
        self.hard_landing_vz_extreme = float(getattr(reward_cfg, "hard_landing_vz_extreme", 2.0))
        self.hard_landing_penalty_extreme = float(getattr(reward_cfg, "hard_landing_penalty_extreme", 20.0))

        # 奖励权重
        self.w_distance_progress = float(getattr(reward_cfg, "w_distance_progress", 5.0))
        self.w_potential = float(getattr(reward_cfg, "w_potential", 1.5))
        self.w_descent = float(getattr(reward_cfg, "w_descent", 4.0))
        self.w_speed_shaping = float(getattr(reward_cfg, "w_speed_shaping", 3.0))
        self.w_touchdown_smooth = float(
            getattr(reward_cfg, "w_touchdown_smooth", 4.0)
        )
        self.w_landed = float(getattr(reward_cfg, "w_landed", 2.5))
        self.w_hold = float(getattr(reward_cfg, "w_hold", 3.0))
        self.success_bonus = float(getattr(reward_cfg, "success_bonus", 80.0))
        self.hard_landing_penalty = float(getattr(reward_cfg, "hard_landing_penalty", 8.0))
        self.flip_penalty = float(getattr(reward_cfg, "flip_penalty", 10.0))
        self.oob_penalty = float(getattr(reward_cfg, "oob_penalty", 5.0))

        # 时间激励
        self.w_time_penalty = float(getattr(reward_cfg, "w_time_penalty", 1.0))
        self.time_bonus_ratio = float(getattr(reward_cfg, "time_bonus_ratio", 0.3))

        # 速度约束
        self.soft_speed_xy = float(getattr(reward_cfg, "soft_speed_xy", 1.2))
        self.hard_speed_xy = float(getattr(reward_cfg, "hard_speed_xy", 2.5))

        # ===== 阶段A1v3：全程抗抖动（物理加速度 + 动作平滑）=====
        # 全程物理加速度惩罚：惩罚 ||vel_w - prev_vel_w||
        self.w_accel_penalty = float(getattr(reward_cfg, "w_accel_penalty", 0.0))
        # 全程动作平滑惩罚：惩罚策略输出帧间跳变 ||action_t - action_{t-1}||
        self.w_action_smooth = float(getattr(reward_cfg, "w_action_smooth", 0.0))
        # 角速度惩罚：pproximity 加权（近地更强）
        self.w_bodyrate = float(getattr(reward_cfg, "w_bodyrate", 0.0))
        # upright 姿态惩罚：pproximity 加权（近地更强）
        self.w_upright = float(getattr(reward_cfg, "w_upright", 0.0))

        # 接触力（默认禁用）
        self.enable_contact_force_tracking = bool(
            getattr(reward_cfg, "enable_contact_force_tracking", False)
        )
        self.contact_force_tracking_safe_env_limit = int(
            getattr(reward_cfg, "contact_force_tracking_safe_env_limit", 256)
        )
        if (
            self.enable_contact_force_tracking
            and self.num_envs > self.contact_force_tracking_safe_env_limit
        ):
            print(
                f"[LandRL_v2] disabling contact-force tracking for "
                f"{self.num_envs} envs to avoid CUDA instability."
            )
            self.enable_contact_force_tracking = False

        # Drone Initialization
        self.drone.initialize(
            track_contact_forces=self.enable_contact_force_tracking
        )
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # Buffers
        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.target_dir = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.height_range = torch.zeros(self.num_envs, 1, 2, device=self.device)
        self.prev_drone_vel_w = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.last_applied_motor_cmd = torch.zeros(
            self.num_envs, self.drone.n, self.drone.num_rotors, device=self.device,
        )
        self.prev_height_above = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_horizontal_error = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_distance_error = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_potential = torch.zeros(self.num_envs, 1, device=self.device)
        self.landing_hold_counter = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device
        )
        self.prev_touchdown_zone = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device
        )
        # 阶段A1：缓冲当前帧和上一帧的速度指令（由 VelocityEMATransform 写入 tensordict）
        self.current_raw_vel_cmd = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.prev_raw_vel_cmd = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # 阶段A1：平滑后速度帧间差（由 VelocityEMATransform 写入 tensordict）
        self.current_smoothed_vel_diff = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # 引用 VelocityEMATransform（由 train.py 注入）
        self.vel_ema_transform = None

    # ---- GPU direct pose API ----
    def _set_drone_root_poses_gpu(self, positions, orientations, env_ids):
        drone_view = getattr(self.drone, "_view", None)
        physics_view = getattr(drone_view, "_physics_view", None) if drone_view is not None else None
        if drone_view is None or physics_view is None or not hasattr(drone_view, "_resolve_env_indices"):
            self.drone.set_world_poses(positions, orientations, env_ids)
            return
        indices = drone_view._resolve_env_indices(env_ids)
        poses = physics_view.get_root_transforms()
        poses[indices, :3] = positions.reshape(-1, 3).to(poses.device)
        poses[indices, 3:] = orientations.reshape(-1, 4)[:, [1, 2, 3, 0]].to(poses.device)
        physics_view.set_root_transforms(poses, indices)

    # ---- Scene Design ----
    def _design_scene(self):
        # 平台参数
        self.platform_size_xy = (0.8, 0.8)
        self.platform_height = 0.10
        self.platform_clearance = 0.20
        self.platform_center_local = torch.tensor(
            [0.0, 0.0, self.platform_height * 0.5 + self.platform_clearance],
            device=self.device, dtype=torch.float,
        )
        self.platform_amp_xy = torch.tensor([3.0, 2.0, 0.0], device=self.device)
        self.platform_vel_xy = torch.tensor([0.25, 0.30, 0.0], device=self.device)
        self.platform_phase = torch.zeros(self.num_envs, 2, device=self.device)
        self.platform_top_offset = torch.tensor(
            [0.0, 0.0, self.platform_height * 0.5], device=self.device
        )

        # 1. 无人机
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name]
        cfg = drone_model.cfg_cls()
        self.drone = drone_model(cfg=cfg)
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # 2. 灯光
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

        # 3. 地面
        # 使用本地几何体地面，避免 GroundPlaneCfg 依赖远程 default_environment.usd。
        # 之前在无网络/资产不可达时会报：Could not open asset ... default_environment.usd，
        # 随后 prim_utils.get_prim_path(None) 触发 AttributeError。
        cfg_ground = sim_utils.CuboidCfg(
            size=(300.0, 300.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        )
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, -0.01))

        # 4. 着陆平台（env_0 模板，GridCloner 复制）
        env_spacing = float(getattr(self.cfg.env, "env_spacing", 10.0))
        cols = int(np.ceil(np.sqrt(self.num_envs)))
        rows = int(np.ceil(self.num_envs / cols))
        origin_path = "/World/envs/env_0/PlatformOrigin"
        prim_utils.create_prim(
            origin_path, "Xform",
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

    # ---- Specs ----
    def _set_specs(self):
        # 9D 观测：rpos(3) + vel(3) + up_vector(3)
        observation_dim = 9
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec,
            })
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

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
            "flip": UnboundedContinuousTensorSpec(1),
            "terminal_speed_violation": UnboundedContinuousTensorSpec(1),
            "unsafe_speed_xy": UnboundedContinuousTensorSpec(1),
            "unsafe_speed_z": UnboundedContinuousTensorSpec(1),
            "fail_too_far": UnboundedContinuousTensorSpec(1),
            "fail_above_bound": UnboundedContinuousTensorSpec(1),
            "fail_below_bound": UnboundedContinuousTensorSpec(1),
            "fail_flip": UnboundedContinuousTensorSpec(1),
            "fail_hard_landing": UnboundedContinuousTensorSpec(1),
            # A1v3：抖动监控指标（EMA 滑动均值）
            "ema_ang_vel_norm": UnboundedContinuousTensorSpec(1),
            "ema_accel_norm": UnboundedContinuousTensorSpec(1),
            "ema_action_diff": UnboundedContinuousTensorSpec(1),
            "ema_tilt": UnboundedContinuousTensorSpec(1),
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

    # ---- Target: 平台顶面（不是悬停点）----
    def reset_target(self, env_ids: torch.Tensor):
        platform_top = self.platform_top_pos[env_ids]
        self.target_pos[env_ids] = platform_top.view(-1, 1, 3)

    # ---- Reset ----
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)

        # 初始位置：平台附近随机圆形分布
        radius_range = getattr(self.cfg.env, "reset_horizontal_radius_range", [0.0, 0.0])
        radius_min = float(radius_range[0]) if len(radius_range) > 0 else 0.0
        radius_max = float(radius_range[1]) if len(radius_range) > 1 else radius_min
        if radius_max < radius_min:
            radius_min, radius_max = radius_max, radius_min
        spawn_h = float(getattr(self.cfg.env, "reset_spawn_height", 2.5))
        theta = torch.rand(env_ids.size(0), device=self.device) * 2 * np.pi
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
        )
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
            plat_pos_w = self.platform_world_pos
            eid = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
            print("[debug] env", eid)
            print("[debug] cloner env position             =", self.envs_positions[eid].detach().cpu().numpy())
            print("[debug] computed platform_pos (center) =", plat_pos_w[eid].detach().cpu().numpy())
            print("[debug] drone reset world_pos           =", pos[0, 0].detach().cpu().numpy())

        # 高度区间
        self.prev_drone_vel_w[env_ids] = 0.
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        init_horizontal_err = horizontal_radius.unsqueeze(-1)
        # height_above = drone_z - platform_top_z (正值=在上方)
        init_height_above = (pos[:, 0, 2] - self.platform_top_pos[env_ids, 2]).unsqueeze(-1)
        init_dz_abs = init_height_above.abs()
        init_distance_3d = torch.sqrt(init_horizontal_err.square() + init_dz_abs.square())

        self.prev_height_above[env_ids] = init_height_above
        self.prev_horizontal_error[env_ids] = init_horizontal_err
        self.prev_distance_error[env_ids] = init_distance_3d
        self.prev_potential[env_ids] = torch.exp(-1.5 * init_distance_3d)
        self.landing_hold_counter[env_ids] = 0
        self.prev_touchdown_zone[env_ids] = False
        self.last_applied_motor_cmd[env_ids] = 0.0
        # 阶段A1：清零速度指令缓冲 + 通知 EMA transform 重置
        self.current_raw_vel_cmd[env_ids] = 0.0
        self.prev_raw_vel_cmd[env_ids] = 0.0
        self.current_smoothed_vel_diff[env_ids] = 0.0
        if self.vel_ema_transform is not None:
            self.vel_ema_transform.reset_envs(env_ids)

        # 清理 stats
        self.stats[env_ids] = 0.
        self.stats["final_horizontal_err"][env_ids] = init_horizontal_err
        self.stats["final_dz_abs"][env_ids] = init_dz_abs
        self.stats["min_horizontal_err"][env_ids] = init_horizontal_err
        self.stats["min_dz_abs"][env_ids] = init_dz_abs

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.last_applied_motor_cmd[:] = actions.detach()
        # 阶段A1：从 VelocityEMATransform 写入的 tensordict 读取策略原始速度指令
        # 用于后续 _compute_state_and_obs 中计算速度平滑惩罚
        if ("info", "raw_vel_cmd") in tensordict.keys(True):
            self.current_raw_vel_cmd[:] = tensordict[("info", "raw_vel_cmd")]
        if ("info", "prev_raw_vel_cmd") in tensordict.keys(True):
            self.prev_raw_vel_cmd[:] = tensordict[("info", "prev_raw_vel_cmd")]
        # 阶段A1：读取平滑后速度帧间差（用于 ema_smoothed_vel_diff_mse 指标）
        if ("info", "smoothed_vel_diff") in tensordict.keys(True):
            self.current_smoothed_vel_diff[:] = tensordict[("info", "smoothed_vel_diff")]
        self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.elapsed_time += float(self.dt)
        return

    # ---- Core: 观测 + 奖励 + 终止 ----
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False)
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.info["applied_motor_cmd"][:] = self.last_applied_motor_cmd
        self.info["motor_throttle"][:] = self.drone.throttle

        # ---- 状态提取 ----
        platform_top = self.platform_top_pos                      # [N, 3]
        pos_w = self.root_state[:, 0, :3]                         # [N, 3]
        vel_w = self.root_state[:, 0, 7:10]                       # [N, 3]
        rot = self.root_state[:, 0, 3:7]                          # [N, 4] quaternion

        # up 向量：body z-axis in world frame
        up_vec = quat_axis(rot, axis=2)                           # [N, 3]

        # 相对位置：平台顶面 - 无人机（目标在上方时 dz > 0）
        rpos = platform_top - pos_w                               # [N, 3]
        dx_dy = rpos[:, :2]                                       # [N, 2]
        dz = rpos[:, 2].unsqueeze(-1)                             # [N, 1]

        # ---- 9D 观测 ----
        drone_state = torch.cat([
            dx_dy,                          # [N, 2] 水平相对位置
            dz,                             # [N, 1] 垂直相对位置
            vel_w[:, 0:1],                  # [N, 1] vx
            vel_w[:, 1:2],                  # [N, 1] vy
            vel_w[:, 2:3],                  # [N, 1] vz
            up_vec,                         # [N, 3] up vector
        ], dim=-1)                          # [N, 9]
        obs = {"state": drone_state}

        # ---- 奖励计算中间量 ----
        horizontal_err = dx_dy.norm(dim=-1, keepdim=True)         # [N, 1]
        dz_abs = dz.abs()                                         # [N, 1]
        distance_3d = rpos.norm(dim=-1, keepdim=True)             # [N, 1]
        vxy = vel_w[:, :2].norm(dim=-1, keepdim=True)             # [N, 1]
        vz = vel_w[:, 2:3]                                        # [N, 1]
        vz_down = (-vz).clamp(min=0)                              # [N, 1] 下降速度（正值）
        height_above = (pos_w[:, 2] - platform_top[:, 2]).unsqueeze(-1)  # [N, 1] 正=在上方
        up_z = up_vec[:, 2:3]                                     # [N, 1]

        # ===== 1. 接近奖励 =====
        reward_distance_progress = (
            self.prev_distance_error - distance_3d
        ).clamp(min=-0.3, max=0.3)

        current_potential = torch.exp(-1.5 * distance_3d)
        reward_potential = current_potential - self.prev_potential

        # ===== 2. 下降奖励：水平对准后鼓励主动下降 =====
        aligned = (horizontal_err < self.descent_align_radius)    # [N, 1] bool
        descent_delta = (
            self.prev_height_above - height_above
        ).clamp(min=-0.15, max=0.15)
        reward_descent = aligned.float() * descent_delta

        on_platform_xy = horizontal_err < self.landing_xy_margin

        # ===== 3. 近地减速：越接近平台越要慢 =====
        proximity = (1.0 - height_above / self.speed_proximity_height).clamp(0, 1)
        speed_excess = (
            (vxy - self.speed_vxy_limit).clamp(min=0)
            + (vz_down - self.speed_vz_limit).clamp(min=0)
        )
        reward_speed_shaping = -proximity * speed_excess

        # ===== 3.5 触地区软着陆 shaping =====
        # 只要进入平台上方的触地区，就开始更强地要求“减速后再落”。
        touchdown_zone = (
            on_platform_xy
            & (height_above < self.touchdown_zone_height)
            & (height_above > -self.landing_below_threshold)
        )
        touchdown_proximity = (
            1.0 - height_above / max(self.touchdown_zone_height, 1e-6)
        ).clamp(0, 1)
        touchdown_speed_excess = (
            (vxy - self.touchdown_vxy_target).clamp(min=0)
            / max(self.touchdown_vxy_target, 1e-6)
            + (vz_down - self.touchdown_vz_target).clamp(min=0)
            / max(self.touchdown_vz_target, 1e-6)
        )
        reward_touchdown_smooth = (
            -touchdown_zone.float() * touchdown_proximity * touchdown_speed_excess
        )

        # ===== 4. 着陆区奖励（几何代理法）=====
        near_surface = height_above < self.landing_dz_threshold
        above_surface = height_above > -self.landing_below_threshold
        low_vxy = vxy < self.landing_vxy_soft
        low_vz = vz.abs() < self.landing_vz_soft
        landed = on_platform_xy & near_surface & above_surface & low_vxy & low_vz

        self.landing_hold_counter = torch.where(
            landed,
            self.landing_hold_counter + 1,
            torch.zeros_like(self.landing_hold_counter),
        )
        reward_landed = landed.float()

        # ===== 5. 保持奖励 =====
        reward_hold = (
            self.landing_hold_counter.float() / max(float(self.landing_hold_steps), 1.0)
        ).clamp(max=1.0)

        # ===== 6. 全程物理加速度惩罚 =====
        # 惩罚无人机真实速度帧间变化量（物理加速度），每步全程生效
        phys_accel = (vel_w - self.prev_drone_vel_w.squeeze(1)).norm(dim=-1, keepdim=True)  # [N, 1]
        reward_accel_penalty = -phys_accel

        # ===== 6.5 全程动作平滑惩罚（A1v3）=====
        # 直接惩罚策略输出的帧间跳变，防止乒乓式振荡导致控制器反复调整姿态
        action_diff = (self.current_raw_vel_cmd.squeeze(1) - self.prev_raw_vel_cmd.squeeze(1))  # [N, 3]
        action_diff_norm = action_diff.norm(dim=-1, keepdim=True)  # [N, 1]
        reward_action_smooth = -action_diff_norm

        # ===== 7. 角速度惩罚（proximity 加权，近地更强）=====
        # 全程有基础惩罚，近地时通过 proximity 放大
        ang_vel = self.root_state[:, 0, 10:13]  # [N, 3] 角速度
        # proximity_bodyrate: 远处=0.2（基础），近地=1.0（最强）
        proximity_bodyrate = (0.2 + 0.8 * proximity)
        reward_bodyrate = -proximity_bodyrate * ang_vel.square().mean(dim=-1, keepdim=True)  # [N, 1]

        # ===== 8. upright 姿态惩罚（A1v2：proximity 加权，近地更强）=====
        # 全程有基础惩罚，近地时通过 proximity 放大
        proximity_upright = (0.2 + 0.8 * proximity)
        reward_upright = -proximity_upright * (1.0 - up_z).clamp(min=0)

        # ===== 总连续奖励 =====
        self.reward = (
            self.w_distance_progress * reward_distance_progress
            + self.w_potential * reward_potential
            + self.w_descent * reward_descent
            + self.w_speed_shaping * reward_speed_shaping  # reward_speed_shaping 已经是负值
            + self.w_touchdown_smooth * reward_touchdown_smooth
            + self.w_landed * reward_landed
            + self.w_hold * reward_hold
            + self.w_accel_penalty * reward_accel_penalty  # 全程物理加速度惩罚
            + self.w_action_smooth * reward_action_smooth  # A1v3：全程动作平滑惩罚
            + self.w_bodyrate * reward_bodyrate            # 角速度抑制（proximity 加权）
            + self.w_upright * reward_upright              # 姿态竖直（proximity 加权）
        )

        # ===== 时间惩罚（鼓励快速降落）=====
        self.reward += self.w_time_penalty * (-0.01)

        # ===== 离散奖惩 =====
        reached_goal = self.landing_hold_counter >= self.landing_hold_steps
        reached_goal_mask = reached_goal.squeeze(-1)
        # 成功奖励 + 时间效率奖励（越快完成额外奖励越多）
        time_ratio = (
            1.0
            - self.progress_buf[reached_goal_mask].unsqueeze(-1).float()
            / self.max_episode_length
        ).clamp(0, 1)
        self.reward[reached_goal_mask] += (
            self.success_bonus
            + self.success_bonus * self.time_bonus_ratio * time_ratio
        )

        # ---- 终止条件 ----
        platform_top_z = platform_top[:, 2].unsqueeze(-1)
        too_far = (horizontal_err > 4.0)
        above_bound = (pos_w[:, 2].unsqueeze(-1) > (platform_top_z + 4.0))
        below_bound = (pos_w[:, 2].unsqueeze(-1) < (platform_top_z - 0.8))
        flipped = (up_z < 0.0)

        # 硬着陆两级制：只在“进入触地区”的时刻判定，避免策略学到
        # “还没真正接地，只是近地高速掠过也算硬着陆” 这种错误语义。
        descending_into_touchdown_zone = touchdown_zone & (~self.prev_touchdown_zone) & (vz <= 0.0)

        # 硬着陆两级制：进入触地区 + 平台内 + 速度过快
        near_ground = (height_above < self.landing_dz_threshold) & above_surface

        # 中等硬着陆：惩罚但不终止（让策略从"差一点"中学习）
        moderate_speed = (vxy > self.hard_landing_vxy) | (vz_down > self.hard_landing_vz_down)
        moderate_hard = descending_into_touchdown_zone & near_ground & moderate_speed
        self.reward[moderate_hard] -= self.hard_landing_penalty  # -8.0

        # 极端硬着陆：终止 + 更大惩罚
        extreme_speed = (vxy > self.hard_landing_vxy_extreme) | (vz_down > self.hard_landing_vz_extreme)
        extreme_hard = descending_into_touchdown_zone & near_ground & extreme_speed
        self.reward[extreme_hard] -= self.hard_landing_penalty_extreme  # -20.0

        # 出界惩罚
        self.reward[too_far] -= self.oob_penalty
        self.reward[above_bound] -= self.oob_penalty
        self.reward[below_bound] -= self.oob_penalty
        self.reward[flipped] -= self.flip_penalty

        # 只有极端硬着陆才终止
        fail = too_far | below_bound | above_bound | flipped | extreme_hard
        self.terminated = fail | reached_goal
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # ---- 更新 prev 缓冲 ----
        self.prev_height_above = height_above.detach()
        self.prev_horizontal_error = horizontal_err.detach()
        self.prev_distance_error = distance_3d.detach()
        self.prev_potential = current_potential.detach()
        self.prev_touchdown_zone = touchdown_zone.detach()
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        # ---- Stats ----
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
            self.stats["inside_platform_once"], on_platform_xy.float()
        )
        self.stats["touchdown_once"] = torch.maximum(
            self.stats["touchdown_once"], landed.float()
        )
        self.stats["landing_hold_max"] = torch.maximum(
            self.stats["landing_hold_max"], self.landing_hold_counter.float()
        )
        self.stats["hard_landing"] = torch.maximum(
            self.stats["hard_landing"], moderate_hard.float()
        )
        self.stats["flip"] = torch.maximum(
            self.stats["flip"], flipped.float()
        )
        self.stats["terminal_speed_violation"][:] = 0.0
        self.stats["unsafe_speed_xy"][:] = 0.0
        self.stats["unsafe_speed_z"][:] = 0.0
        self.stats["fail_too_far"] = too_far.float()
        self.stats["fail_above_bound"] = above_bound.float()
        self.stats["fail_below_bound"] = below_bound.float()
        self.stats["fail_flip"] = flipped.float()
        self.stats["fail_hard_landing"] = extreme_hard.float()
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reached_goal.float()
        self.stats["collision"][:] = 0.0
        self.stats["truncated"] = self.truncated.float()
        # A1v3：抖动监控指标（EMA 滑动均值）
        self.stats["ema_ang_vel_norm"] = 0.9 * self.stats["ema_ang_vel_norm"] + 0.1 * ang_vel.norm(dim=-1, keepdim=True)
        self.stats["ema_accel_norm"] = 0.9 * self.stats["ema_accel_norm"] + 0.1 * phys_accel
        self.stats["ema_action_diff"] = 0.9 * self.stats["ema_action_diff"] + 0.1 * action_diff_norm
        self.stats["ema_tilt"] = 0.9 * self.stats["ema_tilt"] + 0.1 * (1.0 - up_z).clamp(min=0)

        return TensorDict({
            "agents": TensorDict({"observation": obs}, [self.num_envs]),
            "stats": self.stats.clone(),
            "info": self.info.clone(),
        }, self.batch_size)

    def _compute_reward_and_done(self):
        return TensorDict(
            {
                "agents": {"reward": self.reward},
                "done": self.terminated | self.truncated,
                "terminated": self.terminated,
                "truncated": self.truncated,
            },
            self.batch_size,
        )
