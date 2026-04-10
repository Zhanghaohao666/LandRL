import argparse
import json
import os
import sys
import traceback
from contextlib import contextmanager
from types import MethodType
from typing import Dict

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torchrl.envs.transforms import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(PROJECT_DIR, "cfg")
ISAAC_SIM_DIR = os.path.abspath(
    os.path.join(PROJECT_DIR, "..", "nvidia", "isaac-sim")
)
ISAACLAB_APP_DIR = os.path.abspath(
    os.path.join(PROJECT_DIR, "..", "IsaacLab", "source", "apps")
)
ISAACLAB_EXT_DIR = os.path.abspath(
    os.path.join(PROJECT_DIR, "..", "IsaacLab", "source", "extensions")
)
RUNTIME_CACHE_DIR = os.path.join(PROJECT_DIR, ".runtime_cache")
RUNTIME_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "tmp")


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a PPO checkpoint and record an evaluation video."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint_xxx.pt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path. Defaults to <checkpoint_dir>/recordings/<checkpoint_name>.mp4",
    )
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Output json path. Defaults to <checkpoint_dir>/recordings/<checkpoint_name>.json",
    )
    parser.add_argument(
        "--trace-output",
        default=None,
        help="Output trace json path. Defaults to <checkpoint_dir>/recordings/<checkpoint_name>_trace.json",
    )
    parser.add_argument(
        "--details-output",
        default=None,
        help=(
            "Output detailed per-environment eval json path. "
            "Defaults to <checkpoint_dir>/recordings/<checkpoint_name>_details.json"
        ),
    )
    parser.add_argument(
        "--log-video",
        type=parse_bool,
        default=True,
        help="Whether to capture frames and write video/trace outputs. Default: true",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments for replay. Use 1 for the clearest video.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum rollout length. Defaults to cfg.env.max_episode_length.",
    )
    parser.add_argument(
        "--render-interval",
        type=int,
        default=2,
        help="Capture one frame every N simulation steps.",
    )
    parser.add_argument(
        "--headless",
        type=parse_bool,
        default=True,
        help="Whether to launch Isaac Sim without a window. Default: true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Replay seed. Defaults to cfg.seed.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device, for example cuda:0.",
    )
    parser.add_argument(
        "--exploration-type",
        choices=("mode", "mean", "random"),
        default="mode",
        help="Policy exploration mode during replay.",
    )
    parser.add_argument(
        "--camera-follow-drone",
        type=parse_bool,
        default=False,
        help="Whether to keep the replay camera centered on env-0 drone. Recommended for single-env videos.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("fixed", "follow_drone", "adaptive_landing"),
        default=None,
        help=(
            "Camera behavior for replay videos: "
            "'fixed' keeps a static close-up around the landing area, "
            "'follow_drone' locks the camera to env-0 drone, "
            "'adaptive_landing' keeps the landing area in view and automatically zooms out as the drone escapes."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra overrides, for example viewer.resolution=[1280,720].",
    )
    parser.add_argument(
        "--constant-action",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "VZ"),
        default=None,
        help=(
            "Bypass the policy network and replay a constant target velocity "
            "action in m/s, for example --constant-action 0 0 -0.001."
        ),
    )
    parser.add_argument(
        "--constant-motor-cmd",
        nargs=4,
        type=float,
        metavar=("M0", "M1", "M2", "M3"),
        default=None,
        help=(
            "Bypass VelController and the policy network, and directly replay a "
            "constant 4-rotor raw command in [-1, 1]."
        ),
    )
    parser.add_argument(
        "--legacy-force-path",
        type=parse_bool,
        default=False,
        help=(
            "For debugging only: monkey-patch the drone to use the legacy "
            "RigidPrimView force application path instead of the articulation "
            "physics-view path."
        ),
    )
    return parser.parse_args()


def prepend_env_path(key: str, path: str) -> None:
    current = os.environ.get(key, "")
    parts = [p for p in current.split(":") if p]
    if path not in parts:
        os.environ[key] = f"{path}:{current}" if current else path


def bootstrap_isaac_sim() -> None:
    if not os.path.isdir(ISAAC_SIM_DIR):
        raise FileNotFoundError(f"Isaac Sim directory not found: {ISAAC_SIM_DIR}")

    os.makedirs(RUNTIME_TMP_DIR, exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = RUNTIME_TMP_DIR
    os.environ.setdefault("CUDA_CACHE_PATH", os.path.join(RUNTIME_CACHE_DIR, "cuda"))
    os.environ.setdefault("OPTIX_CACHE_PATH", os.path.join(RUNTIME_CACHE_DIR, "optix"))
    os.makedirs(os.environ["CUDA_CACHE_PATH"], exist_ok=True)
    os.makedirs(os.environ["OPTIX_CACHE_PATH"], exist_ok=True)

    os.environ.setdefault("ISAAC_PATH", ISAAC_SIM_DIR)
    os.environ.setdefault("EXP_PATH", os.path.join(ISAAC_SIM_DIR, "apps"))
    os.environ.setdefault("CARB_APP_PATH", os.path.join(ISAAC_SIM_DIR, "kit"))

    # Allow direct execution with the user's Python, even when
    # `isaac_sim_activate.sh` has not been sourced in the shell.
    extra_sys_paths = [
        os.path.join(ISAAC_SIM_DIR, "python_packages"),
        os.path.join(ISAAC_SIM_DIR, "kit", "python", "lib", "python3.10", "site-packages"),
        os.path.join(ISAAC_SIM_DIR, "exts", "omni.isaac.kit"),
        os.path.join(ISAAC_SIM_DIR, "exts", "omni.isaac.gym"),
        os.path.join(ISAAC_SIM_DIR, "kit", "kernel", "py"),
        os.path.join(ISAAC_SIM_DIR, "kit", "plugins", "bindings-python"),
    ]
    for path in reversed(extra_sys_paths):
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            prepend_env_path("PYTHONPATH", path)

    local_paths = [
        os.path.abspath(os.path.join(PROJECT_DIR, "..", "OmniDrones-main")),
        os.path.join(ISAACLAB_EXT_DIR, "omni.isaac.lab"),
        os.path.join(ISAACLAB_EXT_DIR, "omni.isaac.lab_assets"),
        os.path.join(ISAACLAB_EXT_DIR, "omni.isaac.lab_tasks"),
    ]
    for path in local_paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
            prepend_env_path("PYTHONPATH", path)


def resolve_sim_experience(headless: bool) -> str:
    experience_name = (
        "isaaclab.python.headless.rendering.kit"
        if headless
        else "isaaclab.python.rendering.kit"
    )
    experience_path = os.path.join(ISAACLAB_APP_DIR, experience_name)
    if os.path.isfile(experience_path):
        print(f"[record] loading Isaac Lab experience: {experience_path}")
        return experience_path

    print(
        "[record][warn] Isaac Lab experience file not found, fallback to SimulationApp default experience: "
        f"{experience_path}"
    )
    return ""


def configure_gpu_pipeline_compatibility(cfg) -> None:
    import carb

    settings = carb.settings.get_settings()
    settings.set_bool("/isaaclab/render/offscreen", bool(cfg.headless))
    settings.set_bool("/isaaclab/render/rtx_sensors", False)
    print(f"[record] Isaac Lab render flags applied. offscreen={bool(cfg.headless)}")

    if not getattr(cfg.sim, "use_gpu_pipeline", False):
        # [New - 非 GPU pipeline + 非 fabric 的离屏录制]
        # Isaac Lab 的 headless rendering experience 默认把 updateToUsd 关掉。
        # 这对 fabric 路径没问题，因为 SimulationContext.render() 会手动 flush fabric。
        # 但如果我们为了稳定录视频而关闭了 use_fabric/use_flatcache，
        # 那就必须重新打开 updateToUsd，否则物理状态不会同步到渲染场景，
        # 录出来的视频就会像“永远停在 reset 画面”。
        if not getattr(cfg.sim, "use_fabric", getattr(cfg.sim, "use_flatcache", False)):
            usd_sync_settings = {
                "/physics/updateToUsd": True,
                "/physics/updateVelocitiesToUsd": True,
                "/physics/updateParticlesToUsd": False,
                "/physics/updateForceSensorsToUsd": False,
                "/physics/outputVelocitiesLocalSpace": False,
                "/physics/useFastCache": False,
                "/physics/fabricUpdateTransformations": False,
                "/physics/fabricUpdateVelocities": False,
                "/physics/fabricUpdateForceSensors": False,
                "/physics/fabricUpdateJointStates": False,
            }
            for key, value in usd_sync_settings.items():
                settings.set_bool(key, value)
            print("[record] USD sync enabled for non-fabric replay rendering.")
        return

    compat_settings = {
        "/physics/updateToUsd": False,
        "/physics/updateVelocitiesToUsd": False,
        "/physics/updateParticlesToUsd": False,
        "/physics/updateForceSensorsToUsd": False,
        "/physics/outputVelocitiesLocalSpace": False,
        "/physics/useFastCache": False,
        "/physics/fabricUpdateTransformations": False,
        "/physics/fabricUpdateVelocities": False,
        "/physics/fabricUpdateForceSensors": False,
        "/physics/fabricUpdateJointStates": False,
    }
    for key, value in compat_settings.items():
        settings.set_bool(key, value)

    print("[record] GPU pipeline compatibility settings applied for Isaac Lab rendering.")


def _cfg_optional_str(cfg_value):
    if cfg_value is None:
        return None
    value = str(cfg_value).strip()
    return value or None


def _parse_cuda_device_index(device: str | None) -> int | None:
    value = _cfg_optional_str(device)
    if value is None:
        return None
    normalized = value.lower()
    if normalized == "cpu":
        return None
    if not normalized.startswith("cuda"):
        return None
    if ":" not in normalized:
        return 0
    _, index = normalized.split(":", 1)
    try:
        return int(index)
    except ValueError:
        return None


def _build_simulation_app_config(cfg):
    app_config = {"headless": cfg.headless, "anti_aliasing": 1}
    sim_device_index = _parse_cuda_device_index(getattr(cfg.sim, "device", None))
    if sim_device_index is not None:
        app_config["active_gpu"] = sim_device_index
        app_config["physics_gpu"] = sim_device_index
        app_config["multi_gpu"] = False
        app_config["max_gpu_count"] = 1
        log_record(
            "[record] SimulationApp GPU routing: "
            f"active_gpu={sim_device_index}, physics_gpu={sim_device_index}, multi_gpu=False"
        )
    return app_config


def enable_required_isaac_extensions(sim_app) -> None:
    from omni.isaac.core.utils.extensions import enable_extension

    required_extensions = [
        "omni.isaac.debug_draw",
    ]
    for extension_name in required_extensions:
        enabled = enable_extension(extension_name)
        print(f"[record] enable_extension('{extension_name}') -> {enabled}")

    sim_app.update()


def build_output_paths(
    checkpoint_path: str,
    output_path: str | None,
    stats_output_path: str | None,
    trace_output_path: str | None,
    details_output_path: str | None,
    log_video: bool,
) -> tuple[str | None, str, str | None, str]:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    recording_dir = os.path.join(checkpoint_dir, "recordings")
    os.makedirs(recording_dir, exist_ok=True)

    if log_video and output_path is None:
        output_path = os.path.join(recording_dir, f"{checkpoint_stem}.mp4")
    if stats_output_path is None:
        stats_output_path = os.path.join(recording_dir, f"{checkpoint_stem}.json")
    if log_video and trace_output_path is None:
        trace_output_path = os.path.join(recording_dir, f"{checkpoint_stem}_trace.json")
    if details_output_path is None:
        details_output_path = os.path.join(recording_dir, f"{checkpoint_stem}_details.json")

    output_dir = os.path.dirname(output_path) if output_path else None
    stats_dir = os.path.dirname(stats_output_path)
    trace_dir = os.path.dirname(trace_output_path) if trace_output_path else None
    details_dir = os.path.dirname(details_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    if details_dir:
        os.makedirs(details_dir, exist_ok=True)
    return output_path, stats_output_path, trace_output_path, details_output_path


def load_cfg(args: argparse.Namespace):
    overrides = list(args.override)
    overrides.append(f"env.num_envs={args.num_envs}")
    if args.device is not None:
        overrides.append(f"device={args.device}")
        overrides.append(f"sim.device={args.device}")

    with hydra.initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = hydra.compose(config_name="train", overrides=overrides)

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.headless = args.headless
    if args.seed is not None:
        cfg.seed = args.seed

    # Isaac Sim 4.1 replay is more stable in non-GPU-pipeline mode when the
    # environment and policy tensors are also moved to CPU. Training keeps the
    # original CUDA path; this fallback only affects record_checkpoint.py unless
    # the user explicitly requests a device.
    if not getattr(cfg.sim, "use_gpu_pipeline", False) and args.device is None:
        cfg.device = "cpu"
        cfg.sim.device = "cpu"
        if hasattr(cfg.sim, "use_gpu"):
            cfg.sim.use_gpu = False

    # 单环境 headless 回放更看重“录到正确画面”而不是 Fabric 性能。
    # 如果用户没有手动指定 use_flatcache / use_fabric，默认把它们关掉，
    # 避免出现“物理状态在变，但离屏渲染画面几乎不更新”的同步问题。
    user_override_keys = {item.split('=', 1)[0] for item in overrides if '=' in item}
    user_specified_fabric = (
        "sim.use_flatcache" in user_override_keys or "sim.use_fabric" in user_override_keys
    )
    if (
        args.headless
        and args.num_envs == 1
        and not getattr(cfg.sim, "use_gpu_pipeline", False)
        and not user_specified_fabric
    ):
        if hasattr(cfg.sim, "use_flatcache"):
            cfg.sim.use_flatcache = False
        if hasattr(cfg.sim, "use_fabric"):
            cfg.sim.use_fabric = False
        cfg._record_force_disable_fabric = True
    else:
        cfg._record_force_disable_fabric = False
    return cfg


def select_eval_stats(trajs) -> Dict[str, float]:
    done = trajs.get(("next", "done")).squeeze(-1).cpu()
    num_steps = done.shape[1]
    first_done = torch.where(
        done.any(dim=1),
        done.long().argmax(dim=1),
        torch.full((done.shape[0],), num_steps - 1, dtype=torch.long),
    )

    def take_first_episode(tensor: torch.Tensor):
        tensor = tensor.cpu()
        # Gather the first terminal timestep for each environment.
        # For tensors shaped [N, T, ...], the indices passed to take_along_dim
        # must have the same rank: [N, 1, ...].
        indices = first_done.view(first_done.shape[0], *([1] * (tensor.ndim - 1)))
        if tensor.ndim > 2:
            indices = indices.expand(-1, 1, *tensor.shape[2:])
        return torch.take_along_dim(tensor, indices, dim=1).squeeze(1).reshape(-1)

    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }
    info = {
        "eval/stats." + k: torch.mean(v.float()).item()
        for k, v in traj_stats.items()
    }
    reach_goal = traj_stats.get("reach_goal")
    if reach_goal is not None:
        success_count = int((reach_goal > 0.5).sum().item())
        info["eval/summary.success_rate"] = float(reach_goal.float().mean().item())
        info["eval/summary.success_count"] = float(success_count)
        info["eval/summary.num_cases"] = float(reach_goal.numel())
    return info


def select_eval_case_details(trajs) -> Dict[str, object]:
    done = trajs.get(("next", "done")).squeeze(-1).cpu()
    num_envs, num_steps = done.shape
    first_done = torch.where(
        done.any(dim=1),
        done.long().argmax(dim=1),
        torch.full((done.shape[0],), num_steps - 1, dtype=torch.long),
    )

    def take_first_episode(tensor: torch.Tensor):
        tensor = tensor.cpu()
        indices = first_done.view(first_done.shape[0], *([1] * (tensor.ndim - 1)))
        if tensor.ndim > 2:
            indices = indices.expand(-1, 1, *tensor.shape[2:])
        return torch.take_along_dim(tensor, indices, dim=1).squeeze(1)

    obs_state = trajs.get(("next", "agents", "observation", "state"))
    if obs_state is None:
        raise KeyError("Missing ('next', 'agents', 'observation', 'state') in rollout tensordict.")
    obs_state = obs_state.detach().cpu()
    if obs_state.ndim == 4:
        obs_state = obs_state[:, :, 0, :]
    elif obs_state.ndim != 3:
        raise RuntimeError(
            f"Unexpected observation shape for details export: {tuple(obs_state.shape)}"
        )

    drone_state = trajs.get(("next", "info", "drone_state"))
    if drone_state is not None:
        drone_state = drone_state.detach().cpu()
        if drone_state.ndim == 4:
            drone_state = drone_state[:, :, 0, :]
        elif drone_state.ndim != 3:
            raise RuntimeError(
                f"Unexpected drone_state shape for details export: {tuple(drone_state.shape)}"
            )
        final_drone_state = take_first_episode(drone_state)
    else:
        final_drone_state = None

    final_done = take_first_episode(trajs.get(("next", "done")).cpu()).reshape(-1)
    final_terminated = take_first_episode(trajs.get(("next", "terminated")).cpu()).reshape(-1)
    final_truncated = take_first_episode(trajs.get(("next", "truncated")).cpu()).reshape(-1)
    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    cases = []
    for env_index in range(num_envs):
        start_obs = obs_state[env_index, 0]
        case = {
            "env_index": int(env_index),
            "done_step": int(first_done[env_index].item()),
            "done": bool(final_done[env_index].item()),
            "terminated": bool(final_terminated[env_index].item()),
            "truncated": bool(final_truncated[env_index].item()),
            "start_horizontal_err": float(start_obs[0].item()),
            "start_dz": float(start_obs[1].item()),
        }

        if final_drone_state is not None:
            case["start_world_pos"] = drone_state[env_index, 0, :3].tolist()
            case["final_world_pos"] = final_drone_state[env_index, :3].tolist()
            case["final_world_vel"] = final_drone_state[env_index, 7:10].tolist()

        for key, value in traj_stats.items():
            value_env = value[env_index]
            if value_env.numel() == 1:
                case[key] = float(value_env.item())
            else:
                case[key] = value_env.tolist()

        cases.append(case)

    aggregate = {
        "eval/stats." + k: torch.mean(v.float()).item()
        for k, v in traj_stats.items()
    }
    reach_goal = traj_stats.get("reach_goal")
    summary = {}
    if reach_goal is not None:
        success_count = int((reach_goal > 0.5).sum().item())
        summary = {
            "success_rate": float(reach_goal.float().mean().item()),
            "success_count": int(success_count),
            "num_cases": int(reach_goal.numel()),
        }
    return {
        "num_envs": int(num_envs),
        "num_steps": int(num_steps),
        "cases": cases,
        "aggregate_stats": aggregate,
        "summary": summary,
    }


def extract_first_env_trace(
    trajs, step_dt: float, target_top_z: float | None = None
) -> Dict[str, object]:
    actions = trajs.get(("agents", "action"))
    if actions is None:
        raise KeyError("Missing ('agents', 'action') in rollout tensordict.")

    obs_state = trajs.get(("next", "agents", "observation", "state"))
    if obs_state is None:
        raise KeyError("Missing ('next', 'agents', 'observation', 'state') in rollout tensordict.")

    actions = actions.detach().cpu()
    obs_state = obs_state.detach().cpu()

    first_env_actions = actions[0]
    first_env_obs = obs_state[0]

    if first_env_actions.ndim == 1:
        first_env_actions = first_env_actions.unsqueeze(0)
    if first_env_obs.ndim == 3:
        first_env_obs = first_env_obs[:, 0, :]
    elif first_env_obs.ndim != 2:
        raise RuntimeError(
            f"Unexpected observation shape for trace export: {tuple(first_env_obs.shape)}"
        )

    num_steps = min(first_env_actions.shape[0], first_env_obs.shape[0])
    first_env_actions = first_env_actions[:num_steps]
    first_env_obs = first_env_obs[:num_steps]

    horizontal_err = first_env_obs[:, 0]
    dz = first_env_obs[:, 1]
    vx = first_env_obs[:, 6]
    vy = first_env_obs[:, 7]
    vz = first_env_obs[:, 8]
    toward_speed = first_env_obs[:, 9]
    cross_speed = first_env_obs[:, 10]
    vxy = torch.sqrt(vx.square() + vy.square())
    rewards = trajs[("next", "agents", "reward")].detach().cpu()[0]
    if rewards.ndim > 1:
        rewards = rewards.squeeze(-1)

    step = torch.arange(num_steps, dtype=torch.long)
    trace = {
        "env_index": 0,
        "num_steps": int(num_steps),
        "step": step.tolist(),
        "time_sec": (step.float() * float(step_dt)).tolist(),
        "horizontal_err": horizontal_err.tolist(),
        "dz": dz.tolist(),
        "vx": vx.tolist(),
        "vy": vy.tolist(),
        "vxy": vxy.tolist(),
        "vz": vz.tolist(),
        "toward_speed": toward_speed.tolist(),
        "cross_speed": cross_speed.tolist(),
        "reward": rewards[:num_steps].tolist(),
        "action": first_env_actions.tolist(),
    }

    drone_state = trajs.get(("next", "info", "drone_state"))
    if drone_state is not None:
        first_env_drone_state = drone_state.detach().cpu()[0]
        if first_env_drone_state.ndim == 3:
            first_env_drone_state = first_env_drone_state[:num_steps, 0, :]
        elif first_env_drone_state.ndim != 2:
            raise RuntimeError(
                f"Unexpected drone_state shape for trace export: {tuple(first_env_drone_state.shape)}"
            )
        first_env_drone_state = first_env_drone_state[:num_steps]
        trace["world_pos"] = first_env_drone_state[:, :3].tolist()
        trace["world_vel"] = first_env_drone_state[:, 7:10].tolist()

    applied_motor_cmd = trajs.get(("next", "info", "applied_motor_cmd"))
    if applied_motor_cmd is not None:
        first_env_motor_cmd = applied_motor_cmd.detach().cpu()[0]
        if first_env_motor_cmd.ndim == 3:
            first_env_motor_cmd = first_env_motor_cmd[:num_steps, 0, :]
        elif first_env_motor_cmd.ndim != 2:
            raise RuntimeError(
                f"Unexpected applied_motor_cmd shape for trace export: {tuple(first_env_motor_cmd.shape)}"
            )
        trace["applied_motor_cmd"] = first_env_motor_cmd[:num_steps].tolist()

    motor_throttle = trajs.get(("next", "info", "motor_throttle"))
    if motor_throttle is not None:
        first_env_motor_throttle = motor_throttle.detach().cpu()[0]
        if first_env_motor_throttle.ndim == 3:
            first_env_motor_throttle = first_env_motor_throttle[:num_steps, 0, :]
        elif first_env_motor_throttle.ndim != 2:
            raise RuntimeError(
                f"Unexpected motor_throttle shape for trace export: {tuple(first_env_motor_throttle.shape)}"
            )
        trace["motor_throttle"] = first_env_motor_throttle[:num_steps].tolist()

    if target_top_z is not None:
        trace["target_top_z"] = float(target_top_z)
        trace["z"] = (float(target_top_z) - dz).tolist()

    for key_name, key in (
        ("done", ("next", "done")),
        ("terminated", ("next", "terminated")),
        ("truncated", ("next", "truncated")),
    ):
        value = trajs.get(key)
        if value is None:
            continue
        value = value.detach().cpu()[0]
        if value.ndim > 1:
            value = value.squeeze(-1)
        trace[key_name] = value[:num_steps].bool().tolist()

    return trace


def save_video(video_path: str, frames, fps: float) -> None:
    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def overlay_trace_on_frames(frames, trace: Dict[str, object], render_interval: int):
    if not frames:
        return frames

    num_steps = int(trace.get("num_steps", 0))
    if num_steps <= 0:
        return frames

    time_sec = trace.get("time_sec")
    horizontal_err = trace.get("horizontal_err")
    dz = trace.get("dz")
    vxy = trace.get("vxy")
    z = trace.get("z")
    vz = trace.get("vz")
    toward_speed = trace.get("toward_speed")
    cross_speed = trace.get("cross_speed")
    reward = trace.get("reward")
    action = trace.get("action")
    done = trace.get("done")
    terminated = trace.get("terminated")
    truncated = trace.get("truncated")
    world_pos = trace.get("world_pos")
    world_vel = trace.get("world_vel")
    target_top_z = trace.get("target_top_z")

    world_xy = None
    world_z = None
    inset_extent = 1.0
    inset_z_max = 1.0
    if world_pos is not None:
        world_xy = [(float(pos[0]), float(pos[1])) for pos in world_pos]
        world_z = [float(pos[2]) for pos in world_pos]
        xy_peak = max([max(abs(x), abs(y)) for x, y in world_xy] + [0.45])
        inset_extent = max(0.6, xy_peak * 1.25)
        inset_z_max = max(world_z + [float(target_top_z or 0.3), 0.3])

    overlay_frames = []
    for frame_idx, frame in enumerate(frames):
        step_idx = min(frame_idx * render_interval, num_steps - 1)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")

        lines = [
            f"step: {step_idx}",
            f"t: {float(time_sec[step_idx]):.2f}s" if time_sec is not None else None,
            f"z: {float(z[step_idx]):.3f}" if z is not None else None,
            f"dz: {float(dz[step_idx]):+.3f}" if dz is not None else None,
            f"vxy: {float(vxy[step_idx]):.3f}" if vxy is not None else None,
            f"vz: {float(vz[step_idx]):+.3f}" if vz is not None else None,
            (
                f"toward: {float(toward_speed[step_idx]):+.3f}"
                if toward_speed is not None
                else None
            ),
            (
                f"cross: {float(cross_speed[step_idx]):+.3f}"
                if cross_speed is not None
                else None
            ),
            f"reward: {float(reward[step_idx]):+.3f}" if reward is not None else None,
            (
                f"horizontal_err: {float(horizontal_err[step_idx]):.3f}"
                if horizontal_err is not None
                else None
            ),
        ]

        if world_pos is not None:
            pos = world_pos[step_idx]
            lines.append(
                f"world_pos: [{float(pos[0]):+.3f}, {float(pos[1]):+.3f}, {float(pos[2]):+.3f}]"
            )

        if world_vel is not None:
            vel = world_vel[step_idx]
            lines.append(
                f"world_vel: [{float(vel[0]):+.3f}, {float(vel[1]):+.3f}, {float(vel[2]):+.3f}]"
            )

        if action is not None:
            act = action[step_idx]
            act_str = ", ".join(f"{float(v):+.3f}" for v in act)
            lines.append(f"action: [{act_str}]")

        if done is not None:
            lines.append(
                "flags: "
                f"done={bool(done[step_idx])} "
                f"terminated={bool(terminated[step_idx]) if terminated is not None else False} "
                f"truncated={bool(truncated[step_idx]) if truncated is not None else False}"
            )

        lines = [line for line in lines if line is not None]

        padding = 16
        line_height = 20
        box_width = 470
        box_height = padding + line_height * len(lines) + 8
        draw.rounded_rectangle(
            (10, 10, 10 + box_width, 10 + box_height),
            radius=10,
            fill=(0, 0, 0, 150),
        )
        for i, line in enumerate(lines):
            draw.text(
                (padding, padding + i * line_height),
                line,
                fill=(255, 255, 255, 255),
            )

        if world_xy is not None and world_z is not None:
            width, height = image.size
            inset_w = 280
            inset_h = 220
            inset_margin = 18
            inset_x0 = width - inset_w - inset_margin
            inset_y0 = height - inset_h - inset_margin
            inset_x1 = width - inset_margin
            inset_y1 = height - inset_margin
            draw.rounded_rectangle(
                (inset_x0, inset_y0, inset_x1, inset_y1),
                radius=12,
                fill=(10, 16, 24, 210),
                outline=(180, 220, 255, 180),
                width=2,
            )
            draw.text(
                (inset_x0 + 12, inset_y0 + 10),
                "physics trace",
                fill=(235, 245, 255, 255),
            )

            map_x0 = inset_x0 + 12
            map_y0 = inset_y0 + 34
            map_x1 = inset_x1 - 54
            map_y1 = inset_y1 - 14
            map_w = map_x1 - map_x0
            map_h = map_y1 - map_y0

            def project_xy(x: float, y: float) -> tuple[float, float]:
                px = map_x0 + (x + inset_extent) / (2.0 * inset_extent) * map_w
                py = map_y1 - (y + inset_extent) / (2.0 * inset_extent) * map_h
                return px, py

            draw.rectangle(
                (map_x0, map_y0, map_x1, map_y1),
                outline=(120, 150, 180, 180),
                width=1,
            )

            cx, cy = project_xy(0.0, 0.0)
            draw.line([(cx - 8, cy), (cx + 8, cy)], fill=(120, 210, 255, 220), width=2)
            draw.line([(cx, cy - 8), (cx, cy + 8)], fill=(120, 210, 255, 220), width=2)

            pad_half = 0.4
            p0x, p0y = project_xy(-pad_half, -pad_half)
            p1x, p1y = project_xy(pad_half, pad_half)
            draw.rectangle(
                (p0x, p1y, p1x, p0y),
                outline=(120, 210, 255, 220),
                width=2,
            )

            history = world_xy[: step_idx + 1]
            if len(history) >= 2:
                traj_points = [project_xy(x, y) for x, y in history]
                draw.line(traj_points, fill=(255, 196, 72, 255), width=3)

            cur_x, cur_y = history[-1]
            cur_px, cur_py = project_xy(cur_x, cur_y)
            draw.ellipse(
                (cur_px - 5, cur_py - 5, cur_px + 5, cur_py + 5),
                fill=(255, 92, 92, 255),
                outline=(255, 230, 230, 255),
                width=1,
            )

            bar_x0 = inset_x1 - 34
            bar_x1 = inset_x1 - 20
            bar_y0 = map_y0
            bar_y1 = map_y1
            draw.rectangle(
                (bar_x0, bar_y0, bar_x1, bar_y1),
                outline=(120, 150, 180, 180),
                width=1,
            )
            z_norm = min(max(world_z[step_idx] / max(inset_z_max, 1e-6), 0.0), 1.0)
            fill_top = bar_y1 - z_norm * (bar_y1 - bar_y0)
            draw.rectangle(
                (bar_x0 + 2, fill_top, bar_x1 - 2, bar_y1 - 2),
                fill=(82, 196, 255, 220),
            )
            draw.text(
                (bar_x0 - 6, inset_y0 + 10),
                "z",
                fill=(235, 245, 255, 255),
            )
        overlay_frames.append(np.asarray(image))

    return overlay_frames


def log_record(message: str) -> None:
    print(message, flush=True)


def to_exploration_type(name: str) -> ExplorationType:
    if name == "mode":
        return ExplorationType.MODE
    if name == "mean":
        return ExplorationType.MEAN
    if name == "random":
        return ExplorationType.RANDOM
    raise ValueError(f"Unsupported exploration type: {name}")


class ConstantActionPolicy:
    def __init__(self, action, device: str):
        self.action = torch.as_tensor(action, dtype=torch.float32, device=device)

    def eval(self):
        return self

    def __call__(self, tensordict):
        obs = tensordict[("agents", "observation", "state")]
        action = self.action.to(device=obs.device, dtype=obs.dtype)
        view_shape = [1] * (obs.ndim - 1) + [action.shape[-1]]
        action = action.view(*view_shape).expand(*obs.shape[:-1], action.shape[-1]).clone()
        tensordict.set(("agents", "action"), action)
        return tensordict


def resolve_camera_mode(args: argparse.Namespace) -> str:
    if args.camera_mode:
        return args.camera_mode
    return "follow_drone" if args.camera_follow_drone else "fixed"


@contextmanager
def suppress_native_stdio():
    """Temporarily silence C++ plugin logs written to process stdout/stderr."""
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def main():
    args = parse_args()
    bootstrap_isaac_sim()
    from isaacsim import SimulationApp

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path, stats_output_path, trace_output_path, details_output_path = build_output_paths(
        checkpoint_path,
        args.output,
        args.stats_output,
        args.trace_output,
        args.details_output,
        log_video=bool(args.log_video),
    )
    cfg = load_cfg(args)
    max_steps = args.max_steps or cfg.env.max_episode_length
    step_dt = float(cfg.sim.dt * cfg.sim.substeps)
    fps = 1.0 / (cfg.sim.dt * cfg.sim.substeps * args.render_interval)
    log_video = bool(args.log_video)

    log_record(f"[record] checkpoint: {checkpoint_path}")
    log_record(f"[record] output video: {output_path if log_video else '(disabled)'}")
    log_record(f"[record] output stats: {stats_output_path}")
    log_record(f"[record] output trace: {trace_output_path if log_video else '(disabled)'}")
    log_record(f"[record] output details: {details_output_path}")
    log_record(f"[record] log video: {log_video}")
    log_record(f"[record] headless: {cfg.headless}")
    log_record(f"[record] tensor device: {cfg.device}")
    log_record(f"[record] sim device: {cfg.sim.device}")
    if not getattr(cfg.sim, "use_gpu_pipeline", False):
        log_record(
            f"[record] sim.use_gpu_pipeline={cfg.sim.use_gpu_pipeline}; "
            "using CPU-safe replay path."
        )
    if getattr(cfg, "_record_force_disable_fabric", False):
        log_record(
            "[record] forcing sim.use_flatcache/use_fabric=false for single-env headless replay "
            "to keep physics-to-render sync stable."
        )
    log_record(f"[record] num_envs: {cfg.env.num_envs}")
    log_record(f"[record] max_steps: {max_steps}")

    sim_experience = resolve_sim_experience(cfg.headless)
    sim_app_config = _build_simulation_app_config(cfg)
    sim_app = SimulationApp(
        sim_app_config,
        experience=sim_experience,
    )
    configure_gpu_pipeline_compatibility(cfg)
    enable_required_isaac_extensions(sim_app)
    stage = "startup"
    try:
        stage = "import runtime modules"
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl import RenderCallback
        from omni_drones.utils.torchrl.transforms import VelController

        from env import LandingEnv
        from ppo import PPO

        stage = "build env"
        log_record("[record] stage: building environment")
        env = LandingEnv(cfg)
        log_record("[record] stage: environment ready")

        if args.legacy_force_path:
            stage = "patch legacy force path"

            def _legacy_apply_rotor_forces(self):
                self.rotors_view.apply_forces_and_torques_at_pos(
                    self.thrusts.reshape(-1, 3),
                    is_global=False,
                )

            def _legacy_apply_base_wrench(self):
                self.base_link.apply_forces_and_torques_at_pos(
                    self.forces.reshape(-1, 3),
                    self.torques.reshape(-1, 3),
                    is_global=True,
                )

            env.drone._apply_rotor_forces_via_articulation_view = MethodType(
                _legacy_apply_rotor_forces, env.drone
            )
            env.drone._apply_base_wrench_via_articulation_view = MethodType(
                _legacy_apply_base_wrench, env.drone
            )
            log_record("[record] stage: forcing legacy RigidPrimView force path")

        # [New - 修复视频画面冻结]
        # SimulationContext.__init__() 内部调用 _init_stage() → super().__init__(sim_params=...)
        # 可能会把 configure_gpu_pipeline_compatibility() 提前设置的 updateToUsd 覆盖回 False。
        # 这里在 env 创建完毕后（SimulationContext 已完全初始化），再次强制设置 updateToUsd。
        # 当 use_gpu_pipeline=false 且 fabric 关闭时，物理状态必须通过 USD 同步到渲染管线。
        import carb as _carb
        _settings = _carb.settings.get_settings()
        if not getattr(cfg.sim, "use_gpu_pipeline", False):
            _use_fabric = getattr(cfg.sim, "use_fabric", getattr(cfg.sim, "use_flatcache", False))
            if not _use_fabric:
                _usd_sync_keys = {
                    "/physics/updateToUsd": True,
                    "/physics/updateVelocitiesToUsd": True,
                }
                for _key, _val in _usd_sync_keys.items():
                    _settings.set_bool(_key, _val)
                log_record(
                    "[record] 再次强制设置 updateToUsd=True（防止 SimulationContext 初始化覆盖）"
                )
            # 同时确认 Fabric 接口确实未加载
            _sim_ctx = env.sim
            if hasattr(_sim_ctx, "_fabric_iface") and _sim_ctx._fabric_iface is not None:
                log_record(
                    "[record][warn] SimulationContext 仍然加载了 fabric_iface，"
                    "这可能导致渲染走 Fabric 路径而忽略 updateToUsd。"
                )

        # [New - warm-up render]
        # 在 rollout 之前先执行几次 render，让渲染管线完全 warm up，
        # 避免前几帧因为管线尚未就绪而返回黑屏或旧画面。
        stage = "warm up render pipeline"
        if log_video:
            for _i in range(4):
                env.sim.render()
            log_record("[record] stage: render pipeline warmed up")
        else:
            log_record("[record] stage: render pipeline warm-up skipped (stats-only mode)")

        if args.constant_motor_cmd is not None:
            stage = "wrap env without controller transform"
            transformed_env = env.eval()
            transformed_env.set_seed(cfg.seed)
            base_env = env
            log_record("[record] stage: raw-motor environment ready")
        else:
            stage = "build controller"
            controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
            transforms = [VelController(controller, yaw_control=False)]

            stage = "wrap env"
            transformed_env = TransformedEnv(env, Compose(*transforms)).eval()
            transformed_env.set_seed(cfg.seed)
            base_env = transformed_env
            while hasattr(base_env, "base_env"):
                base_env = base_env.base_env
            log_record("[record] stage: transformed environment ready")

        if args.constant_motor_cmd is not None:
            stage = "build constant-motor policy"
            policy = ConstantActionPolicy(args.constant_motor_cmd, cfg.device).eval()
            log_record(
                "[record] stage: constant-motor policy initialized "
                f"with motor_cmd={list(map(float, args.constant_motor_cmd))}"
            )
        elif args.constant_action is not None:
            stage = "build constant-action policy"
            policy = ConstantActionPolicy(args.constant_action, cfg.device).eval()
            log_record(
                "[record] stage: constant-action policy initialized "
                f"with target_vel={list(map(float, args.constant_action))}"
            )
        else:
            stage = "build policy"
            policy = PPO(
                cfg.algo,
                transformed_env.observation_spec,
                transformed_env.action_spec,
                cfg.device,
            )
            log_record("[record] stage: policy initialized")

            stage = "load checkpoint"
            state_dict = torch.load(checkpoint_path, map_location=cfg.device)
            policy.load_state_dict(state_dict)
            policy.eval()
            log_record("[record] stage: checkpoint loaded")

        stage = "enable render"
        base_env.enable_render(log_video)
        if log_video:
            log_record("[record] stage: render enabled")
        else:
            log_record("[record] stage: render disabled (stats-only mode)")

        class BaseEnvRenderCallback(RenderCallback):
            def __init__(
                self,
                base_env,
                interval: int = 2,
                camera_mode: str = "fixed",
                landing_eye_offset: torch.Tensor | None = None,
                landing_target_offset: torch.Tensor | None = None,
                drone_eye_offset: torch.Tensor | None = None,
                drone_target_offset: torch.Tensor | None = None,
            ):
                super().__init__(interval=interval)
                self.base_env = base_env
                self.camera_mode = camera_mode
                self.landing_eye_offset = landing_eye_offset
                self.landing_target_offset = landing_target_offset
                self.drone_eye_offset = drone_eye_offset
                self.drone_target_offset = drone_target_offset

            def _update_camera(self) -> None:
                if self.camera_mode == "fixed":
                    return

                drone_state = self.base_env.drone.get_state(env_frame=False)
                drone_pos = drone_state[0, 0, :3].detach().cpu()
                landing_pos = self.base_env.target_pos[0, 0, :3].detach().cpu()

                if self.camera_mode == "follow_drone":
                    eye = drone_pos + self.drone_eye_offset
                    target = drone_pos + self.drone_target_offset
                elif self.camera_mode == "adaptive_landing":
                    delta = drone_pos - landing_pos
                    distance = float(torch.linalg.norm(delta).item())
                    scale = max(1.0, distance / 8.0)

                    eye = landing_pos.clone()
                    eye[0] += 0.25 * delta[0]
                    eye[1] += float(self.landing_eye_offset[1]) * scale
                    eye[2] += float(self.landing_eye_offset[2]) * scale

                    target = landing_pos.clone()
                    target[0] += 0.55 * delta[0]
                    target[1] += 0.55 * delta[1]
                    upward = max(float(delta[2].item()), 0.0)
                    target[2] += float(self.landing_target_offset[2]) + 0.45 * upward
                else:
                    raise ValueError(f"Unsupported camera mode: {self.camera_mode}")

                camera_prim_path = getattr(
                    self.base_env,
                    "_render_camera_prim_path",
                    "/OmniverseKit_Persp",
                )
                self.base_env.sim.set_camera_view(
                    tuple(float(x) for x in eye),
                    tuple(float(x) for x in target),
                    camera_prim_path=camera_prim_path,
                )

            def __call__(self, env, *args):
                if self.i % self.interval == 0:
                    self._update_camera()
                    frame = self.base_env.render(mode="rgb_array")
                    # 双保险：即便 render() 返回的是可变底层缓冲区，
                    # 这里也显式 copy 一份，避免视频帧互相别名。
                    self.frames.append(frame.copy())
                    self.t.update(self.interval)
                self.i += 1
                return self.i

        exploration_type = to_exploration_type(args.exploration_type)

        # 单环境录视频时，默认在第一个回合结束后就停。
        # 这样更符合“录一局”的直觉，也能避免后面一大段终止后冗余帧。
        stop_on_first_done = (cfg.env.num_envs == 1)
        log_record(f"[record] stop_on_first_done: {stop_on_first_done}")
        camera_mode = resolve_camera_mode(args)
        if camera_mode != "fixed" and cfg.env.num_envs != 1:
            log_record(f"[record][warn] camera mode '{camera_mode}' is only applied when num_envs=1.")
            camera_mode = "fixed"

        stage = "reset env"
        with torch.no_grad(), set_exploration_type(exploration_type):
            initial_td = transformed_env.reset()
            if log_video and getattr(cfg.sim, "use_gpu_pipeline", False) and cfg.headless:
                stage = "warm up offscreen GPU render path"
                log_record("[record] warming up offscreen GPU render path before rollout.")
                import carb as _carb

                _log_iface = _carb.logging.acquire_logging()
                _restore_level = _log_iface.get_level_threshold()
                _restore_enabled = _log_iface.is_log_enabled()
                _log_iface.set_log_enabled(False)
                _log_iface.set_level_threshold(_carb.logging.LEVEL_FATAL)
                with suppress_native_stdio():
                    try:
                        base_env.render(mode="rgb_array")
                        base_env.sim.step(render=True)
                        base_env.render(mode="rgb_array")
                    finally:
                        _log_iface.set_log_enabled(_restore_enabled)
                        _log_iface.set_level_threshold(_restore_level)
                initial_td = transformed_env.reset()
                stage = "reset env"
            target_top_z = float(base_env.target_pos[0, 0, 2].detach().cpu().item())
            initial_drone_pos = base_env.drone.get_state(env_frame=False)[0, 0, :3].detach().cpu()
            initial_landing_pos = base_env.target_pos[0, 0, :3].detach().cpu()
            viewer_eye = torch.tensor(cfg.viewer.eye, dtype=torch.float32)
            viewer_lookat = torch.tensor(cfg.viewer.lookat, dtype=torch.float32)
            landing_eye_offset = viewer_eye - initial_landing_pos
            landing_target_offset = viewer_lookat - initial_landing_pos
            drone_eye_offset = viewer_eye - initial_drone_pos
            drone_target_offset = viewer_lookat - initial_drone_pos
            render_callback = None
            if log_video:
                render_callback = BaseEnvRenderCallback(
                    base_env,
                    interval=args.render_interval,
                    camera_mode=camera_mode,
                    landing_eye_offset=landing_eye_offset,
                    landing_target_offset=landing_target_offset,
                    drone_eye_offset=drone_eye_offset,
                    drone_target_offset=drone_target_offset,
                )
            if log_video and camera_mode != "fixed":
                log_record(
                    f"[record] camera mode: {camera_mode} "
                    f"(landing_eye_offset={landing_eye_offset.tolist()}, "
                    f"landing_target_offset={landing_target_offset.tolist()})"
                )
            log_record("[record] stage: rollout starting")
            stage = "rollout"
            rollout_kwargs = dict(
                max_steps=max_steps,
                policy=policy,
                auto_reset=False,
                break_when_any_done=stop_on_first_done,
                return_contiguous=False,
                tensordict=initial_td,
            )
            if render_callback is not None:
                rollout_kwargs["callback"] = render_callback
            trajs = transformed_env.rollout(**rollout_kwargs)
        log_record("[record] stage: rollout finished")

        stage = "post reset"
        transformed_env.reset()

        trace = None
        video_frames = []
        if log_video:
            stage = "collect trace"
            trace = extract_first_env_trace(
                trajs, step_dt=step_dt, target_top_z=target_top_z
            )
            trace.update(
                {
                    "checkpoint": checkpoint_path,
                    "control_mode": (
                        "constant_motor_cmd"
                        if args.constant_motor_cmd is not None
                        else "constant_action"
                        if args.constant_action is not None
                        else "policy"
                    ),
                    "constant_action": list(map(float, args.constant_action)) if args.constant_action is not None else None,
                    "constant_motor_cmd": list(map(float, args.constant_motor_cmd)) if args.constant_motor_cmd is not None else None,
                    "video_path": output_path,
                    "stats_path": stats_output_path,
                    "trace_path": trace_output_path,
                    "details_path": details_output_path,
                    "headless": cfg.headless,
                    "num_envs": cfg.env.num_envs,
                    "max_steps": max_steps,
                    "step_dt": step_dt,
                    "stop_on_first_done": stop_on_first_done,
                }
            )

            stage = "save trace"
            with open(trace_output_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)
            log_record("[record] stage: trace file written")

            stage = "validate frames"
            if not render_callback.frames:
                raise RuntimeError(
                    "No frames were captured. Try rerunning with --headless false "
                    "or a smaller --render-interval."
                )
            log_record(f"[record] stage: captured {len(render_callback.frames)} frames")

            stage = "save video"
            video_frames = overlay_trace_on_frames(
                render_callback.frames, trace, args.render_interval
            )
            save_video(output_path, video_frames, fps=fps)
            log_record("[record] stage: video file written")
        else:
            log_record("[record] stage: video capture skipped (stats-only mode)")

        stage = "collect stats"
        stats = select_eval_stats(trajs)
        case_details = select_eval_case_details(trajs)
        stats.update(
            {
                "checkpoint": checkpoint_path,
                "control_mode": (
                    "constant_motor_cmd"
                    if args.constant_motor_cmd is not None
                    else "constant_action"
                    if args.constant_action is not None
                    else "policy"
                ),
                "constant_action": list(map(float, args.constant_action)) if args.constant_action is not None else None,
                "constant_motor_cmd": list(map(float, args.constant_motor_cmd)) if args.constant_motor_cmd is not None else None,
                "video_path": output_path,
                "trace_path": trace_output_path,
                "details_path": details_output_path,
                "headless": cfg.headless,
                "num_envs": cfg.env.num_envs,
                "max_steps": max_steps,
                "stop_on_first_done": stop_on_first_done,
                "frame_count": len(video_frames),
                "fps": fps,
            }
        )

        stage = "save stats"
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        stage = "save details"
        case_details.update(
            {
                "checkpoint": checkpoint_path,
                "video_path": output_path,
                "stats_path": stats_output_path,
                "trace_path": trace_output_path,
                "details_path": details_output_path,
                "headless": cfg.headless,
                "num_envs": cfg.env.num_envs,
                "max_steps": max_steps,
                "stop_on_first_done": stop_on_first_done,
                "frame_count": len(video_frames),
                "fps": fps,
            }
        )
        with open(details_output_path, "w", encoding="utf-8") as f:
            json.dump(case_details, f, indent=2, ensure_ascii=False)

        log_record("[record] done")
        for key, value in stats.items():
            if isinstance(value, float):
                log_record(f"[record] {key}: {value}")
    except Exception as exc:
        error_stats = {
            "checkpoint": checkpoint_path,
            "video_path": output_path,
            "trace_path": trace_output_path,
            "headless": cfg.headless,
            "num_envs": cfg.env.num_envs,
            "max_steps": max_steps,
            "stop_on_first_done": (cfg.env.num_envs == 1),
            "failed_stage": stage,
            "error": repr(exc),
        }
        try:
            with open(stats_output_path, "w", encoding="utf-8") as f:
                json.dump(error_stats, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        log_record(f"[record][error] failed at stage '{stage}': {exc}")
        traceback.print_exc()
        raise
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
