import argparse
import math
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
try:
    import isaacsim  # noqa: F401
except ImportError:
    pass
from omni.isaac.kit import SimulationApp
from torchrl.envs.transforms import TransformedEnv, Compose
from torchrl.envs.utils import ExplorationType

FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ISAAC_TRAINING_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", ".."))
THIRD_PARTY_DIR = os.path.join(ISAAC_TRAINING_DIR, "third_party")
ORBIT_EXT_DIR = os.path.join(THIRD_PARTY_DIR, "orbit", "source", "extensions")


def ensure_local_python_paths():
    extra_paths = (
        os.path.join(THIRD_PARTY_DIR, "OmniDrones"),
        os.path.join(ORBIT_EXT_DIR, "omni.isaac.orbit"),
        os.path.join(ORBIT_EXT_DIR, "omni.isaac.orbit_assets"),
        os.path.join(ORBIT_EXT_DIR, "omni.isaac.orbit_tasks"),
    )
    for path in extra_paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


ensure_local_python_paths()


def _hydra_literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return str(value)


def _cfg_list(cfg_value, fallback):
    if cfg_value is None:
        return list(fallback)
    return list(cfg_value)


def _cfg_optional_str(cfg_value):
    if cfg_value is None:
        return None
    value = str(cfg_value).strip()
    return value or None


def _align_runtime_devices(cfg):
    tensor_device = _cfg_optional_str(getattr(cfg, "device", None))
    if tensor_device is None:
        raise ValueError("cfg.device is empty. Please set device in cfg/train.yaml or pass device=... via Hydra override.")

    sim_cfg = getattr(cfg, "sim", None)
    if sim_cfg is None:
        print(f"[device] tensor/model device: {tensor_device}")
        return tensor_device

    sim_device = _cfg_optional_str(getattr(sim_cfg, "device", None))
    if sim_device != tensor_device:
        if sim_device is None:
            print(f"[device] cfg.sim.device is empty; using cfg.device={tensor_device} for simulation and policy tensors.")
        else:
            print(
                f"[device] cfg.sim.device={sim_device} does not match cfg.device={tensor_device}; "
                f"using {tensor_device} for both."
            )
        cfg.sim.device = tensor_device

    print(f"[device] tensor/model device: {tensor_device}")
    print(f"[device] simulation device: {cfg.sim.device}")
    return tensor_device


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
    app_config = {
        "headless": True,
        "anti_aliasing": 0,
        "width": 320,
        "height": 240,
        "subscenes": 0,
    }
    sim_device_index = _parse_cuda_device_index(getattr(cfg.sim, "device", None))
    if sim_device_index is not None:
        app_config["active_gpu"] = sim_device_index
        app_config["physics_gpu"] = sim_device_index
        app_config["multi_gpu"] = False
        app_config["max_gpu_count"] = 1
        print(
            "[device] SimulationApp GPU routing: "
            f"active_gpu={sim_device_index}, physics_gpu={sim_device_index}, multi_gpu=False"
        )
    return app_config


def _load_numeric_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        key: float(value)
        for key, value in data.items()
        if isinstance(value, (int, float))
    }


def _extract_done_mask(tensordict):
    done = tensordict.get(("next", "done"), None)
    if done is None:
        term = tensordict.get(("next", "terminated"), None)
        trunc = tensordict.get(("next", "truncated"), None)
        if term is not None and trunc is not None:
            done = term | trunc
        elif term is not None:
            done = term
    if done is None:
        return None
    return done.squeeze(-1) if done.ndim > 2 else done


def _last_frame_stat_mean(tensordict, key):
    tensor = tensordict.get(("next", "stats", key), None)
    if tensor is None:
        return None
    if tensor.ndim >= 2:
        tensor = tensor[:, -1]
    return float(tensor.float().mean().item())


def _terminal_stat_mean(tensordict, done_mask, key):
    if done_mask is None or not done_mask.any():
        return None
    tensor = tensordict.get(("next", "stats", key), None)
    if tensor is None:
        return None
    values = tensor[done_mask]
    if values.numel() == 0:
        return None
    return float(values.float().mean().item())


def _format_metric_block(prefix, metrics):
    parts = [f"{name}={value:.3f}" for name, value in metrics if value is not None]
    if parts:
        print(f"{prefix} " + " | ".join(parts))


_ANSI_ENABLED = sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb" and "NO_COLOR" not in os.environ
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "blue": "\033[34m",
    "red": "\033[31m",
}


def _style(text, color=None, bold=False):
    if not _ANSI_ENABLED:
        return text
    prefix = ""
    if bold:
        prefix += _ANSI["bold"]
    if color:
        prefix += _ANSI[color]
    return prefix + text + _ANSI["reset"]


def _print_metric_group(title, metrics, color=None, items_per_line=3):
    metrics = [(name, value) for name, value in metrics if value is not None]
    if not metrics:
        return
    print(_style(f"  {title}", color=color, bold=True))

    def _fmt(value):
        if isinstance(value, (int, float)) and abs(float(value) - round(float(value))) < 1e-6:
            return str(int(round(float(value))))
        return f"{float(value):.3f}"

    for start in range(0, len(metrics), items_per_line):
        chunk = metrics[start:start + items_per_line]
        line = " | ".join(f"{name}={_fmt(value)}" for name, value in chunk)
        print(f"    {line}")


def _grid_camera_for_num_envs(num_envs: int, env_spacing: float):
    cols = int(math.ceil(math.sqrt(max(num_envs, 1))))
    rows = int(math.ceil(max(num_envs, 1) / cols))
    center_x = 0.5 * env_spacing * max(cols - 1, 0)
    center_y = 0.5 * env_spacing * max(rows - 1, 0)
    span = env_spacing * max(cols, rows)
    # A slightly top-down diagonal shot keeps the full grid visible while
    # preserving enough depth to see altitude changes and platform alignment.
    eye = [
        center_x - 0.18 * span,
        center_y + 1.08 * span,
        max(16.0, 0.95 * span),
    ]
    lookat = [center_x, center_y, 1.0]
    return eye, lookat


def _collect_external_numeric_info(stats_path: str, prefix: str) -> dict:
    info = {}
    external_stats = _load_numeric_json(stats_path)
    for key, value in external_stats.items():
        normalized_key = key
        if normalized_key.startswith("eval/"):
            normalized_key = normalized_key[len("eval/") :]
        info[f"{prefix}/" + normalized_key] = value
    return info


def _run_external_recording(
    *,
    checkpoint_path: str,
    cfg,
    eval_cfg,
    record_dir: str,
    name_prefix: str,
    step: int,
    num_envs: int,
    render_interval: int,
    exploration_type: str,
    headless: bool,
    use_gpu_pipeline: bool,
    use_flatcache: bool | None,
    use_fabric: bool | None,
    use_gpu: bool | None,
    video_device: str | None,
    timeout_sec: int,
    camera_mode: str,
    eye,
    lookat,
    resolution,
    seed: int,
    max_steps: int | None,
    log_video: bool,
    async_launch: bool = False,
):
    stem = f"{name_prefix}_{step}"
    video_path = os.path.join(record_dir, f"{stem}.mp4") if log_video else None
    stats_path = os.path.join(record_dir, f"{stem}.json")
    trace_path = os.path.join(record_dir, f"{stem}_trace.json") if log_video else None
    details_path = os.path.join(record_dir, f"{stem}_details.json")
    log_path = os.path.join(record_dir, f"{stem}.log")

    command = [
        sys.executable,
        os.path.join(PROJECT_DIR, "record_checkpoint.py"),
        "--checkpoint",
        checkpoint_path,
        "--stats-output",
        stats_path,
        "--details-output",
        details_path,
        "--log-video",
        _hydra_literal(log_video),
        "--num-envs",
        str(int(num_envs)),
        "--render-interval",
        str(int(render_interval)),
        "--headless",
        _hydra_literal(headless),
        "--exploration-type",
        str(exploration_type),
        "--seed",
        str(int(seed)),
        "--camera-mode",
        camera_mode,
        "--override",
        f"viewer.eye={_hydra_literal(eye)}",
        "--override",
        f"viewer.lookat={_hydra_literal(lookat)}",
        "--override",
        f"viewer.resolution={_hydra_literal(resolution)}",
        "--override",
        f"sim.use_gpu_pipeline={_hydra_literal(use_gpu_pipeline)}",
    ]
    if log_video and video_path:
        command.extend(["--output", video_path])
    if log_video and trace_path:
        command.extend(["--trace-output", trace_path])

    if hasattr(cfg.sim, "use_flatcache"):
        resolved_use_flatcache = (
            bool(cfg.sim.use_flatcache) if use_flatcache is None else bool(use_flatcache)
        )
        command.extend(
            ["--override", f"sim.use_flatcache={_hydra_literal(resolved_use_flatcache)}"]
        )
    if hasattr(cfg.sim, "use_fabric"):
        resolved_use_fabric = (
            bool(cfg.sim.use_fabric) if use_fabric is None else bool(use_fabric)
        )
        command.extend(
            ["--override", f"sim.use_fabric={_hydra_literal(resolved_use_fabric)}"]
        )
    if hasattr(cfg.sim, "use_gpu"):
        resolved_use_gpu = (
            bool(cfg.sim.use_gpu) if use_gpu is None else bool(use_gpu)
        )
        command.extend(
            ["--override", f"sim.use_gpu={_hydra_literal(resolved_use_gpu)}"]
        )
    if max_steps:
        command.extend(["--max-steps", str(int(max_steps))])
    if video_device:
        command.extend(["--device", str(video_device)])

    print(f"[LandRL_v2]: start external recorder '{stem}'", flush=True)
    print(f"[LandRL_v2]: recorder log -> {log_path}", flush=True)
    fps = 1.0 / (cfg.sim.dt * cfg.sim.substeps * max(int(render_interval), 1))
    result = {
        "video_path": video_path,
        "stats_path": stats_path,
        "trace_path": trace_path,
        "details_path": details_path,
        "log_path": log_path,
        "fps": fps,
        "command": command,
    }

    log_file = open(log_path, "w", encoding="utf-8")
    try:
        if async_launch:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            result["process"] = process
            result["log_handle"] = log_file
            return result

        try:
            subprocess.run(
                command,
                cwd=PROJECT_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout_sec,
            )
        finally:
            log_file.close()
    except Exception:
        if not log_file.closed:
            log_file.close()
        raise

    return result


def _load_best_checkpoint_meta(run_dir: str) -> dict:
    meta_path = os.path.join(run_dir, "checkpoint_best_meta.json")
    if not os.path.exists(meta_path):
        return {
            "success_rate": -1.0,
            "return": float("-inf"),
            "step": None,
            "metric_prefix": None,
        }
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "success_rate": float(meta.get("success_rate", -1.0)),
        "return": float(meta.get("return", float("-inf"))),
        "step": meta.get("step"),
        "metric_prefix": meta.get("metric_prefix"),
    }


def _remove_file_quietly(path: str | None):
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _materialize_external_result_info(prefix: str, result: dict, log_video: bool):
    info = {}
    if log_video and os.path.exists(result["video_path"]):
        info[f"{prefix}/video"] = wandb.Video(
            result["video_path"],
            fps=max(int(round(result["fps"])), 1),
            format="mp4",
        )
    info.update(_collect_external_numeric_info(result["stats_path"], prefix))
    info[f"{prefix}/video_path"] = result["video_path"]
    info[f"{prefix}/stats_path"] = result["stats_path"]
    info[f"{prefix}/trace_path"] = result["trace_path"]
    info[f"{prefix}/details_path"] = result["details_path"]
    return info


def _poll_external_eval_jobs(run, pending_jobs: list[dict], best_state: dict):
    completed_info = {}
    remaining_jobs = []

    for job in pending_jobs:
        process = job["process"]
        return_code = process.poll()
        if return_code is None:
            remaining_jobs.append(job)
            continue

        log_handle = job.get("log_handle")
        if log_handle is not None and not log_handle.closed:
            log_handle.close()

        prefix = job["name_prefix"]
        step = int(job["step"])
        print(
            f"[LandRL_v2]: async external recorder '{prefix}_{step}' finished with return code {return_code}",
            flush=True,
        )

        job_info = {}
        if return_code == 0 and os.path.exists(job["stats_path"]):
            job_info = _materialize_external_result_info(
                prefix,
                job,
                log_video=bool(job.get("log_video", False)),
            )
            job_info[f"{prefix}/evaluated_checkpoint_step"] = float(step)

            eval_print_keys = (
                "reach_goal",
                "collision",
                "hard_landing",
                "flip",
                "final_horizontal_err",
                "final_dz_abs",
                "final_vxy",
                "final_vz_abs",
                "min_horizontal_err",
                "min_dz_abs",
                "inside_platform_once",
                "touchdown_once",
                "landing_hold_max",
                "terminal_speed_violation",
                "unsafe_speed_xy",
                "unsafe_speed_z",
                "fail_too_far",
                "fail_above_bound",
                "fail_below_bound",
                "fail_flip",
                "fail_hard_landing",
            )
            async_metrics = [
                (key, job_info.get(f"{prefix}/stats." + key))
                for key in eval_print_keys
            ]
            if any(value is not None for _, value in async_metrics):
                title = "Eval Suite Stats (Async)" if prefix == "eval_suite" else "Eval Video Stats (Async)"
                color = "magenta" if prefix == "eval_suite" else "green"
                _print_metric_group(title, async_metrics, color=color, items_per_line=3)

            best_state = _maybe_save_best_checkpoint(
                policy=None,
                run=run,
                step=step,
                info=job_info,
                best_state=best_state,
                checkpoint_source_path=job.get("checkpoint_path"),
            )
        else:
            print(
                f"[LandRL_v2][warn]: async external recorder '{prefix}_{step}' did not produce usable stats. "
                f"See log: {job['log_path']}",
                flush=True,
            )
            job_info[f"{prefix}/returncode"] = float(return_code)
            job_info[f"{prefix}/evaluated_checkpoint_step"] = float(step)

        completed_info.update(job_info)
        _remove_file_quietly(job.get("checkpoint_path"))

    return completed_info, remaining_jobs, best_state


def _maybe_save_best_checkpoint(
    policy,
    run,
    step: int,
    info: dict,
    best_state: dict,
    checkpoint_source_path: str | None = None,
):
    for metric_prefix in ("eval_suite/stats",):
        success_key = f"{metric_prefix}.reach_goal"
        return_key = f"{metric_prefix}.return"
        if success_key not in info:
            continue

        success_rate = float(info[success_key])
        return_value = float(info.get(return_key, float("-inf")))
        summary_prefix = metric_prefix.rsplit("/", 1)[0]
        success_count = info.get(f"{summary_prefix}/summary.success_count")
        num_cases = info.get(f"{summary_prefix}/summary.num_cases")
        is_better = (
            success_rate > best_state["success_rate"] + 1e-9
            or (
                abs(success_rate - best_state["success_rate"]) <= 1e-9
                and return_value > best_state["return"] + 1e-9
            )
        )
        if not is_better:
            return best_state

        checkpoint_best_path = os.path.join(run.dir, "checkpoint_best.pt")
        checkpoint_best_meta_path = os.path.join(run.dir, "checkpoint_best_meta.json")
        if checkpoint_source_path and os.path.exists(checkpoint_source_path):
            shutil.copyfile(checkpoint_source_path, checkpoint_best_path)
        elif policy is not None:
            torch.save(policy.state_dict(), checkpoint_best_path)
        else:
            return best_state
        best_meta = {
            "step": int(step),
            "metric_prefix": metric_prefix,
            "success_rate": success_rate,
            "return": return_value,
            "checkpoint_path": checkpoint_best_path,
            "source_checkpoint_path": checkpoint_source_path,
            "summary_stats_path": info.get(f"{metric_prefix.rsplit('/', 1)[0]}/stats_path"),
            "details_path": info.get(f"{metric_prefix.rsplit('/', 1)[0]}/details_path"),
            "video_path": info.get(f"{metric_prefix.rsplit('/', 1)[0]}/video_path"),
            "success_count": int(success_count) if success_count is not None else None,
            "num_cases": int(num_cases) if num_cases is not None else None,
        }
        with open(checkpoint_best_meta_path, "w", encoding="utf-8") as f:
            json.dump(best_meta, f, indent=2, ensure_ascii=False)
        if success_count is not None and num_cases is not None:
            print(
                f"[LandRL_v2]: best checkpoint updated at step {step} "
                f"(eval_suite success_rate={success_rate:.3f}, success_count={int(success_count)}/{int(num_cases)})"
            )
        else:
            print(
                f"[LandRL_v2]: best checkpoint updated at step {step} "
                f"(eval_suite success_rate={success_rate:.3f})"
            )
        return {
            "success_rate": success_rate,
            "return": return_value,
            "step": int(step),
            "metric_prefix": metric_prefix,
        }

    return best_state


def run_external_eval_video(policy, cfg, run, step: int, allow_async_suite: bool = True):
    eval_cfg = getattr(cfg, "eval", None)
    if eval_cfg is None:
        return {}, []

    record_dir = os.path.join(run.dir, "external_recordings")
    os.makedirs(record_dir, exist_ok=True)

    checkpoint_path = os.path.join(record_dir, f"checkpoint_eval_{step}.pt")
    torch.save(policy.state_dict(), checkpoint_path)

    info = {}
    pending_jobs = []
    video_device = _cfg_optional_str(getattr(eval_cfg, "video_device", None))
    if video_device is None:
        video_device = _cfg_optional_str(getattr(cfg, "device", None))

    raw_video_use_gpu_pipeline = getattr(eval_cfg, "video_use_gpu_pipeline", None)
    if raw_video_use_gpu_pipeline is None or (
        isinstance(raw_video_use_gpu_pipeline, str) and not raw_video_use_gpu_pipeline.strip()
    ):
        video_use_gpu_pipeline = bool(getattr(cfg.sim, "use_gpu_pipeline", False))
    else:
        video_use_gpu_pipeline = bool(raw_video_use_gpu_pipeline)

    keep_checkpoint = False
    try:
        primary_num_envs = int(getattr(eval_cfg, "video_num_envs", 1))
        primary_camera_mode = str(getattr(eval_cfg, "video_camera_mode", "fixed") or "fixed").strip().lower()
        primary_eye = _cfg_list(getattr(eval_cfg, "video_eye", None), cfg.viewer.eye)
        primary_lookat = _cfg_list(getattr(eval_cfg, "video_lookat", None), cfg.viewer.lookat)
        primary_resolution = _cfg_list(getattr(eval_cfg, "video_resolution", None), cfg.viewer.resolution)

        primary_result = _run_external_recording(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            eval_cfg=eval_cfg,
            record_dir=record_dir,
            name_prefix="eval_video",
            step=step,
            num_envs=primary_num_envs,
            render_interval=int(getattr(eval_cfg, "render_interval", 2)),
            exploration_type=str(getattr(eval_cfg, "video_exploration_type", "mean")),
            headless=bool(getattr(eval_cfg, "video_headless", cfg.headless)),
            use_gpu_pipeline=video_use_gpu_pipeline,
            use_flatcache=None,
            use_fabric=None,
            use_gpu=None,
            video_device=video_device,
            timeout_sec=int(getattr(eval_cfg, "video_timeout_sec", 1800)),
            camera_mode=primary_camera_mode,
            eye=primary_eye,
            lookat=primary_lookat,
            resolution=primary_resolution,
            seed=int(cfg.seed),
            max_steps=getattr(eval_cfg, "video_max_steps", None),
            log_video=True,
        )

        primary_info = _materialize_external_result_info(
            "eval_video",
            primary_result,
            log_video=True,
        )
        if "eval_video/video" in primary_info:
            info["eval/video"] = primary_info.pop("eval_video/video")
        info.update(primary_info)

        suite_enabled = bool(getattr(eval_cfg, "multi_seed_enabled", False))
        suite_num_envs = int(getattr(eval_cfg, "multi_seed_num_envs", 64))
        if suite_enabled and suite_num_envs > 1:
            suite_eye_cfg = getattr(eval_cfg, "multi_seed_video_eye", None)
            suite_lookat_cfg = getattr(eval_cfg, "multi_seed_video_lookat", None)
            if suite_eye_cfg is None or suite_lookat_cfg is None:
                default_eye, default_lookat = _grid_camera_for_num_envs(
                    suite_num_envs,
                    float(getattr(cfg.env, "env_spacing", 8.0)),
                )
            else:
                default_eye, default_lookat = cfg.viewer.eye, cfg.viewer.lookat

            raw_suite_use_gpu_pipeline = getattr(eval_cfg, "multi_seed_use_gpu_pipeline", None)
            if raw_suite_use_gpu_pipeline is None or (
                isinstance(raw_suite_use_gpu_pipeline, str) and not raw_suite_use_gpu_pipeline.strip()
            ):
                suite_use_gpu_pipeline = bool(getattr(cfg.sim, "use_gpu_pipeline", False))
            else:
                suite_use_gpu_pipeline = bool(raw_suite_use_gpu_pipeline)
            suite_use_flatcache = getattr(eval_cfg, "multi_seed_use_flatcache", False)
            suite_use_fabric = getattr(eval_cfg, "multi_seed_use_fabric", False)
            suite_use_gpu = getattr(eval_cfg, "multi_seed_use_gpu", False)
            suite_video_device = _cfg_optional_str(getattr(eval_cfg, "multi_seed_video_device", None))
            if suite_video_device is None:
                suite_video_device = _cfg_optional_str(getattr(cfg, "device", None))
            suite_async = bool(getattr(eval_cfg, "multi_seed_async", True))

            if suite_async and not allow_async_suite:
                print(
                    f"[LandRL_v2][warn]: skipping async eval_suite_{step} because a previous suite job is still running.",
                    flush=True,
                )
                info["eval_suite/pending_skipped_step"] = float(step)
            else:
                suite_result = _run_external_recording(
                    checkpoint_path=checkpoint_path,
                    cfg=cfg,
                    eval_cfg=eval_cfg,
                    record_dir=record_dir,
                    name_prefix="eval_suite",
                    step=step,
                    num_envs=suite_num_envs,
                    render_interval=int(getattr(eval_cfg, "multi_seed_render_interval", max(int(getattr(eval_cfg, "render_interval", 2)), 4))),
                    exploration_type=str(getattr(eval_cfg, "multi_seed_exploration_type", "mean")),
                    headless=bool(getattr(eval_cfg, "multi_seed_headless", getattr(eval_cfg, "video_headless", cfg.headless))),
                    use_gpu_pipeline=suite_use_gpu_pipeline,
                    use_flatcache=suite_use_flatcache,
                    use_fabric=suite_use_fabric,
                    use_gpu=suite_use_gpu,
                    video_device=suite_video_device,
                    timeout_sec=int(getattr(eval_cfg, "multi_seed_timeout_sec", 3600)),
                    camera_mode=str(getattr(eval_cfg, "multi_seed_camera_mode", "fixed") or "fixed").strip().lower(),
                    eye=_cfg_list(suite_eye_cfg, default_eye),
                    lookat=_cfg_list(suite_lookat_cfg, default_lookat),
                    resolution=_cfg_list(getattr(eval_cfg, "multi_seed_video_resolution", None), cfg.viewer.resolution),
                    seed=int(getattr(eval_cfg, "multi_seed_seed", cfg.seed)),
                    max_steps=getattr(eval_cfg, "multi_seed_max_steps", getattr(eval_cfg, "video_max_steps", None)),
                    log_video=bool(getattr(eval_cfg, "multi_seed_log_video", False)),
                    async_launch=suite_async,
                )

                if suite_async:
                    keep_checkpoint = True
                    suite_result.update(
                        {
                            "name_prefix": "eval_suite",
                            "step": int(step),
                            "checkpoint_path": checkpoint_path,
                            "log_video": bool(getattr(eval_cfg, "multi_seed_log_video", False)),
                        }
                    )
                    pending_jobs.append(suite_result)
                    info["eval_suite/pending_step"] = float(step)
                    info["eval_suite/log_path"] = suite_result["log_path"]
                else:
                    suite_info = _materialize_external_result_info(
                        "eval_suite",
                        suite_result,
                        log_video=bool(getattr(eval_cfg, "multi_seed_log_video", False)),
                    )
                    info.update(suite_info)
    finally:
        if not keep_checkpoint:
            _remove_file_quietly(checkpoint_path)

    return info, pending_jobs


def resolve_sim_experience(cfg) -> str:
    # This tree runs on Isaac Sim 4.2 + Orbit. The copied LandRL code originally
    # targeted Isaac Lab app files, so we use the default SimulationApp experience.
    return ""


def configure_gpu_pipeline_compatibility(cfg):
    import carb

    settings = carb.settings.get_settings()
    settings.set_bool("/app/window/enabled", not bool(cfg.headless))
    settings.set_bool("/rtx/rendermode", False)
    print(f"[LandRL_v2]: headless render flags applied. headless={bool(cfg.headless)}")

    if not getattr(cfg.sim, "use_gpu_pipeline", False):
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

    print("[LandRL_v2]: GPU pipeline compatibility settings applied for Isaac Lab rendering.")


def enable_required_isaac_extensions(sim_app):
    from omni.isaac.core.utils.extensions import enable_extension

    required_extensions = [
        "omni.isaac.debug_draw",
    ]
    for extension_name in required_extensions:
        enabled = enable_extension(extension_name)
        print(f"[LandRL_v2]: enable_extension('{extension_name}') -> {enabled}")

    sim_app.update()


def enable_gpu_reset_debug_hooks():
    if os.environ.get("LANDRL_DEBUG_GPU_RESET", "0") != "1":
        return

    from omni.isaac.core.prims.xform_prim_view import XFormPrimView
    from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
    from omni.isaac.core.articulations.articulation_view import ArticulationView
    from omni_drones.views import ArticulationView as OmniDronesArticulationView
    from omni_drones.views import RigidPrimView as OmniDronesRigidPrimView

    def _make_hook(name):
        def _hook(self, *args, **kwargs):
            prims = getattr(self, "_prim_paths", None)
            print(f"\n[LANDRL_DEBUG_GPU_RESET] intercepted {name}")
            print(f"[LANDRL_DEBUG_GPU_RESET] prims={prims[:3] if prims else prims}")
            print("[LANDRL_DEBUG_GPU_RESET] Python stack:")
            print("".join(traceback.format_stack(limit=16)))
            raise RuntimeError(f"[LANDRL_DEBUG_GPU_RESET] intercepted old path: {name}")

        return _hook

    XFormPrimView.set_world_poses = _make_hook("XFormPrimView.set_world_poses")
    XFormPrimView.post_reset = _make_hook("XFormPrimView.post_reset")
    RigidPrimView.set_world_poses = _make_hook("RigidPrimView.set_world_poses")
    RigidPrimView.post_reset = _make_hook("RigidPrimView.post_reset")
    ArticulationView.set_world_poses = _make_hook("ArticulationView.set_world_poses")
    ArticulationView.post_reset = _make_hook("ArticulationView.post_reset")
    OmniDronesRigidPrimView.set_world_poses = _make_hook("omni_drones.views.RigidPrimView.set_world_poses")
    OmniDronesRigidPrimView.post_reset = _make_hook("omni_drones.views.RigidPrimView.post_reset")
    OmniDronesArticulationView.set_world_poses = _make_hook("omni_drones.views.ArticulationView.set_world_poses")
    OmniDronesArticulationView.post_reset = _make_hook("omni_drones.views.ArticulationView.post_reset")
    print("[LANDRL_DEBUG_GPU_RESET] debug hooks enabled.")

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    runtime_device = _align_runtime_devices(cfg)
    # Simulation App
    sim_experience = resolve_sim_experience(cfg)
    sim_app_config = _build_simulation_app_config(cfg)
    sim_app = SimulationApp(
        sim_app_config,
        experience=sim_experience,
    )
    configure_gpu_pipeline_compatibility(cfg)
    enable_required_isaac_extensions(sim_app)
    enable_gpu_reset_debug_hooks()
    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
    from omni_drones.utils.torchrl.transforms import VelController
    from vel_ema_transform import VelocityEMATransform  # 阶段A1：速度层 EMA 平滑
    from ppo import PPO
    from tools import evaluate
    # Use Wandb to monitor training
    # 将 cfg 转换为标准的字典格式，resolve=True 会解析所有的引用变量
    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    # Navigation Training Environment
    from env import LandingEnv  # use LandingEnv
    env = LandingEnv(cfg)

    # Transformed Environment
    # 控制链路：策略(vx,vy,vz) → VelocityEMATransform(平滑) → VelController(Lee) → 电机命令
    # Compose._inv_call 按逆序执行，所以 vel_ema 放在列表末尾 = 最先执行
    transforms = []
    controller = LeePositionController(9.81, env.drone.params).to(runtime_device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    # 阶段A1：速度层 EMA 平滑（alpha<1.0 时启用，alpha=1.0 向后兼容）
    ema_alpha = float(getattr(cfg.reward, "action_ema_alpha", 1.0))
    vel_ema = VelocityEMATransform(
        num_envs=env.num_envs, device=runtime_device, alpha=ema_alpha,
    )
    transforms.append(vel_ema)
    # 注入 transform 引用到 env，用于 _reset_idx 时同步清零缓冲
    env.vel_ema_transform = vel_ema
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    # 在 non-rendering headless.kit 下，Isaac/Replicator 的 set_global_seed 可能无法创建
    # /Replicator/SDGPipeline；这不应阻塞训练。失败时退回 torch 手动种子。
    try:
        transformed_env.set_seed(cfg.seed)
    except Exception as exc:
        print(f"[LandRL_v2][warn] transformed_env.set_seed({cfg.seed}) failed: {exc}")
        torch.manual_seed(int(cfg.seed))
    # PPO Policy
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, runtime_device)
    # Resume from checkpoint if specified
    resume_ckpt = getattr(cfg, "resume_checkpoint", None)
    if resume_ckpt and os.path.isfile(resume_ckpt):
        print(f"[LandRL_v2] Resuming from checkpoint: {resume_ckpt}")
        state_dict = torch.load(resume_ckpt, map_location=runtime_device)
        policy.load_state_dict(state_dict)
        print(f"[LandRL_v2] Checkpoint loaded successfully.")
    elif resume_ckpt:
        print(f"[LandRL_v2] WARNING: resume_checkpoint={resume_ckpt} not found, training from scratch.")

    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)
    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=runtime_device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )
    # Training Loop
    start_time = datetime.datetime.now()
    eval_record_video = bool(getattr(cfg.eval, "record_video", True))
    eval_render_interval = int(getattr(cfg.eval, "render_interval", 2))
    eval_video_backend = str(getattr(cfg.eval, "video_backend", "internal")).lower()
    tracked_running_stats = (
        "final_horizontal_err",
        "final_dz_abs",
        "final_vxy",
        "final_vz_abs",
        "min_horizontal_err",
        "min_dz_abs",
        "inside_platform_once",
        "touchdown_once",
        "landing_hold_max",
        "flip",
        "terminal_speed_violation",
        "unsafe_speed_xy",
        "unsafe_speed_z",
        # A1v3：抖动监控指标
        "ema_ang_vel_norm",
        "ema_accel_norm",
        "ema_action_diff",
        "ema_tilt",
    )
    tracked_terminal_stats = (
        "reach_goal",
        "collision",
        "hard_landing",
        "flip",
        "final_horizontal_err",
        "final_dz_abs",
        "final_vxy",
        "final_vz_abs",
        "min_horizontal_err",
        "min_dz_abs",
        "inside_platform_once",
        "touchdown_once",
        "landing_hold_max",
        "terminal_speed_violation",
        "unsafe_speed_xy",
        "unsafe_speed_z",
        "fail_too_far",
        "fail_above_bound",
        "fail_below_bound",
        "fail_flip",
        "fail_hard_landing",
        # A1v3：抖动监控指标
        "ema_ang_vel_norm",
        "ema_accel_norm",
        "ema_action_diff",
        "ema_tilt",
    )
    best_checkpoint_state = _load_best_checkpoint_meta(run.dir)
    pending_external_jobs = []
    for i, data in enumerate(collector):
        print(_style(f"================= Iteration {i} =================", color="magenta", bold=True))

        # 获取并打印 done 统计
        done = _extract_done_mask(data)
        total_done = int(done.sum()) if done is not None else 0
        total_transitions = int(done.numel()) if done is not None else 0
        total_landed = None
        success_rate = None

        # 打印reach_goal统计信息
        if ("next", "stats", "reach_goal") in data.keys(True, True):
            landed = data[("next", "stats", "reach_goal")]
            if done is not None:
                landed_on_done = landed[done]
                total_landed = int(landed_on_done.sum())
                if total_done > 0:
                    success_rate = total_landed / float(total_done)
            else:
                total_landed = int(landed.sum())

        batch_metrics = [
            ("done_total", float(total_done)),
            ("total_transitions", float(total_transitions)),
        ]
        if total_landed is not None:
            batch_metrics.append(("landed_total", float(total_landed)))
        if success_rate is not None:
            batch_metrics.append(("success_rate(ended_this_batch_only)", success_rate))
        _print_metric_group("Batch", batch_metrics, color="cyan", items_per_line=2)

        running_metrics = [
            (key, _last_frame_stat_mean(data, key))
            for key in tracked_running_stats
        ]
        _print_metric_group("Running Stats", running_metrics, color="green", items_per_line=3)

        terminal_metrics = [
            (key, _terminal_stat_mean(data, done, key))
            for key in tracked_terminal_stats
        ]
        if done is not None and int(done.sum()) > 0:
            _print_metric_group("Terminal Stats", terminal_metrics, color="yellow", items_per_line=3)

        for_time = datetime.datetime.now()
        print(_style(f"  Time Elapsed: {for_time - start_time}", color="blue"))
        start_time = for_time
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        async_info, pending_external_jobs, best_checkpoint_state = _poll_external_eval_jobs(
            run=run,
            pending_jobs=pending_external_jobs,
            best_state=best_checkpoint_state,
        )
        info.update(async_info)

        # Train Policy
        train_loss_stats = policy.update(data)
        info.update(train_loss_stats) # log training loss info

        # Calculate and log training episode stats
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs: # evaluate once if all agents finished one episode
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # Evaluate policy and log info
        # [旧代码 v2] if (i == 1) or (i > 0 and i % cfg.eval_interval == 0):
        if (i == 1) or (i > 0 and i % cfg.eval_interval == 0):
            print("[LandRL_v2]: start evaluating policy at training step: ", i)
            try:
                use_internal_video = eval_record_video and eval_video_backend == "internal"
                env.enable_render(use_internal_video)
                env.eval()
                eval_info = evaluate(
                    env=transformed_env, 
                    policy=policy,
                    seed=cfg.seed, 
                    cfg=cfg,
                    exploration_type=ExplorationType.MODE,
                    record_video=use_internal_video,
                    render_interval=eval_render_interval,
                )
                info.update(eval_info)
                if eval_record_video and eval_video_backend == "external_subprocess":
                    max_pending_suite_jobs = int(getattr(cfg.eval, "multi_seed_max_pending_jobs", 1))
                    running_suite_jobs = sum(
                        1 for job in pending_external_jobs if job.get("name_prefix") == "eval_suite"
                    )
                    video_info, new_external_jobs = run_external_eval_video(
                        policy,
                        cfg,
                        run,
                        i,
                        allow_async_suite=running_suite_jobs < max_pending_suite_jobs,
                    )
                    info.update(video_info)
                    pending_external_jobs.extend(new_external_jobs)
                eval_print_keys = (
                    "reach_goal",
                    "collision",
                    "hard_landing",
                    "flip",
                    "final_horizontal_err",
                    "final_dz_abs",
                    "final_vxy",
                    "final_vz_abs",
                    "min_horizontal_err",
                    "min_dz_abs",
                    "inside_platform_once",
                    "touchdown_once",
                    "landing_hold_max",
                    "terminal_speed_violation",
                    "unsafe_speed_xy",
                    "unsafe_speed_z",
                    "fail_too_far",
                    "fail_above_bound",
                    "fail_below_bound",
                    "fail_flip",
                    "fail_hard_landing",
                )
                eval_metrics = [
                    (key, info.get("eval/stats." + key))
                    for key in eval_print_keys
                ]
                _print_metric_group("Eval Stats", eval_metrics, color="cyan", items_per_line=3)
                eval_video_metrics = [
                    (key, info.get("eval_video/stats." + key))
                    for key in eval_print_keys
                ]
                if any(value is not None for _, value in eval_video_metrics):
                    _print_metric_group("Eval Video Stats", eval_video_metrics, color="green", items_per_line=3)
                eval_suite_metrics = [
                    (key, info.get("eval_suite/stats." + key))
                    for key in eval_print_keys
                ]
                if any(value is not None for _, value in eval_suite_metrics):
                    _print_metric_group("Eval Suite Stats", eval_suite_metrics, color="magenta", items_per_line=3)
                suite_success_rate = info.get("eval_suite/summary.success_rate")
                suite_success_count = info.get("eval_suite/summary.success_count")
                suite_num_cases = info.get("eval_suite/summary.num_cases")
                if suite_success_rate is not None:
                    if suite_success_count is not None and suite_num_cases is not None:
                        print(
                            _style(
                                f"  Eval Suite Success: {int(suite_success_count)}/{int(suite_num_cases)} "
                                f"(rate={float(suite_success_rate):.3f})",
                                color="magenta",
                                bold=True,
                            )
                        )
                    else:
                        print(
                            _style(
                                f"  Eval Suite Success Rate: {float(suite_success_rate):.3f}",
                                color="magenta",
                                bold=True,
                            )
                        )
                print("\n[LandRL_v2]: evaluation done.")
            except Exception as e:
                print(f"\n[LandRL_v2]: evaluation failed (non-fatal): {e}")
                print("[LandRL_v2]: continuing training...")
            finally:
                # [修复 v4] 评估后恢复渲染状态（无头模式下关闭渲染以提升训练性能）
                env.enable_render(not cfg.headless)
                env.train()
                env.reset()

        best_checkpoint_state = _maybe_save_best_checkpoint(
            policy=policy,
            run=run,
            step=i,
            info=info,
            best_state=best_checkpoint_state,
        )
        
        # Update wand info
        run.log(info)


        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[LandRL_v2]: model saved at training step: ", i)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    if pending_external_jobs:
        print(
            f"[LandRL_v2]: waiting for {len(pending_external_jobs)} pending external eval job(s) to finish...",
            flush=True,
        )
    while pending_external_jobs:
        time.sleep(5.0)
        async_info, pending_external_jobs, best_checkpoint_state = _poll_external_eval_jobs(
            run=run,
            pending_jobs=pending_external_jobs,
            best_state=best_checkpoint_state,
        )
        if async_info:
            run.log(async_info)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    
