import torch
import torch.nn as nn
import wandb
import numpy as np
import os
from contextlib import contextmanager
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchvision.models import resnet18

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        #定义两个 Softplus 激活函数，通常用于把某些参数强制约束为正数
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        return alpha, beta

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    #原本 [N_envs, T_steps] -> [N_envs * T_steps]
    tensordict = tensordict.reshape(-1) 
    # torch.randperm(n)：生成 [0, n-1] 的随机排列。
    # (tensordict.shape[0] // num_minibatches) * num_minibatches：
    # 确保能整除 num_minibatches，舍弃尾部多余数据。
    # .reshape(num_minibatches, -1)：
    # 将索引分成 num_minibatches 行，每行对应一个小批次的索引。
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    # 遍历每一行索引，返回对应的 TensorDict 切片。
    # yield：返回生成器，不一次性返回所有批次，节省内存。
    for indices in perm:
        yield tensordict[indices]


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

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MODE,
    record_video: bool=True,
    render_interval: int=2,
):
    render_interval = max(int(render_interval), 1)
    base_env = env
    while hasattr(base_env, "base_env"):
        base_env = base_env.base_env

    class BaseEnvRenderCallback(RenderCallback):
        def __init__(self, base_env, interval: int = 2):
            super().__init__(interval=interval)
            self.base_env = base_env

        def __call__(self, env, *args):
            if self.i % self.interval == 0:
                frame = self.base_env.render(mode="rgb_array")
                # Some Isaac Sim render backends reuse the underlying host buffer.
                # Copy the frame before storing it to keep recorded videos from
                # aliasing to a stale image.
                self.frames.append(np.array(frame, copy=True))
                self.t.update(self.interval)
            self.i += 1
            return self.i

    # [修复 v4 - Isaac Sim 4.1.0 GPU pipeline 渲染兼容]
    # 评估时整体对齐 Isaac Lab rendering experience 的 PhysX/Fabric 同步设置，
    # 避免 headless/offscreen 渲染落回旧式 USD 同步路径。
    import carb
    _settings = carb.settings.get_settings()
    use_gpu_pipeline = getattr(cfg.sim, 'use_gpu_pipeline', False)
    _compat_setting_keys = [
        "/physics/updateToUsd",
        "/physics/updateVelocitiesToUsd",
        "/physics/updateParticlesToUsd",
        "/physics/updateForceSensorsToUsd",
        "/physics/outputVelocitiesLocalSpace",
        "/physics/useFastCache",
        "/physics/fabricUpdateTransformations",
        "/physics/fabricUpdateVelocities",
        "/physics/fabricUpdateForceSensors",
        "/physics/fabricUpdateJointStates",
    ]
    _orig_compat_settings = {key: _settings.get(key) for key in _compat_setting_keys}
    if record_video and use_gpu_pipeline:
        print("[NavRL]: GPU pipeline 模式 — 固定 PhysX/Fabric 同步设置，保证 headless 渲染链路")
        for key in _compat_setting_keys:
            _settings.set_bool(key, False)

    base_env.enable_render(record_video)
    base_env.eval()
    env.eval()
    env.set_seed(seed)

    render_callback = (
        BaseEnvRenderCallback(base_env, interval=render_interval)
        if record_video
        else None
    )

    # [New - Isaac Sim 4.1.0] 手动 reset 获取初始 tensordict，
    # 然后用 auto_reset=False 评估 → 评估期间不触发任何 reset
    initial_td = env.reset()

    if record_video and use_gpu_pipeline and not getattr(base_env, "_offscreen_render_warmed", False):
        print("[NavRL]: warming up offscreen GPU render path before evaluation.", flush=True)
        log_iface = carb.logging.acquire_logging()
        log_restore_level = log_iface.get_level_threshold()
        log_restore_enabled = log_iface.is_log_enabled()
        log_iface.set_log_enabled(False)
        log_iface.set_level_threshold(carb.logging.LEVEL_FATAL)
        with suppress_native_stdio():
            try:
                # Isaac Sim 4.1 headless GPU rendering emits a one-time burst of
                # `PxArticulationLink::setGlobalPose()` errors the first time
                # the full offscreen pipeline (annotator attach + render step +
                # frame readback) hits the GPU path. Warm the whole chain once here
                # and then reset, so the real evaluation rollout starts clean.
                base_env.render(mode="rgb_array")
                base_env.sim.step(render=True)
                base_env.render(mode="rgb_array")
            finally:
                log_iface.set_log_enabled(log_restore_enabled)
                log_iface.set_level_threshold(log_restore_level)
        base_env._offscreen_render_warmed = True
        initial_td = env.reset()
    
    try:
        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=False,
                break_when_any_done=False,
                return_contiguous=False,
                tensordict=initial_td,
            )
    finally:
        # [修复 v4] 评估结束后恢复 updateToUsd 设置
        if record_video and use_gpu_pipeline:
            for key, value in _orig_compat_settings.items():
                _settings.set_bool(key, value if value is not None else False)

    # [旧代码] if should_render:
    # [旧代码]     env.enable_render(True)
    env.reset()
    
    done = trajs.get(("next", "done")) 
    total_done = int(done.sum())
    print(f"eval_done_total = {total_done} / {done.numel()}")
    first_done = torch.argmax(done.long(), dim=1).cpu() # idx of first done will be return for each trajs

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    info = {
        "eval/stats." + k: torch.mean(v.float()).item() 
        for k, v in traj_stats.items()
    }

    if render_callback is not None and render_callback.frames:
        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=1.0 / (cfg.sim.dt * cfg.sim.substeps * render_interval),
            format="mp4"
        )
    base_env.train()
    base_env.enable_render(not cfg.headless)
    env.train()

    return info


def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)
    # print("vec: ", vec.shape)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # goal direction y
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coodirnate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel


def construct_input(start, end):
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        # 将任意输入通道自适应映射到3通道，保持与懒模块初始化流程兼容
        self.to3 = nn.LazyConv2d(3, kernel_size=1)

        # 构建 resnet18 并做轻微调整以适配小尺寸/非RGB输入
        self.backbone = resnet18(weights=None)
        # 更接近原先的 5x3/stride 设置：5x5, stride=2；避免过度下采样关闭 maxpool
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()

        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_feats, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )

    def forward(self, x):
        x = self.to3(x)
        x = self.backbone(x)  # 全局平均池化后展平
        x = self.head(x)      # [N, out_dim]
        return x


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, H=36, W=4, in_channels=3,d_model=16, nhead=2, num_layers=1, out_dim=128):
        
        super().__init__()
        self.H = H
        self.W = W
        self.C = in_channels
        self.d_model = d_model

        # Step1: token embedding
        # 将单通道雷达距离映射到 d_model 维度
        self.token_embed = nn.Linear(self.C, d_model)

        # Step2: learnable 2D position embedding
        # H*W 个 token，每个 token 对应一个位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, H*W, d_model))

        # Step3: Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2,
            activation='relu',
            batch_first=True  # 使用 batch_first=True，输入形状为 [N, S, E]
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # Step4: feature head
        self.head = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU()
        )

    def forward(self, x):
        """
        x: [N, 1, H, W] 雷达输入
        returns: [N, out_dim] 特征向量
        """
        N, C, H, W = x.shape
        assert C == self.C and H == self.H and W == self.W, "输入尺寸不匹配"

        # 展平为 token 序列
        x = x.view(N, H*W, C)            # [N, 36*4=144, 1]
        x = self.token_embed(x)           # [N, 144, d_model]
        x = x + self.pos_embed            # 加上位置编码

        # Transformer encoder
        x = self.transformer(x)           # [N, 144, d_model]

        # 全局平均池化
        x = x.mean(dim=1)                 # [N, d_model]

        # feature head
        x = self.head(x)                  # [N, out_dim]
        return x
