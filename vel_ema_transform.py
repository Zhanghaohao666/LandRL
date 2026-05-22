"""
VelocityEMATransform — 在策略输出的速度指令上施加 EMA 低通滤波。

控制链路：
    RL策略 → (vx,vy,vz) → [VelocityEMATransform] → 平滑后速度 → [VelController(Lee)] → 电机命令 → env

在 TorchRL 的 Compose 中，_inv_call 按逆序执行。
因此本 transform 需要放在 VelController 之后（transforms 列表末尾），
这样 _inv_call 时它先执行（拿到策略原始速度），再由 VelController 转电机命令。
"""

import torch
from tensordict import TensorDictBase
from torchrl.envs.transforms import Transform


class VelocityEMATransform(Transform):
    """
    对策略输出的 3D 速度指令施加 EMA 平滑：
        u_exec = alpha * u_new + (1 - alpha) * u_prev_smoothed

    注意：TorchRL/OmniDrones 中策略 action 有时是 [N, 3]，有时是 [N, 1, 3]。
    本 transform 内部统一压成 [N, 3] 做 EMA，写回时保持原 action 形状。
    写给 env 的 info 始终使用 [N, 1, 3]，便于 env 计算奖励/统计。
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        alpha: float = 1.0,
        action_key=("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[action_key])
        self.alpha = float(alpha)
        self.action_key = action_key

        # 内部缓冲区统一为 [num_envs, 3]
        self.register_buffer("prev_smoothed", torch.zeros(num_envs, 3, device=device))
        self.register_buffer("prev_raw", torch.zeros(num_envs, 3, device=device))

    def _flatten_action(self, action: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Return [N,3] action and whether original had a singleton agent dim."""
        if action.ndim >= 3 and action.shape[-2] == 1 and action.shape[-1] == 3:
            return action.squeeze(-2), True
        if action.ndim >= 2 and action.shape[-1] == 3:
            return action, False
        raise RuntimeError(f"VelocityEMATransform expected action shape [N,3] or [N,1,3], got {tuple(action.shape)}")

    @staticmethod
    def _with_agent_dim(action_flat: torch.Tensor) -> torch.Tensor:
        return action_flat.unsqueeze(-2)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        raw_action = tensordict[self.action_key]
        raw_vel, had_agent_dim = self._flatten_action(raw_action)  # [N, 3]

        if raw_vel.shape != self.prev_smoothed.shape:
            raise RuntimeError(
                f"VelocityEMATransform buffer/action shape mismatch: raw={tuple(raw_vel.shape)}, "
                f"buffer={tuple(self.prev_smoothed.shape)}"
            )

        # ===== EMA 低通滤波 =====
        if self.alpha < 1.0:
            smoothed_vel = self.alpha * raw_vel + (1.0 - self.alpha) * self.prev_smoothed
        else:
            smoothed_vel = raw_vel

        smoothed_diff = smoothed_vel - self.prev_smoothed

        # ===== 暴露速度信息供 env 计算奖励/统计 =====
        tensordict.set(("info", "raw_vel_cmd"), self._with_agent_dim(raw_vel.detach().clone()))
        tensordict.set(("info", "prev_raw_vel_cmd"), self._with_agent_dim(self.prev_raw.detach().clone()))
        tensordict.set(("info", "smoothed_vel_diff"), self._with_agent_dim(smoothed_diff.detach().clone()))

        # ===== 更新缓冲区（先暴露 prev/diff，再更新）=====
        self.prev_smoothed.copy_(smoothed_vel.detach())
        self.prev_raw.copy_(raw_vel.detach())

        # ===== 写回 action，保持原 action 形状 =====
        tensordict.set(self.action_key, self._with_agent_dim(smoothed_vel) if had_agent_dim else smoothed_vel)
        return tensordict

    def reset_envs(self, env_ids: torch.Tensor):
        """当 env reset 时清零对应 env 的 EMA 缓冲。"""
        self.prev_smoothed[env_ids] = 0.0
        self.prev_raw[env_ids] = 0.0
