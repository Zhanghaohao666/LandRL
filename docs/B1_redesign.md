# LandRL_v2 阶段 B1：奖励 + 观测 + 坐标系重设计

**作者**：Cascade  
**日期**：2026-04-27  
**状态**：📝 设计稿，待审阅  
**目标**：解决 A1v3 训练遗留的"先飞上方再垂直降"和"机身抖动"问题，为动态降落 + 视觉部署打基础

---

## 0. 文档导读

- 第 1 节：现状与失败模式分析（**为什么旧方法不行**）
- 第 2 节：设计原则（参考 NavRL）
- 第 3-5 节：核心设计（**坐标系 + 观测 + 奖励**）
- 第 6-7 节：终止条件 + 训练超参
- 第 8 节：分阶段实施计划 + 每阶段验证标准
- 第 9 节：风险与回滚
- 第 10 节：待定问题

---

## 1. 现状与失败模式分析

### 1.1 当前状态（A1v3 best）

| 指标 | 值 |
|------|---|
| success_rate | 1.0 |
| return | 91.5 |
| episode_len | 559 步 (12.0 秒) |
| 水平误差 | 0.169 m |
| 着陆速度 | 0.026 m/s |
| ema_action_diff | 0.88 |

### 1.2 失败模式

#### 模式 1：fly-above-then-descend（先飞上方再垂直降）

**轨迹分析**（A1v3 best, eval step 4500）：

```
起点: (-1.68, -0.97, 2.75)，距平台 herr=1.94m, h=2.45m
着陆: 749 步 (12.0 秒)

阶段 1（0~2s, 16% 时间）：水平飞向平台正上方，期间反而上升 0.38m
阶段 2（2~12s, 84% 时间）：在平台正上方近乎纯垂直下降

中空段速度比 vz/vxy = 6.12 → 主要在垂直降高
```

**理想行为**应该是：从 (起点) 到 (平台中心) 的**对角直线**，约 4-6 秒完成。

#### 模式 2：机身抖动（shaking）

策略输出的 velocity command 帧间跳变剧烈 → LeePositionController 把命令振荡转换成姿态振荡 → 物理速度被低通滤波看起来"还行"，但 attitude 一直在调整。

加了 `action_smooth` 惩罚后 `ema_action_diff` 从 1.0 降到 0.88，**收效有限**。

#### 模式 3：训练后期崩塌（late-stage collapse）

A1v2 在 ~12750 步崩塌，A1v3 在 ~15450 步崩塌，**模式相同**：
- success_rate 从 1.0 急降到 0.2
- action_diff 和 ang_vel 同时降到很低
- 策略变成"几乎不动"

可能原因：长时间训练 + 无 LR 衰减 → 策略退化到保守的 trivial solution。

### 1.3 根本原因诊断

#### 原因 A：奖励信号鼓励"先水平对齐"

```@/mnt/A/hust_myc/RL/LandRL_v2/env.py:479-484
        # ===== 2. 下降奖励：水平对准后鼓励主动下降 =====
        aligned = (horizontal_err < self.descent_align_radius)    # [N, 1] bool
        descent_delta = (
            self.prev_height_above - height_above
        ).clamp(min=-0.15, max=0.15)
        reward_descent = aligned.float() * descent_delta
```

**`reward_descent` 被 `aligned` 门控**——只有水平对齐时降高才奖励。策略当然学到"必须先对齐才能开始降"。这是元凶。

#### 原因 B：`distance_progress` 不惩罚振荡

| 飞行方式 | distance_progress 给的 reward |
|---------|------------------------------|
| 直线飞 1m/s 朝目标 | 1.0 |
| 折线飞 (1m/s 速度模长) | 1.0 |
| 振荡飞但平均朝目标 | 1.0 |

distance_progress 只看"距离变化"，不区分**直飞**和**抖飞**。所以策略可以振荡飞而不被惩罚。

对比 NavRL 的 `velocity · target_dir`：
| 飞行方式 | NavRL reward |
|---------|---------------|
| 直线飞 1m/s 朝目标 | 1.0 |
| 折线飞 (1m/s 速度模长) | 0.5 |
| 振荡飞 | ≈ 0 |

**几何上自带反振荡机制**——速度向量垂直于目标方向的分量被完全浪费。

#### 原因 C：奖励项过多，相互冲突

当前 12+ 项奖励：`distance_progress + potential + descent + speed_shaping + touchdown_smooth + landed + hold + accel_penalty + action_smooth + bodyrate + upright + time_penalty + ...`

奖励项越多 → 局部最优越多 → 策略容易卡在"满足多个约束但行为怪异"的解上。NavRL 用 5 项就解决比降落更复杂的导航任务（带障碍）。

#### 原因 D：观测缺少 prev_action

```@/mnt/A/hust_myc/RL/LandRL_v2/env.py:450-459
        # ---- 9D 观测 ----
        drone_state = torch.cat([
            dx_dy,                          # [N, 2] 水平相对位置
            dz,                             # [N, 1] 垂直相对位置
            vel_w[:, 0:1],                  # [N, 1] vx
            vel_w[:, 1:2],                  # [N, 1] vy
            vel_w[:, 2:3],                  # [N, 1] vz
            up_vec,                         # [N, 3] up vector
        ], dim=-1)                          # [N, 9]
```

**没有 prev_action，没有 ang_vel**。策略不知道自己上一步发了什么命令，怎么平滑？

#### 原因 E：世界系坐标，未利用对称性

策略要分别学习"平台在东、北、西、南……不同方向"的飞行方式，**等价于 N 倍的样本量需求**。NavRL 用 goal-frame 把所有方向坍缩成"+x_g"，**样本效率提升 4-8 倍**。

### 1.4 总结：旧方法是结构性问题，不是参数问题

我们已经做了 A1 → A1v2 → A1v3 三轮"调权重 + 加惩罚"的修补，效果**边际递减**。继续修补无意义。**必须重设计**。

---

## 2. 设计原则

参考 NavRL 项目（`/mnt/A/hust_myc/RL/NavRL`）成功经验：

1. **奖励极简**：核心 5 项，每项物理意义清晰，无门控
2. **几何对称利用**：goal-frame 消除"目标方向"伪任务
3. **观测信息充分**：包含 prev_action + 角速度，让策略能做平滑控制
4. **物理直觉**：垂直方向保留世界系（重力对齐），不混入 goal-frame
5. **坐标系/接口贴近视觉**：用 direction + distance 而非绝对坐标

---

## 3. 坐标系：2D Goal-Frame

### 3.1 数学定义

定义 yaw 角 `θ = atan2(rpos_y, rpos_x)`，其中 `rpos = platform_pos - drone_pos`。

**Goal-frame** 是绕世界 z 轴旋转 `-θ` 的坐标系：
- `+x_g`：水平指向平台（投影方向）
- `+y_g`：水平垂直于 `+x_g`
- `+z_g`：与世界 z 一致（**不变**）

### 3.2 变换公式

给定世界系向量 `v_w = (v_x, v_y, v_z)` 和水平 goal 单位方向 `(c, s) = (cos θ, sin θ)`：

```
v_g_x = v_x · c + v_y · s     # 朝目标方向分量
v_g_y = -v_x · s + v_y · c    # 横向偏离分量
v_g_z = v_z                   # 垂直方向不变
```

代码（约 10 行）：

```python
def to_goal_frame_2d(vec_w, target_dir_xy):
    """
    vec_w:          [..., 3] 世界系向量
    target_dir_xy:  [..., 2] 水平 goal 单位方向
    """
    cos_t = target_dir_xy[..., 0:1]
    sin_t = target_dir_xy[..., 1:2]
    vx = vec_w[..., 0:1] * cos_t + vec_w[..., 1:2] * sin_t
    vy = -vec_w[..., 0:1] * sin_t + vec_w[..., 1:2] * cos_t
    vz = vec_w[..., 2:3]
    return torch.cat([vx, vy, vz], dim=-1)
```

### 3.3 边界处理：drone 在平台正上方

当 `rpos_xy.norm() < 0.05` 时，`target_dir_xy` 数值不稳定。**fallback**：

```python
horizontal_dist = rpos[:, :2].norm(dim=-1, keepdim=True)
target_dir_xy = torch.where(
    horizontal_dist > 0.05,
    rpos[:, :2] / horizontal_dist.clamp_min(1e-6),
    self.last_valid_target_dir_xy,   # 用上次有效方向
)
self.last_valid_target_dir_xy = target_dir_xy.detach()
```

这种情况主要发生在最后阶段（drone 已在平台正上方），不影响训练主流程。

### 3.4 动态平台兼容性

每步重新计算 `target_dir_xy`，平台运动 → 方向自动更新 → frame 自动转。**奖励代码完全不需要改**。

---

## 4. 观测空间重设计

### 4.1 新观测维度（14D，goal-frame）

```python
obs = [
    # === 几何信息 (2D) ===
    distance_horizontal,    # [1] 水平距离 sqrt(rpos_x² + rpos_y²)
    rpos_z,                 # [1] 垂直相对位置（platform_z - drone_z, 正=平台在上方）

    # === 速度信息 (3D, goal-frame) ===
    vel_g_x,                # [1] 朝平台水平速度
    vel_g_y,                # [1] 横向偏离速度（应趋近 0）
    vel_z,                  # [1] 垂直速度（世界系，下降为负）

    # === 姿态信息 (3D, goal-frame) ===
    up_g_x,                 # [1] up vector 在 goal-frame 的 x 分量
    up_g_y,                 # [1] up vector 在 goal-frame 的 y 分量
    up_z,                   # [1] up vector 的 z 分量（≈1 表示竖直）

    # === 角速度 (3D, goal-frame) ===
    ang_vel_g_x,            # [1]
    ang_vel_g_y,            # [1]
    ang_vel_z,              # [1] yaw rate（世界系即可）

    # === 历史动作 (3D, goal-frame) ===
    prev_action_g_x,        # [1] 上一步的 vel_cmd 在 goal-frame 投影
    prev_action_g_y,        # [1]
    prev_action_z,          # [1]
]  # 共 14 维
```

### 4.2 设计理由

| 字段 | 理由 |
|------|------|
| `distance_horizontal` + `rpos_z` | 显式分离水平/垂直距离，比 `rpos` 三分量更易学 |
| `vel_g`（goal-frame） | 策略只需学"vel_g_x 应正"，不用管平台方位 |
| `up_g`（goal-frame） | 与 vel_g 一致，姿态信息也对齐到 goal-frame |
| `ang_vel`（**新增**） | 让策略观测到自己在转 → 能主动稳定姿态 |
| `prev_action`（**新增**） | 让策略知道自己上一步发了什么 → 能做平滑决策 |

### 4.3 动态平台扩展（B2 阶段）

加入 3 维 platform velocity（goal-frame）：

```python
obs += [
    platform_vel_g_x,       # 平台远离/靠近的速度
    platform_vel_g_y,       # 平台横向移动速度
    platform_vel_z,         # 平台垂直速度（一般 0）
]  # 共 17 维
```

策略学会"vel_g_x > platform_vel_g_x 才能追上"等拦截行为。

### 4.4 视觉部署扩展（B3 阶段）

观测里 platform 信息加入噪声/延迟/丢失，模拟视觉不完美：

```python
# 平台位置：5cm 高斯噪声
platform_pos_obs = platform_pos_true + N(0, 0.05)

# 帧延迟：2 帧
platform_pos_obs = delay_buffer.push_pop(platform_pos_obs, k=2)

# 5% 概率检测丢失（用上次有效）
mask = (rand() < 0.05)
platform_pos_obs[mask] = self.last_valid_platform_pos[mask]
```

---

## 5. 奖励函数重设计

### 5.1 整体结构

参考 NavRL，**砍掉 80% 的旧奖励项**，只保留 5 项核心：

```python
self.reward = (
    # === 1. 朝目标速度（核心，几何反振荡）===
    + w_vel_toward * reward_vel_toward
    
    # === 2. 物理加速度惩罚（NavRL 同款）===
    - w_accel_penalty * penalty_phys_accel
    
    # === 3. 时间惩罚（鼓励快速完成）===
    - w_time_penalty
    
    # === 4. 着陆奖励（强终止激励）===
    + w_landed * reward_landed
    
    # === 5. 安全惩罚（终止性，硬着陆/出界/翻车）===
    - terminal_penalties  # 仅在终止时触发
)
```

**砍掉**的旧项及理由：

| 砍掉的项 | 理由 |
|---------|------|
| `distance_progress` | 被 `vel_toward` 取代（且更好，自带反振荡） |
| `potential` | 与 `distance_progress` 冗余 |
| `descent`（aligned 门控）| **罪魁祸首**，删除 |
| `speed_shaping` | `vel_toward` 自然鼓励合理速度 |
| `touchdown_smooth` | 物理加速度惩罚 + 着陆条件足够 |
| `hold`（10 步保持）| 改为单点判定，简化 |
| `action_smooth` | NavRL 没用也不抖，留 phys_accel 就够 |
| `bodyrate`（proximity 加权）| 角速度入观测后策略自己学会稳定 |
| `upright`（proximity 加权）| 同上 |

### 5.2 详细公式

#### 5.2.1 `reward_vel_toward`（核心）

```python
# 3D 单位向量（drone → platform）
target_dir_3d = rpos / (distance_3d + 1e-6)

# 速度沿目标方向分量（世界系点积，与 goal-frame 等价）
vel_toward = (vel_w * target_dir_3d).sum(dim=-1, keepdim=True)

# clamp 防止过大
reward_vel_toward = vel_toward.clamp(min=-1.0, max=2.0)
```

**含义**：drone 朝平台方向飞得越快，奖励越高。clip 到 2.0 防止策略为冲奖励疯狂加速；clip 到 -1.0 限制远离时的负奖励，避免梯度过大。

**权重**：`w_vel_toward = 1.0`（单位 m/s 给 1.0 reward/step）

#### 5.2.2 `penalty_phys_accel`（物理加速度）

```python
penalty_phys_accel = (vel_w - prev_vel_w).norm(dim=-1, keepdim=True)
```

**权重**：`w_accel_penalty = 0.1`（NavRL 同款）

#### 5.2.3 `reward_time_penalty`（时间惩罚）

```python
reward_time_penalty = -1.0  # 每步固定 -1
```

**权重**：`w_time_penalty = 0.05`（即每步 -0.05，750 步累积 -37.5）

#### 5.2.4 `reward_landed`（着陆奖励）

着陆判定保留几何代理法，但**简化**：

```python
on_platform_xy = horizontal_err < landing_xy_margin    # 0.35m
near_surface = (height_above < landing_dz_threshold) & (height_above > -landing_below_threshold)
low_vxy = vxy < landing_vxy_soft                        # 0.25 m/s
low_vz = vz_down < landing_vz_soft                      # 0.20 m/s
landed = on_platform_xy & near_surface & low_vxy & low_vz
```

**着陆奖励**（一次性大额）：

```python
reward_landed = landed.float()  # 着陆瞬间给 1
```

**权重**：`w_landed = 50.0`（直接的 +50 终止奖励，确保碾压悬停收益）

**判定条件去掉 `landing_hold_steps`**——一次满足就成功，简化训练信号。

#### 5.2.5 终止性惩罚

```python
# 翻车（up_z < 0）
self.reward[flipped] -= flip_penalty  # 10.0

# 出界（水平太远 / 高度太高 / 高度太低）
self.reward[oob] -= oob_penalty  # 5.0

# 极端硬着陆（速度过快进触地区）
self.reward[extreme_hard] -= hard_landing_penalty_extreme  # 30.0
```

中等硬着陆惩罚 (`hard_landing_penalty = 12`) **保留**，但不再终止——让策略从"差一点的硬着陆"中学习软着陆。

### 5.3 奖励数学验证：快速降落 vs 悬停

假设起点距平台 3m，最大速度 2 m/s。

**理想直降**（150 步 = 3 秒）：
```
reward_vel_toward: 150 × 2.0 (clamp) = 300
penalty_phys_accel: 150 × 0.1 × 0.05 ≈ -1
time_penalty: 150 × -0.05 = -7.5
landed: +50
合计: ≈ 341 ✅
```

**慢速降**（750 步 = 15 秒, 平均 0.4 m/s）：
```
reward_vel_toward: 750 × 0.4 = 300
penalty_phys_accel: -1
time_penalty: 750 × -0.05 = -37.5
landed: +50
合计: ≈ 311 ❌ 比快降少 30
```

**悬停不动**（750 步超时）：
```
reward_vel_toward: 0
penalty_phys_accel: 0
time_penalty: -37.5
landed: 0
合计: -37.5 ❌❌ 远低于其他选择
```

**振荡飞**（速度模长 2 但方向乱）：
```
reward_vel_toward: 平均 ≈ 0.3 (大部分速度浪费在垂直方向)
penalty_phys_accel: 大（频繁加减速）
合计: 远低于直飞 ❌
```

**结论**：奖励数值设计能正确诱导"快速直线降落"，且自动惩罚悬停和振荡。

---

## 6. 终止条件

### 6.1 成功

```python
reached_goal = landed  # 单步满足即成功（去掉 hold 计数）
```

### 6.2 失败终止

```python
too_far = horizontal_err > 4.0
above_bound = pos_z > platform_z + 4.0
below_bound = pos_z < platform_z - 0.8
flipped = up_z < 0.0
extreme_hard = (descending_into_touchdown_zone) & near_ground & extreme_speed

terminate = reached_goal | too_far | above_bound | below_bound | flipped | extreme_hard
```

### 6.3 截断

```python
truncate = progress_buf >= 750
```

---

## 7. 训练超参

### 7.1 关键变化

| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| `max_frame_num` | 12e8 | **6e8** | 缩半防过拟合，A1v3 12000 步就崩塌了 |
| `resume_checkpoint` | A1v2 best | **null** | 从零训练（旧策略习惯太重） |
| `eval_interval` | 500 | 500 | 保持 |
| `learning_rate_schedule` | 无 | **cosine** | 防后期崩塌 |
| `lr_min_ratio` | - | 0.1 | 衰减到初始 10% |

### 7.2 PPO 超参

保持当前配置，先不动。

### 7.3 速度限制（控制器层面）

```yaml
soft_speed_xy: 1.5           # 1.2 → 1.5 (允许更快对角飞)
hard_speed_xy: 3.0           # 2.5 → 3.0
soft_speed_vz_down: 1.2      # 0.9 → 1.2 (允许更快下降，时间效率)
hard_speed_vz_down: 2.5      # 2.2 → 2.5
```

**注意**：硬着陆判定（`hard_landing_vz_down=0.75`）不变——这是接地瞬间的安全约束，不是飞行速度上限。

---

## 8. 实施阶段与验证标准

### Stage 1：核心改动（B1.0）

**改动**：
- ✅ 加 goal-frame 2D 变换
- ✅ 改观测为 14D goal-frame
- ✅ 加 `prev_action` 缓冲与传递
- ✅ 改奖励为 5 项 NavRL 风格
- ✅ 简化着陆判定（去 hold）
- ✅ 加 LR cosine 衰减
- ✅ 缩短 max_frame_num
- ✅ 从零训练

**验证标准**（eval_suite 在 step 5000 前满足）：

| 指标 | 目标 | 容忍 |
|------|------|------|
| success_rate | ≥ 0.95 | ≥ 0.85 |
| episode_len | < 250 步 (5s) | < 350 步 |
| **vz/vxy 比（中空）** | < 2.0 | < 3.0 |
| ema_action_diff | < 0.5 | < 0.7 |
| hard_landing | ≤ 0.05 | ≤ 0.1 |

**关键判断**：trace 视频里能看到**对角飞行轨迹**，而非"先上后下"。

**Stage 1 失败的可能原因 & 应对**：
- 失败 A：振荡未减 → 检查是不是观测维度漏 `prev_action`
- 失败 B：success 上不去 → 检查 reward_landed 权重是否过小
- 失败 C：训练发散 → 检查 LR 衰减是否生效，或权重过大

### Stage 2：精修（B1.1）

**前提**：Stage 1 主要指标达成。

**改动**：
- 加近地 proximity 单一惩罚（替代被砍的 bodyrate + upright + speed_shaping）
- 微调权重
- 可选：加 ang_vel 输出层 EMA 平滑

**验证**：在 Stage 1 基础上进一步：
- vz/vxy 比 < 1.5
- ema_action_diff < 0.3
- 连续 5000 步无崩塌

### Stage 3：动态平台（B2）

**改动**：
- 加移动平台 wrapper（curriculum：静→慢→快→轨迹）
- 观测加 `platform_vel_g`（17D）
- 奖励**完全不动**

**验证**：
- 慢速直线（0.3 m/s）：success_rate ≥ 0.9
- 快速直线（0.8 m/s）：success_rate ≥ 0.7
- 圆轨迹（0.5 m/s）：success_rate ≥ 0.5

### Stage 4：视觉对接（B3，远期）

- 加 platform pose 噪声/延迟/丢失（DR）
- 接 CV detector
- 实机部署

---

## 9. 风险与回滚

### 9.1 主要风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Goal-frame 变换实现 bug | 中 | 高 | 写单元测试，对几个固定向量手算验证 |
| 简化奖励学不出着陆 | 低 | 高 | reward_landed 权重大（50），且保留 hard_landing 惩罚兜底 |
| 从零训练比 resume 慢 | 高 | 中 | max_frame 缩半已考虑，且新奖励信号更干净应该更快 |
| 14D 观测对策略容量不足 | 低 | 低 | PPO MLP 默认隐藏层 256，14→256 有冗余 |
| Drone 在平台正上方时 frame 不稳定 | 中 | 中 | 已用 last_valid_dir fallback |

### 9.2 回滚方案

代码改动前先做：

```bash
git checkout -b B1_redesign
git tag pre_B1_$(date +%Y%m%d)
cp env.py env.py.pre_B1.bak
cp cfg/train.yaml cfg/train.yaml.pre_B1.bak
```

如果 Stage 1 训了 5000 步都达不到容忍指标 → 回滚到 main 分支重新审视设计。

### 9.3 监控告警

训练中如果出现以下情况，立即停训：
- success_rate 持续 5 个 eval（约 2500 步）下降
- return 出现 NaN / Inf
- ema_action_diff > 1.5（说明抖动比 A1v3 还严重）

---

## 10. 待定问题

### Q1：要不要加 yaw 控制？

NavRL 用 `yaw_control=False`，drone yaw 自由旋转。我们目前同款。

**保留**：landing 任务里 yaw 也不重要，先不加。

### Q2：`reward_vel_toward` 用 3D 还是 2D 投影？

设计稿里用 **3D 点积**（`vel_w · target_dir_3d`）。备选是只看水平：`vel_w[:, :2] · target_dir_xy`。

**保留 3D**：3D 版本天然鼓励"既靠近又下降"，不需额外的 descent 项。如果 Stage 1 发现下降不积极，再考虑加单独的 descent 项。

### Q3：着陆判定要不要加角度约束？

当前判定不要求 drone 姿态竖直（up_z 接近 1）。可能允许"侧着陆"的 corner case。

**暂不加**：观测里有 `up_g`，策略应该自己学会保持竖直。如果 Stage 2 发现侧着陆，再加。

### Q4：是否启用 contact force tracking？

当前 `enable_contact_force_tracking: false`。

**保留 false**：物理仿真里 contact force 噪声大，且几何代理法已够用。

### Q5：训练时 platform 高度是否 randomize？

当前固定。视觉部署时实际高度会有变化。

**Stage 3 阶段处理**：在动态 curriculum 时加入 platform 高度 randomization。

---

## 附录 A：与 NavRL 的对比表

| 维度 | NavRL | LandRL_v1（旧） | **LandRL_v2 B1（新）** |
|------|-------|----------------|----------------------|
| 任务 | 导航避障 | 静态降落 | 静态/动态降落 |
| 坐标系 | 2D goal-frame | 世界系 | **2D goal-frame** |
| 观测维度 | 8 + lidar | 9 | **14** |
| 包含 prev_action | ❌ | ❌ | ✅ |
| 包含 ang_vel | ❌ | ❌ | ✅ |
| 奖励项数 | 5 | 12+ | **5** |
| 核心奖励 | `vel·target_dir + alive` | `distance_progress + 多项` | **`vel·target_dir + landed`** |
| Alive bonus | +1.0 | 无 | 无（用 time_penalty） |
| 物理加速度惩罚 | 0.1 | 0.1 | **0.1** |
| Action smooth | ❌ | 0.15 | 砍掉（NavRL 没用也不抖） |

---

## 附录 B：实施 Checklist

### Stage 1 实施 checklist

代码：
- [ ] 备份当前代码（`git tag` + 文件 backup）
- [ ] 在 `env.py` 加 `to_goal_frame_2d` 工具函数
- [ ] 修改 `_compute_state_and_obs`：构造 14D goal-frame 观测
- [ ] 修改 `_pre_sim_step` / 其他位置：缓存 `prev_action`
- [ ] 重写 `_compute_reward_and_done`：5 项奖励
- [ ] 修改 `_set_specs`：观测维度 9 → 14
- [ ] 简化着陆判定（去 hold counter）
- [ ] 在 `train.py` 加 LR cosine 衰减
- [ ] 修改 `cfg/train.yaml`：新权重 + max_frame=6e8 + resume=null
- [ ] 删除 stats 中已废弃的字段（避免 spec 不一致）

测试：
- [ ] 单元测试 `to_goal_frame_2d`（手算 3 个向量验证）
- [ ] 跑 100 步看是否 NaN
- [ ] 跑 1000 步看 reward 趋势是否合理

训练：
- [ ] 启动训练（tmux session: `B1_train`）
- [ ] 每 2500 步检查关键指标
- [ ] eval video 看 5000 步轨迹形状

---

## 文档结束

**审阅请关注**：
1. 第 5.2 节的奖励权重数值（`w_vel_toward=1.0, w_landed=50, w_time_penalty=0.05` 是否合理）
2. 第 4.1 节观测维度（14 维是否合适，是否有遗漏）
3. 第 8 节阶段验证标准（指标阈值是否合理）
4. 第 10 节待定问题（是否同意当前默认选择）

**审阅通过后**进入 Stage 1 实施。
