# LandRL_v2 代码备份与训练时间线说明

> 生成时间：2026-04-22（America/New_York）
> 作用：帮助快速看清当前目录下 **主代码状态 / 备份目录 / 训练日志 / wandb run / checkpoint 来源**。

---

## 1. 当前主代码状态（你现在正在编辑/准备继续实验的版本）

当前主代码文件：

- `env.py`
- `train.py`
- `ppo.py`
- `cfg/train.yaml`
- `cfg/ppo.yaml`
- `cfg/drone.yaml`
- `cfg/sim.yaml`
- `tools.py`

当前主代码最近修改时间：

- `env.py`：2026-04-22 00:59:57
- `cfg/train.yaml`：2026-04-22 00:58:38
- `cfg/ppo.yaml`：2026-04-22 00:58:47
- `train.py`：2026-04-21 03:13:11

### 当前主代码处于哪一步？
当前主代码已经进入 **阶段 A1（治抖动 / 抗抖动微调）** 的改动状态，核心包括：

- 动作 EMA 平滑
- 近地动作平滑惩罚
- 近地角速度惩罚
- 近地 upright 惩罚
- 学习率下调用于微调
- `eval_interval` 调为 150

**注意：截至本文件生成时，这一版 A1 代码本身还没有对应新的完整 wandb 训练 run。**  
也就是说：

- 当前代码 = **A1 改完后的代码状态**
- 但最近一个完整跑满的训练结果，仍对应 **A1 修改前** 的代码版本（见下文第 4 节）

---

## 2. 代码备份目录说明

### 2.1 `backup_before_A1_20260422/`

备份时间：**2026-04-22 00:53:31**

这是一个非常重要的备份，它表示：

> **在开始修改阶段 A1（EMA + 抗抖动奖励）之前，对当时主代码做的一次完整快照。**

备份内容：

- `backup_before_A1_20260422/env.py`
- `backup_before_A1_20260422/train.py`
- `backup_before_A1_20260422/ppo.py`
- `backup_before_A1_20260422/tools.py`
- `backup_before_A1_20260422/train.yaml`
- `backup_before_A1_20260422/ppo.yaml`
- `backup_before_A1_20260422/drone.yaml`
- `backup_before_A1_20260422/sim.yaml`

### 这个备份对应哪一步？
这个备份对应的是：

- **静态软着陆强化训练已经完成**
- 成功率已经很高
- `hard_landing` 已经基本压到 0
- 但是你通过视频观察发现：
  - 降落 **不够快**
  - 动作 **不够丝滑**
  - 机身仍有 **抖动 / 频繁微调**

所以这个备份可以理解为：

> **“进入 A1 抗抖动改动之前”的稳定基线代码**

如果之后 A1 改坏了，回滚时优先参考这个备份目录。

---

## 3. `wandb/latest-run` 当前指向哪里？

当前：

- `wandb/latest-run -> offline-run-20260421_131100-rlw3e38m`

含义：

> `latest-run` 表示 **最新一次 run**，不是“最优 run”。

所以：

- `latest-run` = 最新 run 目录
- `checkpoint_best.pt` = 该 run 内部记录的最佳 checkpoint

---

## 4. 训练日志与 wandb run 对照表

下面这张表是最重要的总览。

| 本地训练日志 | 对应 wandb run | 恢复来源 | 训练状态 | 到哪一步了 | 这一轮在做什么 |
|---|---|---|---|---|---|
| `training_live.log` | `offline-run-20260420_134834-2jto5wwj` | 无（从头/原始版本） | 已跑满 | Iteration 18310 / 1.2e9 frames | 原始早期静态降落训练 |
| `training_live_resume_20260421.log` | `offline-run-20260421_032122-5j6zfjnz` | 从 `20260420` 的 best 恢复 | 已跑满 | Iteration 18310 / 1.2e9 frames | 修改奖励/逻辑后的静态降落训练 |
| `training_live_softlanding_20260421_v2.log` | `offline-run-20260421_113741-4bs0j4ab` | 从 `20260421_032122` 的 best 恢复 | **中途停止** | Iteration 1172 | 第一次“软着陆强化”尝试 |
| `training_live_softlanding_20260421_v3.log` | `offline-run-20260421_131100-rlw3e38m` | 从 `v2` 的 best@300 恢复 | 已跑满 | Iteration 18310 / 1.2e9 frames | tmux 后台重启后的完整软着陆强化训练 |

另外还有一个 run：

- `offline-run-20260421_032008-31yw1vwd`

这个 run 只有 wandb 初始化记录，没有正式训练产物：

- 没有 `checkpoint_best_meta.json`
- 没有 `checkpoint_final.pt`
- `logs/debug.log` 显示它在启动后十几秒内就结束了

可以理解为：

> **一次很早期的短暂试跑/启动即结束 run，不作为主训练结果参考。**

---

## 5. 各轮训练结果摘要（按时间顺序）

### 5.1 第一轮：`offline-run-20260420_134834-2jto5wwj`

对应日志：`training_live.log`

特点：

- 原始早期训练
- 成功率明显不够
- 后期有退化问题

关键结果：

- 已完整跑满
- best checkpoint：`step = 3000`
- best success rate：`0.484375`
- final 结果较差，不建议作为后续 warm start 终点使用

意义：

> 第一轮证明“原始版本可以学到一部分着陆能力，但稳定性和最终效果不够好”。

---

### 5.2 第二轮：`offline-run-20260421_032122-5j6zfjnz`

对应日志：`training_live_resume_20260421.log`

恢复来源：

- 从第一轮 best checkpoint 恢复

特点：

- 修改奖励/逻辑后继续训练
- 静态着陆成功率大幅提升
- 但 `hard_landing` 仍偏高

关键结果：

- 已完整跑满
- best checkpoint：`step = 8400`
- best success rate：`1.0`
- final（18300）约：
  - `reach_goal = 0.96875`
  - `hard_landing = 0.625`
  - `flip = 0.03125`
  - `landing_hold_max = 9.875`
  - `final_dz_abs ≈ 0.0697`

意义：

> 第二轮已经把“能稳定落住”做出来了，但落地仍偏猛，属于“成功率高、动作质量一般”的版本。

---

### 5.3 第三轮（未跑完）：`offline-run-20260421_113741-4bs0j4ab`

对应日志：`training_live_softlanding_20260421_v2.log`

恢复来源：

- 从第二轮 best（`032122` 的 `checkpoint_best.pt`）恢复

特点：

- 第一次尝试更强的“软着陆强化”思路
- 训练没有完整跑满
- 但在很早期就拿到了一个很强的 best checkpoint

关键结果：

- **中途停止**，没有 final checkpoint
- 训练停在：`Iteration 1172`
- best checkpoint：`step = 300`
- best success rate：`1.0`
- 该 best 被后续第四轮继续拿来 warm start

意义：

> 虽然这轮没跑完，但它的 `step 300` best checkpoint 非常有价值，是后面完整跑满训练的直接起点。

---

### 5.4 第四轮（当前最重要基线）：`offline-run-20260421_131100-rlw3e38m`

对应日志：`training_live_softlanding_20260421_v3.log`

恢复来源：

- 从第三轮 `v2` 的 **best@300** 恢复

特点：

- 使用 `tmux` 后台完整跑满
- 静态软着陆已经非常成熟
- `hard_landing` 基本压到 0
- 但是你通过视频观察发现：
  - **动作还不够快**
  - **末端不够丝滑**
  - **机身仍有抖动/微调感**

关键结果：

- 已完整跑满
- best checkpoint：`step = 17100`
- best success rate：`1.0`
- best return：`94.67245483398438`
- final checkpoint 也已保存

后期正式评估（非常强）：

- `17100`: success = `1.0`
- `17400`: success = `1.0`
- `17700`: success = `0.984375`
- `18000`: success = `0.984375`
- `18300`: success = `1.0`

`18300` 的代表性指标：

- `reach_goal = 1.0`
- `hard_landing = 0.0`
- `flip = 0.0`
- `touchdown_once = 1.0`
- `landing_hold_max = 10.34375`
- `final_dz_abs ≈ 0.0620`

意义：

> 这是目前最值得保留的“静态软着陆基线训练结果”。

**注意：A1 改动前的代码备份（`backup_before_A1_20260422/`）对应的就是这一阶段。**

---

## 6. 当前最重要的 checkpoint 是哪些？

### 6.1 当前最重要的“静态软着陆基线” checkpoint

路径：

- `/mnt/A/hust_myc/RL/LandRL_v2/wandb/offline-run-20260421_131100-rlw3e38m/files/checkpoint_best.pt`

这是目前最推荐保留、后续实验常用的基线模型。

### 6.2 当前主代码默认恢复点

当前 `cfg/train.yaml` 中：

- `resume_checkpoint` 已指向上面这个 run 的 `checkpoint_best.pt`

即：

> 当前代码默认会从 **第四轮（静态软着陆已成熟）** 的 best checkpoint 继续微调。

---

## 7. 如果以后要快速判断“现在到哪一步了”，看哪里？

### 看当前代码是哪一代
看这些文件的修改时间：

- `env.py`
- `cfg/train.yaml`
- `cfg/ppo.yaml`

### 看最近训练对应哪一轮
优先看：

- `wandb/latest-run`
- 或本地日志：
  - `training_live.log`
  - `training_live_resume_20260421.log`
  - `training_live_softlanding_20260421_v2.log`
  - `training_live_softlanding_20260421_v3.log`

### 看当前最重要基线 checkpoint
优先看：

- `wandb/offline-run-20260421_131100-rlw3e38m/files/checkpoint_best.pt`

### 看 A1 修改前的代码备份
优先看：

- `backup_before_A1_20260422/`

---

## 8. 当前目录结构里最值得记住的几个对象

### 代码
- `env.py`：环境、奖励、状态、终止逻辑的核心文件
- `train.py`：训练主入口、checkpoint 恢复、collector、评估逻辑
- `ppo.py`：PPO 算法本体
- `cfg/train.yaml`：训练总配置、resume、reward、eval 等
- `cfg/ppo.yaml`：学习率等 PPO 超参数

### 备份
- `backup_before_A1_20260422/`：A1 抗抖动改动前快照

### 日志
- `training_live*.log`：本地训练过程完整 stdout 记录

### wandb 结果
- `wandb/offline-run-*/`：每一轮训练的离线结果目录
- `wandb/latest-run`：当前最新 run 的软链接

---

## 9. 建议的维护方式（以后可以继续往这个文件里追加）

以后如果你继续做：

- A1 训练
- A2（加观测）
- 阶段 B（提速）
- 动态降落 curriculum

建议每次只补 4 件事：

1. **新建了什么备份目录**
2. **这次改了哪些文件**
3. **对应哪个 wandb run**
4. **训练跑到哪一步 / best checkpoint 是哪一个**

这样以后回看就会非常清楚。

---

## 10. 当前一句话总结

> 当前目录里，最重要的代码备份是 `backup_before_A1_20260422/`；最重要的训练基线是 `offline-run-20260421_131100-rlw3e38m`；当前主代码已经进入 A1 抗抖动改动状态，但 A1 本身还没有形成新的完整训练 run（截至本文件生成时）。
