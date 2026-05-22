# LandRL v2

这是当前工作区里用于 Isaac Sim 4.2 / Isaac Orbit 训练的 LandRL v2 任务代码。

当前推荐目录结构：

```text
/mnt/F/rl_train/RL-Train/
└── isaac-training/
    ├── run_train_landrl_v2_local.sh
    ├── third_party/
    │   ├── OmniDrones/
    │   └── orbit/
    └── training/
        └── landrl_v2/
```

## 运行

在当前机器上推荐从 `isaac-training` 目录启动：

```bash
cd /mnt/F/rl_train/RL-Train/isaac-training
./run_train_landrl_v2_local.sh max_frame_num=1024 env.num_envs=16 env.max_episode_length=128 env.num_obstacles=0 env_dyn.num_obstacles=0 eval_interval=999999 save_interval=999999
```

脚本默认使用：

- conda 环境：`NavRL`
- 设备：`cuda:0`
- 任务入口：`training/landrl_v2/train.py`

可以通过环境变量覆盖：

```bash
LANDRL_CONDA_ENV=NavRL LANDRL_DEVICE=cuda:0 ./run_train_landrl_v2_local.sh
```

## 已验证

当前代码已在本机 Isaac Sim 4.2 环境下完成过训练冒烟测试：

- `max_frame_num=64 env.num_envs=1`
- `max_frame_num=1024 env.num_envs=16`

训练可以正常启动、采样、更新 PPO，并写出 checkpoint。
