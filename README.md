# LandRL

本项目依赖 NVIDIA Isaac Sim（提供 `isaacsim` / `omni.*` Python 模块与运行时库）。
Isaac Sim 安装路径（本机）：
- `/mnt/A/hust_myc/RL/nvidia/isaac-sim/`

## 运行方式

### 方式 A：直接用 Isaac Sim 自带 Python（最稳）
```bash
cd /mnt/A/hust_myc/RL/LandRL
/mnt/A/hust_myc/RL/nvidia/isaac-sim/python.sh train.py
```

### 方式 B：用自己的 conda 环境运行（自动注入 Isaac Sim 环境）
目标：每次 `conda activate rl_drone` 时自动执行：
`source /mnt/A/hust_myc/RL/nvidia/isaac-sim/setup_python_env.sh`

#### 1) 在 rl_drone 环境里创建 conda hooks 目录
```bash
conda activate rl_drone
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
```

#### 2) 创建 activate/deactivate 脚本
# 覆盖 activate 脚本
```bash
cat > "$CONDA_PREFIX/etc/conda/activate.d/isaac_sim.sh" <<'EOF'
# 1. 基础路径设置
export ISAAC_SIM_PATH="/mnt/A/hust_myc/RL/nvidia/isaac-sim"

# 2. 备份原始变量
export _OLD_PYTHONPATH="${PYTHONPATH}"
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"

# 3. 补全 Isaac Sim 4.1.0 缺失的所有核心变量
export ISAAC_PATH="${ISAAC_SIM_PATH}"
export EXP_PATH="${ISAAC_SIM_PATH}/apps"
export CARB_APP_PATH="${ISAAC_SIM_PATH}/kit"
export ISAAC_SIM_LIBRARY_PATH="${ISAAC_SIM_PATH}/bin"

# 4. 执行官方环境设置脚本 (处理 PYTHONPATH 和 LD_LIBRARY_PATH)
if [ -f "${ISAAC_SIM_PATH}/setup_python_env.sh" ]; then
    source "${ISAAC_SIM_PATH}/setup_python_env.sh"
fi

# 5. 额外路径补丁，确保 exts 目录下的模块可见
export PYTHONPATH="${ISAAC_SIM_PATH}/exts/omni.isaac.kit:${PYTHONPATH}"
EOF
```

# 覆盖 deactivate 脚本
```bash
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/isaac_sim.sh" <<'EOF'
export PYTHONPATH="${_OLD_PYTHONPATH}"
export LD_LIBRARY_PATH="${_OLD_LD_LIBRARY_PATH}"
unset _OLD_PYTHONPATH
unset _OLD_LD_LIBRARY_PATH
unset ISAAC_SIM_PATH
unset EXP_PATH
unset ISAAC_SIM_LIBRARY_PATH
unset OMNI_APP_PATH
EOF
```

#### 3) 验证
```bash
conda deactivate
conda activate rl_drone
python -c "from isaacsim import SimulationApp; print('import ok')"
```

之后即可：
```bash
cd /mnt/A/hust_myc/RL/LandRL
python train.py
```

## 常见问题

### conda activate/deactivate 报错：anaconda-auth / pydantic_core
报错样例：
`Error while loading conda entry point: anaconda-auth (No module named 'pydantic_core._pydantic_core')`

含义：conda 自身加载 `anaconda-auth` 插件时缺少依赖。

推荐修复（在 base 环境卸载它）：
```bash
conda activate base
pip uninstall anaconda-auth
```

## 运行
conda activate rl_drone
python train.py headless=True

## 录制视频  
python record_checkpoint.py \
  --checkpoint /mnt/A/hust_myc/RL/LandRL_zgh/wandb/offline-run-20260319_085855-ydg9ujzm/files/checkpoint_0.pt \
  --num-envs 1 \
  --max-steps 2200 \
  --render-interval 1 \
  --headless true \
  --override sim.use_gpu_pipeline=false

### 镜头更近
python record_checkpoint.py \
  --checkpoint /mnt/A/hust_myc/RL/LandRL_zgh/wandb/offline-run-20260319_085855-ydg9ujzm/files/checkpoint_0.pt \
  --num-envs 1 \
  --max-steps 2200 \
  --render-interval 1 \
  --headless true \
  --override sim.use_gpu_pipeline=false \
  --override viewer.eye='[0.0,12.0,8.0]' \
  --override viewer.lookat='[0.0,0.0,1.2]'

  <!-- --override viewer.eye='[0.0,8.0,6.0]' \
  --override viewer.lookat='[0.0,0.0,2.0]' -->



  python record_checkpoint.py \
  --checkpoint /mnt/A/hust_myc/RL/LandRL_zgh/wandb/offline-run-20260319_085855-ydg9ujzm/files/checkpoint_0.pt \
  --num-envs 1 \
  --max-steps 2200 \
  --render-interval 1 \
  --headless true \
  --exploration-type random \  #随机策略测试
  --override sim.use_gpu_pipeline=false \
  --override viewer.eye='[0.0,12.0,8.0]' \
  --override viewer.lookat='[0.0,0.0,1.2]'


  python record_checkpoint.py \
  --checkpoint /mnt/A/hust_myc/RL/LandRL_zgh/wandb/offline-run-20260319_085855-ydg9ujzm/files/checkpoint_0.pt \
  --num-envs 1 \
  --max-steps 2200 \
  --render-interval 1 \
  --headless true \
  --override sim.use_gpu_pipeline=false \
  --override viewer.eye='[0.0,12.0,8.0]' \
  --override viewer.lookat='[0.0,0.0,1.2]'

  
/mnt/A/hust_myc/RL/LandRL_zgh/wandb/offline-run-20260320_091332-r9y8jmwc/files/checkpoint_2000.pt

  python record_checkpoint.py \
  --checkpoint /mnt/A/hust_myc/RL/LandRL_zgh/wandb/offline-run-20260320_091332-r9y8jmwc/files/checkpoint_2000.pt \
  --num-envs 1 \
  --max-steps 2200 \
  --render-interval 1 \
  --headless true \
  --override sim.use_gpu_pipeline=false \
  --override viewer.eye='[0.0,12.0,8.0]' \
  --override viewer.lookat='[0.0,0.0,1.2]'



  2026-3-30
  固定看起飞点/降落区：
  cd /mnt/A/hust_myc/RL/LandRL_zgh
/mnt/A/hust_myc/RL/nvidia/isaac-sim/python.sh record_checkpoint.py \
  --checkpoint wandb/latest-run/files/checkpoint_6000.pt \
  --camera-mode fixed \
  --num-envs 1 \
  --headless true \
  --override viewer.eye=[0.0,12.0,8.0] \
  --override viewer.lookat=[0.0,0.0,1.2] \
  --override viewer.resolution=[1280,720] \
  --override sim.use_gpu_pipeline=false

锁定无人机，推荐用这个正上方视角：
cd /mnt/A/hust_myc/RL/LandRL_zgh
/mnt/A/hust_myc/RL/nvidia/isaac-sim/python.sh record_checkpoint.py \
  --checkpoint wandb/latest-run/files/checkpoint_6000.pt \
  --camera-mode follow_drone \
  --num-envs 1 \
  --headless true \
  --override viewer.eye=[0.0,0.0,10.0] \
  --override viewer.lookat=[0.0,0.0,0.0] \
  --override viewer.resolution=[1280,720] \
  --override sim.use_gpu_pipeline=false



tmux运行

开一个session
 tmux new -s landrl 

source /home/ai/app/anaconda3/etc/profile.d/conda.sh
conda activate rl_drone

cd /mnt/A/hust_myc/RL/LandRL_zgh
/mnt/A/hust_myc/RL/nvidia/isaac-sim/python.sh train.py \
  headless=true \
  env.num_envs=256 \
  max_frame_num=2e7 \
  eval_interval=200 \
  save_interval=200 \
  eval.video_backend=external_subprocess \
  eval.video_num_envs=1 \
  eval.video_max_steps=300 \
  eval.video_camera_mode=fixed

  挂到后台  ctrl+b+d

  重新连回去 tmux attach -t landrl