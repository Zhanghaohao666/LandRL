# 备份，便于 deactivate 还原
export _OLD_PYTHONPATH="${PYTHONPATH}"
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"

# 注入 Isaac Sim 环境
source /mnt/A/hust_myc/RL/nvidia/isaac-sim/setup_python_env.sh
