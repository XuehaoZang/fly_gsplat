#!/bin/bash
# 顺序跑一批"有效帧选择"(select_frame_window.py)筛出的视频：每个视频的居中480帧训练
# + kinematics(含eta_unwrap)。取代旧的run_mid200_sweep.sh —— mid200把"sparse mat总帧
# 槽位数"误当成"有效追踪信号长度"，19/29视频崩在Phase A，见select_frame_window.py
# 模块级docstring。config目录下每个视频一个config(generate_configs_from_selection.py
# 生成，只包含select_frame_window.py判定selected=True的视频)；schedule.py内部各起
# GPU0/GPU1共12 worker，配置间严格顺序执行。每个视频训练完立刻跑该视频的kinematics
# (batch_calc_kinematics.py，非交互，不阻塞)，再进入下一个视频。单个视频训练或
# kinematics失败都不影响后续视频继续跑(错误记录在各自log里)。
#
# 用法: gpu/schedule/run_valid480_sweep.sh <configs_dir> [param_set_name]
set -uo pipefail
cd "$(dirname "$0")/../.."

CONFIGS_DIR="${1:?usage: run_valid480_sweep.sh <configs_dir> [param_set_name]}"
PARAM_SET="${2:-ratio3_sh0_dense}"
PY=/home/computer0/anaconda3/envs/fly_gsplat/bin/python
LOG_DIR=gpu/schedule/logs
mkdir -p "$LOG_DIR"

shopt -s nullglob
CONFIGS=("$CONFIGS_DIR"/*.json)
shopt -u nullglob
echo "=== [$(date -Is)] ${#CONFIGS[@]} config(s) in ${CONFIGS_DIR} ==="

for cfg in "${CONFIGS[@]}"; do
    name="$(basename "$cfg" .json)"
    train_log="${LOG_DIR}/${name}.log"
    kin_log="${LOG_DIR}/${name}_kinematics.log"

    echo "=== [$(date -Is)] training ${name} ==="
    "$PY" gpu/schedule/schedule.py --config "$cfg" 2>&1 | tee "$train_log"
    echo "=== [$(date -Is)] training done ${name} ==="

    echo "=== [$(date -Is)] kinematics ${name} ==="
    "$PY" -m postprocessing.batch_calc_kinematics --sweep-name "$name" --group "$PARAM_SET" 2>&1 | tee "$kin_log"
    echo "=== [$(date -Is)] kinematics done ${name} ==="
done

echo "=== all ${#CONFIGS[@]} videos done: $(date -Is) ==="
