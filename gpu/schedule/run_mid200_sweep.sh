#!/bin/bash
# 顺序跑 009_25052026 场次29个正常视频的 ratio3_sh0_dense 中间200帧(f1100-f1299)训练
# + kinematics。每个视频一个config，schedule.py内部各起GPU0/GPU1共12 worker；配置间
# 严格顺序执行(schedule.py本身逐config阻塞，不并行多config)。每个视频训练完立刻跑
# 该视频的kinematics(batch_calc_kinematics.py，非交互，不阻塞)，再进入下一个视频。
# 单个视频训练或kinematics失败都不影响后续视频继续跑(错误记录在各自log里)。
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=/home/computer0/anaconda3/envs/fly_gsplat/bin/python
LOG_DIR=gpu/schedule/logs
mkdir -p "$LOG_DIR"

MOVS=(001 002 003 004 005 006 007 009 013 014 015 016 020 021 022 023 024 025 028 029 032 033 036 037 039 040 042 043 045)

for mov in "${MOVS[@]}"; do
    name="ctrl_009_${mov}_ratio3_sh0_dense_mid200"
    cfg="gpu/schedule/configs/ctrl_009_mid200/${name}.json"
    train_log="${LOG_DIR}/${name}.log"
    kin_log="${LOG_DIR}/${name}_kinematics.log"

    echo "=== [$(date -Is)] training ${name} ==="
    "$PY" gpu/schedule/schedule.py --config "$cfg" 2>&1 | tee "$train_log"
    echo "=== [$(date -Is)] training done ${name} ==="

    echo "=== [$(date -Is)] kinematics ${name} ==="
    "$PY" -m postprocessing.batch_calc_kinematics --sweep-name "$name" --group ratio3_sh0_dense 2>&1 | tee "$kin_log"
    echo "=== [$(date -Is)] kinematics done ${name} ==="
done

echo "=== all 29 videos done: $(date -Is) ==="
