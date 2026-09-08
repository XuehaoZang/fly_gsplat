#!/bin/bash
# Round 2 联合网格：GPU训练(6 configs, 8400任务, 2 GPU x 6 worker=12并发，跟round1一样
# 复用schedule.py默认并发，不改schedule.py本身的worker数) -> CPU侧kinematics+视频渲染
# (每组T1-T4+角度图+标注点云视频，训练全部跑完后单独跑，用满剩余CPU核) -> 结果聚合。
#
# 单个config/kinematics组失败不阻断后续，记录在各自log里，最后在总log里汇总失败列表
# (同run_round1_sweep.sh的容错约定)。
#
# 用法: gpu/schedule/run_round2_sweep.sh
#   nohup gpu/schedule/run_round2_sweep.sh > gpu/schedule/logs/round2_master.log 2>&1 &
#   disown
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=/home/computer0/anaconda3/envs/fly_gsplat/bin/python
LOG_DIR=gpu/schedule/logs
mkdir -p "$LOG_DIR"

shopt -s nullglob
ALL_CONFIGS=(gpu/schedule/configs/round2/*.json)
shopt -u nullglob
CONFIGS=()
for cfg in "${ALL_CONFIGS[@]}"; do
    base="$(basename "$cfg")"
    [[ "$base" == _* ]] && continue   # _smoketest*.json：一次性验证用，不进正式sweep
    CONFIGS+=("$cfg")
done

echo "=== [$(date -Is)] Round 2 GPU sweep: ${#CONFIGS[@]} config(s) queued ==="
FAILED=()
for cfg in "${CONFIGS[@]}"; do
    name="$(basename "$cfg" .json)"
    log="${LOG_DIR}/round2_${name}.log"

    echo "=== [$(date -Is)] training ${name} ==="
    if ! "$PY" gpu/schedule/schedule.py --config "$cfg" 2>&1 | tee "$log"; then
        echo "=== [$(date -Is)] FAILED ${name} (see ${log}) ==="
        FAILED+=("$name")
    else
        echo "=== [$(date -Is)] done ${name} ==="
    fi
done

echo "=== [$(date -Is)] GPU sweep phase done, ${#FAILED[@]} config(s) failed ==="
if [ "${#FAILED[@]}" -gt 0 ]; then
    printf '  FAILED: %s\n' "${FAILED[@]}"
fi

echo "=== [$(date -Is)] Round 2 kinematics+video phase (CPU, 8-way parallel) ==="
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 \
    "$PY" -m gpu.schedule.analysis.run_round2_kinematics --workers 8 \
    2>&1 | tee "${LOG_DIR}/round2_kinematics.log"

echo "=== [$(date -Is)] aggregating results ==="
"$PY" -m gpu.schedule.analysis.aggregate_round2 2>&1 | tee "${LOG_DIR}/round2_aggregate.log"

echo "=== [$(date -Is)] round2 sweep complete ==="
