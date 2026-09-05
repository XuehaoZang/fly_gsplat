#!/bin/bash
# 顺序跑完 Round 1 (gpu/schedule/configs/round1/) + Round 1.5
# (gpu/schedule/configs/round1_5/) 的全部config，每个config内部沿用schedule.py默认的
# 2 GPU x WORKERS_PER_GPU(=6) = 12 worker并发(不改schedule.py，本脚本只是顺序调度多个
# config)。跑完全部config后跑一次结果聚合(gpu/schedule/analysis/aggregate_round1.py)。
# 单个config失败(non-zero exit)不阻断后续config，记录在各自log里，最后在总log里汇总
# 一行失败列表。
#
# 用法: gpu/schedule/run_round1_sweep.sh
#   nohup gpu/schedule/run_round1_sweep.sh > gpu/schedule/logs/round1_master.log 2>&1 &
#   disown
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=/home/computer0/anaconda3/envs/fly_gsplat/bin/python
LOG_DIR=gpu/schedule/logs
mkdir -p "$LOG_DIR"

shopt -s nullglob
CONFIGS=(gpu/schedule/configs/round1/*.json gpu/schedule/configs/round1_5/*.json)
shopt -u nullglob

echo "=== [$(date -Is)] ${#CONFIGS[@]} config(s) queued ==="
FAILED=()

for cfg in "${CONFIGS[@]}"; do
    name="$(basename "$cfg" .json)"
    log="${LOG_DIR}/${name}.log"

    echo "=== [$(date -Is)] training ${name} ==="
    if ! "$PY" gpu/schedule/schedule.py --config "$cfg" 2>&1 | tee "$log"; then
        echo "=== [$(date -Is)] FAILED ${name} (see ${log}) ==="
        FAILED+=("$name")
    else
        echo "=== [$(date -Is)] done ${name} ==="
    fi
done

echo "=== [$(date -Is)] all ${#CONFIGS[@]} config(s) attempted, ${#FAILED[@]} failed ==="
if [ "${#FAILED[@]}" -gt 0 ]; then
    printf '  FAILED: %s\n' "${FAILED[@]}"
fi

echo "=== [$(date -Is)] aggregating results ==="
"$PY" gpu/schedule/analysis/aggregate_round1.py 2>&1 | tee "${LOG_DIR}/round1_aggregate.log"

echo "=== [$(date -Is)] round1 sweep complete ==="
