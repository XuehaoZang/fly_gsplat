#!/bin/bash
# 4cam Round 2 (round2_4cam_sweep_plan.md)：GPU训练(每视频4个config: d0/iters750/p2/opt) ->
# 结果聚合。P2(去腿诊断)组的数据必须已经用
# `python gpu/schedule/prepare_round2_variants.py` 提前准备好，本脚本不做数据准备。
#
# 单个config失败不阻断后续，记录在各自log里，最后在总log里汇总失败列表。
#
# 用法: gpu/schedule/run_round2_sweep.sh [--skip-opt] [--skip-p2]
#   nohup gpu/schedule/run_round2_sweep.sh > gpu/schedule/logs/round2_4cam_master.log 2>&1 &
#   disown
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=/home/computer0/anaconda3/envs/fly_gsplat/bin/python
LOG_DIR=gpu/schedule/logs
mkdir -p "$LOG_DIR"

SKIP_OPT=0
SKIP_P2=0
for arg in "$@"; do
    case "$arg" in
        --skip-opt) SKIP_OPT=1 ;;
        --skip-p2) SKIP_P2=1 ;;
    esac
done

CONFIGS=(gpu/schedule/configs/round2/ctrl_009_004_d0.json
         gpu/schedule/configs/round2/ctrl_009_016_d0.json
         gpu/schedule/configs/round2/ctrl_009_004_iters750.json
         gpu/schedule/configs/round2/ctrl_009_016_iters750.json)
if [ "$SKIP_P2" -eq 0 ]; then
    CONFIGS+=(gpu/schedule/configs/round2/ctrl_009_004_p2.json
              gpu/schedule/configs/round2/ctrl_009_016_p2.json)
fi
if [ "$SKIP_OPT" -eq 0 ]; then
    CONFIGS+=(gpu/schedule/configs/round2/ctrl_009_004_opt.json
              gpu/schedule/configs/round2/ctrl_009_016_opt.json)
fi

echo "=== [$(date -Is)] 4cam Round 2 GPU sweep: ${#CONFIGS[@]} config(s) queued ==="
FAILED=()
for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        echo "=== [$(date -Is)] SKIP (missing) ${cfg} ==="
        continue
    fi
    name="$(basename "$cfg" .json)"
    log="${LOG_DIR}/round2_4cam_${name}.log"

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

echo "=== [$(date -Is)] aggregating results ==="
"$PY" -m gpu.schedule.analysis.aggregate_round2 2>&1 | tee "${LOG_DIR}/round2_4cam_aggregate.log"

echo "=== [$(date -Is)] 4cam round2 sweep complete ==="
