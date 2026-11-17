#!/usr/bin/env bash
# FineTune 本番キュー（提案書 016）: A / B / C を順に学習 → マージ → F16 GGUF（⑤ gemma4-train、ホストから nohup で起動）
#   bash eval/scripts/run_ft_queue.sh ["A B C"] [MAX_ROWS] [EVAL_ROWS]
#   T4 + float32（Unsloth が Gemma 4 の fp16 を拒否）で 1 秒 2.1 件 → 15,000 件で約 2 時間/本
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C=(docker compose -f "$HERE/env/docker/compose.yaml")
RUNS=${1:-"A B C"}; MAX_ROWS=${2:-16000}; EVAL_ROWS=${3:-300}
for run in $RUNS; do
  echo "=== [$(date +%H:%M:%S)] train $run (max_rows=$MAX_ROWS)"
  "${C[@]}" exec -T -w /tmp train python /work/eval/scripts/train_lora.py --run "$run" --data "/work/eval/data/sft/sft_$run.jsonl" \
      --max-rows "$MAX_ROWS" --eval-rows "$EVAL_ROWS" --base /work/models/hf/gemma-4-E2B-it --out /work/models/ft \
      > "$HERE/logs/train_${run}_$(date +%Y%m%d).log" 2>&1
  echo "--- [$(date +%H:%M:%S)] $run exit=$? gguf: $(ls -la $HERE/models/ft/$run/*.gguf 2>/dev/null | awk '{print $5, $9}')"
done
echo "=== [$(date +%H:%M:%S)] ft queue done: [$RUNS]"
