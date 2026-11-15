#!/usr/bin/env bash
# test（2024 年）での最終確認: 最良 1 本（既定 FT-B Q4_K_M-im = 実機候補）を test 層化 2,500 行で 1 回だけ採点する（016 §4-3。台帳に記録）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; C=(docker compose -f "$HERE/env/docker/compose.yaml")
run=${1:-B}; v=${2:-Q4_K_M-im}; gguf=/work/models/ft/$run/gemma-4-E2B-wr-$run-$v.gguf
while pgrep -f '^bash .*run_ft_extra[.]sh' >/dev/null; do sleep 30; done   # 自分を起動したシェルの文字列に一致しないよう先頭一致にする
echo "=== [$(date +%H:%M:%S)] test eval FT-$run-$v"
bash "$HERE/eval/scripts/serve_gguf.sh" start "$gguf" 4
"${C[@]}" exec -T -w /work/eval/scripts eval python run_baseline.py --url http://quant:8080 --tag "FT-$run-$v-TEST" --mode raw --temp 0 --seed 0 \
    --label p2h --prompts /work/eval/data/baseline/test_prompts_p2h.jsonl --concurrency 4 | tail -1
bash "$HERE/eval/scripts/serve_gguf.sh" stop
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v "$HERE":/w -w /w gemma4-eval python3 -I eval/scripts/score_baseline.py \
    --sample test --cond-filter 'TEST' --figdir eval/results/figs_test --summary-dir eval/results/baseline/out_017_test > "$HERE/logs/score_ft_test_20261016.log" 2>&1
echo "=== [$(date +%H:%M:%S)] test done exit=$?"
