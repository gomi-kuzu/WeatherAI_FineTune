#!/usr/bin/env bash
# FineTune 済み実機候補（FT-B Q4_K_M-im）の追加確認: 温度 0.2 × 3 回（500 行、行ごと seed）と、F16 に対する KL（train 側の校正文で）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; C=(docker compose -f "$HERE/env/docker/compose.yaml")
run=${1:-B}; d=/work/models/ft/$run; q=$d/gemma-4-E2B-wr-$run-Q4_K_M-im.gguf; f16=$d/gemma-4-E2B-wr-$run-F16.gguf
echo "=== [$(date +%H:%M:%S)] noise: $q"
bash "$HERE/eval/scripts/serve_gguf.sh" start "$q" 4
for seed in 0 1 2; do
  "${C[@]}" exec -T -w /work/eval/scripts eval python run_baseline.py --url http://quant:8080 --tag "FT-$run-Q4_K_M-im" --mode raw --temp 0.2 --seed $seed --limit 500 --label p2h --prompts /work/eval/data/baseline/dev_prompts_p2h.jsonl --concurrency 4 | tail -1
done
bash "$HERE/eval/scripts/serve_gguf.sh" stop
echo "=== [$(date +%H:%M:%S)] KL: F16 logits → Q4_K_M-im / Q4_K_M（校正文 = imatrix_calib.txt の先頭 200 段落）"
"${C[@]}" exec -T eval python -c "
t=open('$d/imatrix_calib.txt',encoding='utf-8').read().split('\n\n')[:200]; open('$d/kl_text.txt','w',encoding='utf-8').write('\n\n'.join(t)); print(len(t))"
"${C[@]}" exec -T quant llama-perplexity -m "$f16" -f "$d/kl_text.txt" --kl-divergence-base "$d/kld_f16.bin" -ngl 99 -c 512 -b 512 > "$HERE/logs/kl_${run}_base.log" 2>&1
for v in Q4_K_M-im Q4_K_M; do
  "${C[@]}" exec -T quant llama-perplexity -m "$d/gemma-4-E2B-wr-$run-$v.gguf" --kl-divergence-base "$d/kld_f16.bin" --kl-divergence -ngl 99 -c 512 -b 512 > "$HERE/logs/kl_${run}_$v.log" 2>&1
  echo "--- $v: $(grep -E 'Mean KLD|KLD' "$HERE/logs/kl_${run}_$v.log" | head -3 | tr '\n' ' ')"
done
echo "=== [$(date +%H:%M:%S)] extra done"
