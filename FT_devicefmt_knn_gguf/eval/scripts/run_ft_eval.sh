#!/usr/bin/env bash
# FineTune 済みモデルの評価キュー（提案書 016 §4-3。ホストから実行。中身は全部コンテナ内）
#   bash eval/scripts/run_ft_eval.sh eval  "A B C"          # F16 GGUF を ① に載せ、013 と同じ 2,500 行を流す（C は P0 文面、他は P2'）
#   bash eval/scripts/run_ft_eval.sh quant A                # 最良 run を Q4_K_M（imatrix 無し / 有り）に量子化して同じ採点
# 結果: eval/results/baseline/FT-<run>[-Q4_K_M|-Q4_K_M-im]-<style>_raw_t0_s0.jsonl
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C=(docker compose -f "$HERE/env/docker/compose.yaml")
NP=${NP:-4}
style_of() { [ "$1" = "C" ] && echo p0 || echo p2h; }
prompts_of() { [ "$1" = "p0" ] && echo /work/eval/data/baseline/dev_prompts.jsonl || echo /work/eval/data/baseline/dev_prompts_$1.jsonl; }
eval_one() {  # tag gguf style [limit]
  local tag=$1 gguf=$2 style=$3 limit=${4:-0}
  bash "$HERE/eval/scripts/serve_gguf.sh" start "$gguf" "$NP"
  "${C[@]}" exec -T -w /work/eval/scripts eval python run_baseline.py --url http://quant:8080 --tag "$tag" --mode raw --temp 0 --seed 0 \
      --limit "$limit" --label "$style" --prompts "$(prompts_of $style)" --concurrency "$NP"
  bash "$HERE/eval/scripts/serve_gguf.sh" stop
}
cmd=${1:?eval|quant}; shift
case "$cmd" in
  eval)
    for run in ${1:-"A B C"}; do
      gguf=/work/models/ft/$run/gemma-4-E2B-wr-$run-F16.gguf; st=$(style_of $run)
      echo "=== [$(date +%H:%M:%S)] eval FT-$run ($st) $gguf"; eval_one "FT-$run" "$gguf" "$st"
    done ;;
  quant)
    run=${1:?run}; st=$(style_of $run); d=/work/models/ft/$run; f16=$d/gemma-4-E2B-wr-$run-F16.gguf
    echo "=== [$(date +%H:%M:%S)] quantize $run: Q4_K_M (no imatrix)"
    "${C[@]}" exec -T quant llama-quantize "$f16" "$d/gemma-4-E2B-wr-$run-Q4_K_M.gguf" Q4_K_M > "$HERE/logs/quantize_${run}_q4km.log" 2>&1
    echo "=== [$(date +%H:%M:%S)] imatrix from 2,000 training rows (train 分割の文面 + 回答。dev は使わない)"
    "${C[@]}" exec -T eval python - <<EOF
import json
rows=[json.loads(l) for l in open("/work/eval/data/sft/sft_$run.jsonl", encoding="utf-8")][-2000:]   # 学習に使っていない末尾側
open("/work/models/ft/$run/imatrix_calib.txt","w",encoding="utf-8").write("\n\n".join(r["prompt"]+r["completion"] for r in rows))
print("calib rows", len(rows))
EOF
    "${C[@]}" exec -T quant llama-imatrix -m "$f16" -f "$d/imatrix_calib.txt" -o "$d/imatrix.gguf" -ngl 99 -c 512 > "$HERE/logs/imatrix_${run}.log" 2>&1
    "${C[@]}" exec -T quant llama-quantize --imatrix "$d/imatrix.gguf" "$f16" "$d/gemma-4-E2B-wr-$run-Q4_K_M-im.gguf" Q4_K_M > "$HERE/logs/quantize_${run}_q4km_im.log" 2>&1
    ls -la "$HERE/models/ft/$run/"*.gguf
    for v in Q4_K_M Q4_K_M-im; do
      echo "=== [$(date +%H:%M:%S)] eval FT-$run-$v ($st)"; eval_one "FT-$run-$v" "$d/gemma-4-E2B-wr-$run-$v.gguf" "$st"
    done ;;
esac
echo "=== [$(date +%H:%M:%S)] $cmd done"
