#!/usr/bin/env bash
# baseline 評価の本番キュー（提案書 012 §3-2）: モデルごとに ① で llama-server を立て、② から実機文面を流す（ホストから実行。中身は全部コンテナ内）
#   bash eval/scripts/run_baseline_sweep.sh ["<tags...>"] ["<conds...>"]
#   tags  既定: "BF16 UD-Q4_K_XL"   （models/gemma-4-E2B-it-<tag>.gguf）
#   conds 既定: "raw:0:0 chat:0:0"  （mode:temp:seed[:style]。style を付けると dev_prompts_<style>.jsonl を使い結果名に -<style> が付く）
#   環境変数: LIMIT（全条件の行数上限。疎通テスト用）/ NOISE（1 なら BF16 に N1 = raw:0.2:{0,1,2} を先頭 NOISE_N 行で流す。既定 0）
#             NOISE_N（既定 500）/ NP（並列スロット、既定 4）/ CTX（1 スロットの文脈長、既定 1024）/ QMAX（q* スタイルの出力上限、既定 600）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C=(docker compose -f "$HERE/env/docker/compose.yaml")
TAGS=${1:-"BF16 UD-Q4_K_XL"}; CONDS=${2:-"raw:0:0 chat:0:0"}
LIMIT=${LIMIT:-0}; NOISE=${NOISE:-0}; NOISE_N=${NOISE_N:-500}; NP=${NP:-4}; CTX=${CTX:-1024}
run_one() {  # tag mode temp seed limit [style]
  local extra=""
  if [ -n "${6:-}" ]; then extra="--prompts /work/eval/data/baseline/dev_prompts_$6.jsonl --label $6"; fi
  case "${6:-}" in
    q3) extra="$extra --parse fmtidx --max-tokens ${QMAX:-600}";;   # 追加パターン ①'（降雨指数。021）
    q*) extra="$extra --parse fmt4 --max-tokens ${QMAX:-600}";;     # 追加パターン ①②（書式付き回答。018）
  esac
  "${C[@]}" exec -T -w /work/eval/scripts eval python run_baseline.py --url http://quant:8080 --tag "$1" --mode "$2" --temp "$3" --seed "$4" --limit "$5" --concurrency "$NP" $extra
}
for tag in $TAGS; do
  gguf=/work/models/gemma-4-E2B-it-$tag.gguf
  echo "=== [$(date +%H:%M:%S)] $tag: start llama-server ($gguf)"
  "${C[@]}" exec -T quant pkill -9 -x llama-server || true
  for i in $(seq 1 30); do "${C[@]}" exec -T quant pgrep -x llama-server >/dev/null 2>&1 || break; sleep 1; done
  "${C[@]}" exec -d quant bash -c "exec llama-server -m $gguf -ngl 99 -c $((CTX*NP)) -np $NP --host 0.0.0.0 --port 8080 > /tmp/llama-server_$tag.log 2>&1"
  for i in $(seq 1 120); do "${C[@]}" exec -T quant curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done
  "${C[@]}" exec -T quant curl -sf http://127.0.0.1:8080/health >/dev/null || { echo "server failed"; "${C[@]}" exec -T quant tail -20 /tmp/llama-server_$tag.log; exit 1; }
  served=$("${C[@]}" exec -T quant curl -sf http://127.0.0.1:8080/props | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_path',''))")
  [ "$served" = "$gguf" ] || { echo "model mismatch: $served != $gguf"; exit 1; }
  echo "serving: $served"
  for cond in $CONDS; do IFS=: read -r mode temp seed style <<< "$cond"; echo "--- [$(date +%H:%M:%S)] $tag $mode temp=$temp seed=$seed style=${style:-p0}"; run_one "$tag" "$mode" "$temp" "$seed" "$LIMIT" "${style:-}"; done
  if [ "$tag" = "BF16" ] && [ "$NOISE" = "1" ]; then
    n=$NOISE_N; [ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "$n" ] && n=$LIMIT
    for seed in 0 1 2; do echo "--- [$(date +%H:%M:%S)] $tag raw temp=0.2 seed=$seed (N1, $n rows)"; run_one "$tag" raw 0.2 "$seed" "$n"; done
  fi
  "${C[@]}" exec -T quant pkill -x llama-server || true
done
echo "=== [$(date +%H:%M:%S)] baseline sweep done: [$TAGS]"
