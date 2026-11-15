#!/usr/bin/env bash
# ① quant コンテナで任意の GGUF の llama-server を起動／停止する（FineTune 済みモデルの評価用。提案書 016 §4-3）
#   bash eval/scripts/serve_gguf.sh start <container 内の gguf パス> [NP] [CTX]
#   bash eval/scripts/serve_gguf.sh stop
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C=(docker compose -f "$HERE/env/docker/compose.yaml")
cmd=${1:?start|stop}
stop_server() {
  "${C[@]}" exec -T quant pkill -9 -x llama-server || true
  for i in $(seq 1 30); do "${C[@]}" exec -T quant pgrep -x llama-server >/dev/null 2>&1 || break; sleep 1; done
}
case "$cmd" in
  stop) stop_server ;;
  start)
    gguf=${2:?gguf}; NP=${3:-4}; CTX=${4:-1024}
    stop_server
    "${C[@]}" exec -d quant llama-server -m "$gguf" -ngl 99 -c $((CTX*NP)) -np "$NP" --host 0.0.0.0 --port 8080
    for i in $(seq 1 120); do "${C[@]}" exec -T quant curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done
    "${C[@]}" exec -T quant curl -sf http://127.0.0.1:8080/health >/dev/null || { echo "server failed"; exit 1; }
    served=$("${C[@]}" exec -T quant curl -sf http://127.0.0.1:8080/props | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_path',''))")
    [ "$served" = "$gguf" ] || { echo "model mismatch: $served"; exit 1; }
    echo "serving: $served" ;;
esac
