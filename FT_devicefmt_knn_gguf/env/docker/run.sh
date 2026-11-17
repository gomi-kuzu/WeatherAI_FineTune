#!/usr/bin/env bash
# Gemma 4 × QRB2210 検証環境の操作入口（ホストで実行するのはこのスクリプト＝docker compose だけ）
#   bash env/docker/run.sh binfmt         arm64 エミュレーション登録（再起動で消えるので up の前処理で毎回確認）
#   bash env/docker/run.sh build [svc]    イメージ構築（省略時: quant eval arm64）
#   bash env/docker/run.sh up             3 コンテナをアイドル起動（sleep infinity）
#   bash env/docker/run.sh down           停止
#   bash env/docker/run.sh smoke          疎通確認（GPU・arm64・Python・GGUF 読み込み）
#   bash env/docker/run.sh server-quant <gguf> [extra args]   ① で llama-server を起動（port 8081）
#   bash env/docker/run.sh server-arm64 <gguf> [extra args]   ③ で llama-server を起動（port 8083）
#   bash env/docker/run.sh sh <svc>       コンテナにシェルで入る
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
COMPOSE=(docker compose -f "$HERE/compose.yaml")

ensure_binfmt() {
  if [ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]; then
    echo "[binfmt] qemu-aarch64 registered"
  else
    echo "[binfmt] registering arm64 (tonistiigi/binfmt)"
    docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null
  fi
  docker run --rm --platform linux/arm64 debian:bookworm-slim uname -m | grep -q aarch64 \
    && echo "[binfmt] arm64 container OK"
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  binfmt) ensure_binfmt ;;
  build)  "${COMPOSE[@]}" build "$@" ;;
  up)     ensure_binfmt; "${COMPOSE[@]}" up -d quant eval arm64 ;;
  down)   "${COMPOSE[@]}" down ;;
  ps)     "${COMPOSE[@]}" ps ;;
  sh)     "${COMPOSE[@]}" exec "${1:?svc}" bash ;;
  smoke)  bash "$HERE/smoke.sh" ;;
  server-quant)
    gguf="${1:?gguf path (container 内 /work/models/...)}"; shift || true
    "${COMPOSE[@]}" exec -d quant llama-server -m "$gguf" --host 0.0.0.0 --port 8080 -ngl 99 "$@"
    echo "quant llama-server starting: http://127.0.0.1:8081/health" ;;
  server-arm64)
    gguf="${1:?gguf path}"; shift || true
    "${COMPOSE[@]}" exec -d arm64 llama-server -m "$gguf" --host 0.0.0.0 --port 8080 -t 4 "$@"
    echo "arm64 llama-server starting: http://127.0.0.1:8083/health" ;;
  *) sed -n 2,11p "$0" ;;
esac
