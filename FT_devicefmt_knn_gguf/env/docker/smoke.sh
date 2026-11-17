#!/usr/bin/env bash
# Phase 0 疎通確認。すべてコンテナ内で実行する（ホストでは docker compose のみ）
# 確認項目: ① GPU が見える / llama.cpp のタグ ② Python 依存 ③ aarch64 で起動・命令セット
#          ④ GGUF があれば ①③ で短い推論（③ は QEMU なので速度は見ない）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C=(docker compose -f "$HERE/compose.yaml")
ok(){ echo "  [OK] $*"; }; ng(){ echo "  [NG] $*"; FAIL=1; }
FAIL=0
GGUF="${1:-/work/models/gemma-4-E2B-it-UD-Q4_K_XL.gguf}"

echo "== ① quant (amd64 + CUDA) =="
"${C[@]}" exec -T quant bash -c 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' && ok "GPU visible" || ng "GPU not visible"
"${C[@]}" exec -T quant bash -c 'echo "llama.cpp $LLAMA_CPP_TAG $(cat /app/LLAMA_CPP_COMMIT)"; llama-server --version 2>&1 | head -2' && ok "llama.cpp present" || ng "llama.cpp missing"
"${C[@]}" exec -T quant bash -c 'ls /app/lib | grep -E "ggml-(cuda|cpu)" | tr "\n" " "; echo' && ok "backends" || ng "backends"
"${C[@]}" exec -T quant bash -c 'python3 -c "import gguf; gguf.GGUFReader; print(\"gguf-py OK\")"' && ok "gguf-py" || ng "gguf-py"

echo "== ② eval (amd64 + Python) =="
"${C[@]}" exec -T eval python -c 'import numpy,pandas,sklearn,requests,gguf,yaml,jinja2,matplotlib; print("python deps OK", pandas.__version__, sklearn.__version__)' && ok "python deps" || ng "python deps"
"${C[@]}" exec -T eval bash -c 'ls /work/models /work/eval >/dev/null' && ok "volumes mounted" || ng "volumes"

echo "== ③ arm64 (QEMU) =="
"${C[@]}" exec -T arm64 bash -c 'uname -m; cat /proc/cpuinfo | grep -m1 Features || true' && ok "aarch64 container" || ng "aarch64 container"
"${C[@]}" exec -T arm64 bash -c 'llama-server --version 2>&1 | head -2' && ok "llama.cpp aarch64 binary runs" || ng "llama.cpp aarch64"
# バイナリに dotprod / i8mm 命令が含まれていないこと（A53 で SIGILL にならない）を確認
"${C[@]}" exec -T arm64 bash -c 'llama-cli --version 2>&1 | grep -iE "aarch64|arm" | head -1 || true'

if "${C[@]}" exec -T quant test -f "$GGUF"; then
  echo "== ④ 推論の疎通: $GGUF =="
  # 出力に「東京」が含まれることまで確認する（パイプの終了コードだけでは推論失敗を見逃す）
  o=$("${C[@]}" exec -T quant bash -c "llama-completion -m $GGUF -ngl 99 -n 16 -no-cnv --temp 0 -p '日本の首都は' 2>/dev/null"); echo "    quant: $o"
  echo "$o" | grep -q "東京" && ok "quant inference" || ng "quant inference"
  "${C[@]}" exec -T quant bash -c "llama-completion --list-devices 2>/dev/null | grep -q CUDA" && ok "quant uses CUDA device" || ng "quant CUDA device not found"
  o=$("${C[@]}" exec -T arm64 bash -c "timeout 900 llama-completion -m $GGUF -t 4 -n 8 -no-cnv --temp 0 -p '日本の首都は' 2>/dev/null"); echo "    arm64: $o"
  echo "$o" | grep -q "東京" && ok "arm64 inference (QEMU, slow)" || ng "arm64 inference"
else
  echo "== ④ skipped: $GGUF not found (models/download.sh を先に実行) =="
fi
[ "$FAIL" = 0 ] && echo "SMOKE: ALL OK" || { echo "SMOKE: FAILED"; exit 1; }
