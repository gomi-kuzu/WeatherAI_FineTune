#!/usr/bin/env bash
# Gemma 4 E2B のテキスト専用 GGUF を取得する（mmproj は取得しない: 画像入力なし）
# 使い方: bash models/download.sh   （再実行は差分のみ取得、冪等）
set -euo pipefail
cd "$(dirname "$0")"
BASE="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main"
FILES=(
  gemma-4-E2B-it-BF16.gguf          # 参照モデル（量子化の元）
  gemma-4-E2B-it-UD-Q4_K_XL.gguf    # QAT 4bit（実機候補の本命）
  imatrix_unsloth.gguf_file         # 配布元の imatrix（自作 imatrix との比較用）
  mtp-gemma-4-E2B-it.gguf           # MTP ヘッド（投機的生成の検証用）
)
for f in "${FILES[@]}"; do
  echo "== $f"
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "$f" "$BASE/$f"
done
sha256sum *.gguf *.gguf_file > SHA256SUMS.txt
echo "done"
