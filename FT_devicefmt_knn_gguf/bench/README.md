# bench — 速度計測と予測
- `predict.py`: GGUF のテンソル表から 1 トークンあたり読み出し量を集計し、実機の decode tok/s と 1 判定の所要秒を見積もる（Phase 3）。
- `calibration.yaml`: 実機 llama-bench の実測値（B・prefill tok/s）。Phase 4 で記録して predict.py の既定を置き換える。
- `llama-bench` の実行例（③ QEMU では速度は無意味。④ 実機で）:
  `llama-bench -m /work/models/gemma-4-E2B-it-UD-Q4_K_XL.gguf -t 4 -p 128,512 -n 64 -o csv > bench/results/unoq_<date>.csv`
