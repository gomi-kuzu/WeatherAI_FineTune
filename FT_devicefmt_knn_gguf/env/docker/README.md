# env/docker — Gemma 4 × QRB2210 検証環境（提案書 revB §4 の実装）

ホスト OS で動かすのは `docker compose`（`run.sh` 経由）だけ。llama.cpp・Python はすべてコンテナ内。

| サービス | イメージ | 役割 | 入口 |
|---|---|---|---|
| `quant` | `gemma4-quant:v0.6.0`（nvidia/cuda 12.8 + llama.cpp v0.6.0 CUDA sm_75） | ① BF16 参照推論・imatrix・量子化・KL/PPL・llama-server | `127.0.0.1:8081` |
| `eval` | `gemma4-eval:latest`（python 3.12 + pandas/sklearn） | ② 評価データ作成・プロンプト生成・採点 | — |
| `eval-lmeval` | `gemma4-eval:lmeval`（+torch, lm-eval） | ②' 補助ベンチ（任意 `--profile lmeval`） | — |
| `arm64` | `gemma4-arm64:v0.6.0`（debian bookworm aarch64 + llama.cpp v0.6.0 armv8-a） | ③ 実機同一バイナリの疎通・RSS、④ 実機にそのまま配布 | `127.0.0.1:8083` |

## 手順
```bash
bash models/download.sh            # GGUF 取得（BF16 / QAT UD-Q4_K_XL / imatrix / MTP）
bash env/docker/run.sh build       # 3 イメージ構築（quant は CUDA ビルドで 20〜40 分）
bash env/docker/run.sh up          # binfmt 確認 → 3 コンテナをアイドル起動
bash env/docker/run.sh smoke       # 疎通確認
bash env/docker/run.sh server-quant /work/models/gemma-4-E2B-it-BF16.gguf   # ① で API サーバー
bash env/docker/run.sh server-arm64 /work/models/gemma-4-E2B-it-UD-Q4_K_XL.gguf -c 2048
```

## 実機（UNO Q 4GB）への配布
```bash
docker save gemma4-arm64:v0.6.0 | gzip > gemma4-arm64_v0.6.0.tar.gz   # → 実機へ転送して docker load
# 実機側: docker run --rm -p 8080:8080 -v /path/models:/work/models gemma4-arm64:v0.6.0 \
#           llama-server -m /work/models/gemma-4-E2B-it-UD-Q4_K_XL.gguf -t 4 -c 2048
```

## 固定した版
- llama.cpp `v0.6.0`（commit は各イメージの `/app/LLAMA_CPP_COMMIT`）
- arm64 ビルド: `-march=armv8-a -mtune=cortex-a53`、OpenMP ON、KleidiAI OFF、静的リンク。
  dotprod / i8mm / fp16 演算を使う経路を含めない（A53 で SIGILL を出さないため）。
- Python 依存: `requirements-eval.txt` / `requirements-lmeval.txt`
