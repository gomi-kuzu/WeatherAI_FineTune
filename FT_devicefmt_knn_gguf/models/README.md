# models/ — モデルの置き場（本体は git に入れない）

| パス | 中身 | 取得・生成の方法 | 大きさ |
|---|---|---|---:|
| `gemma-4-E2B-it-BF16.gguf` ほか | 元モデルの GGUF（参照推論・量子化の元・実機候補 QAT 4bit・imatrix・MTP） | `bash models/download.sh`（配布元 `unsloth/gemma-4-E2B-it-GGUF`。`SHA256SUMS.txt` で検算） | 約 13 GB |
| `hf/gemma-4-E2B-it/` | 学習に使う Hugging Face 形式の重み（safetensors、BF16） | `hf download unsloth/gemma-4-E2B-it --local-dir models/hf/gemma-4-E2B-it`（`pip install -U huggingface_hub`。利用条件への同意が必要） | 10.2 GB |
| `ft/<run>/lora/` | 学習した LoRA アダプタ（`train_lora.py` の出力） | `bash eval/scripts/run_ft_queue.sh` | 97 MB |
| `ft/<run>/merged/` | アダプタを足し込んだ 16 bit の HF 重み | 同上 | 10.2 GB |
| `ft/<run>/gemma-4-E2B-wr-<run>-F16.gguf` | マージ後の F16 GGUF（評価の BF16 欄に相当） | 同上 | 9.3 GB |
| `ft/<run>/gemma-4-E2B-wr-<run>-Q4_K_M[-im].gguf` | 4 ビット量子化（imatrix 無し / 有り）。`-im` が実機候補 | `bash eval/scripts/run_ft_eval.sh quant <run>` | 3.4 GB |
| `ft/<run>/train_log.json` | 学習の設定・損失の推移・所要（git に入れてある。B は 2026-10-15 の実績） | `train_lora.py` が書く | 7 KB |
| `ft/<run>/lora/adapter_config.json` | LoRA の設定（r・alpha・対象の行列。git に入れてある） | 同上 | 2 KB |

- `SHA256SUMS.txt` は 2026-10-07 に取得したときの値。`download.sh` を実行すると上書きされる。
- 学習済みの GGUF（FT-B Q4_K_M-im、3.4 GB）は git に入れていない。必要なら README の再現手順で作り直すか、別経路で受け渡す。
