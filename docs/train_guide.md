# ファインチューニング手順ガイド（train_guide）

本ドキュメントは、[eval/data/weather_sft](../eval/data/weather_sft) の SFT データを用いて、
Hugging Face の PEFT（LoRA / QLoRA）で Gemma 系 LLM をファインチューニングするための
**準備・環境構築手順**をまとめたものです。

> [!NOTE]
> **進捗状況（2026-11-03 時点）**
> - ✅ 環境構築（conda 環境・ライブラリ一式）は**完了・検証済み**。
> - ✅ 学習用の HF 重み `google/gemma-4-E2B-it` を取得し、4bit ロードまで**検証済み**。
> - ⬜ 学習スクリプトの作成・学習・評価は**未着手**。
>
> VRAM 目安など一部に一般的な想定値が残っています。検証が進み次第更新してください。

---

## 1. 前提知識・注意点

- **GPU が必要**: LoRA / QLoRA であっても学習には CUDA 対応 GPU が必要です。
- **使用モデル（確定）**: `google/gemma-4-E2B-it`
  - デプロイ先で実行実績のある Gemma 4 E2B を学習ベースにする。
  - `Gemma4ForConditionalGeneration` の**統合マルチモーダルモデル**（text / audio / vision を内包）。
    今回はテキストタスクのため、学習は**テキスト層のみ**を対象にする。
  - `transformers 5.22.1` は gemma4 に対応済み（`Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration`）。
- **GGUF は学習に使えない（重要）**:
  - 手元の `model/gemma-4-E2B_q4_0-it.gguf` は llama.cpp / Ollama 用の**推論専用**形式で、
    さらに 4bit 量子化済みのため勾配計算ができず、PEFT 学習には使えません。
  - 学習には **HF safetensors 版**（`google/gemma-4-E2B-it`）が必要（取得済み）。
  - 学習後に llama.cpp で GGUF へ変換・量子化し、既存のデプロイ環境へ戻す流れにする。
- **VRAM 実測（検証済み）**:

  | 項目 | 手法 | VRAM |
  |---|---|---|
  | モデルロードのみ | 4bit(QLoRA) | **約 6.7 GB**（実測） |
  | bf16 本体サイズ | - | 約 9.6 GB（ディスク） |

  > 12GB クラスの GPU なら QLoRA(4bit) で学習可能な見込み。学習時は活性化メモリが加わる。

- **Gemma はライセンス同意が必要**: Hugging Face のモデルページで利用規約に同意し、
  アクセストークンを取得する必要があります（手順は後述。本環境では取得・ログイン済み）。

---

## 2. 環境の事前確認

学習を始める前に、実行環境の GPU・Python・CUDA を確認します。

```bash
# OS
uname -a

# GPU（枚数・VRAM・ドライバ）
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Python
python3 --version

# CUDA ツールキット（任意）
nvcc --version
```

- 複数 GPU がある場合は、**VRAM の大きい GPU** を学習に割り当てます（後述の `CUDA_VISIBLE_DEVICES`）。
- `nvidia-smi` が見つからない場合は、GPU ドライバのインストールが別途必要です。

---

## 3. 専用の Python 環境を作成

ベース環境を汚さないよう、専用環境を作成します（conda の例）。

```bash
conda create -n weather-ft python=3.11 -y
conda activate weather-ft
```

> venv を使う場合:
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> ```
>
> Python は 3.11 を推奨（ライブラリ対応が安定しているため）。3.12 でも動作する可能性はありますが未検証です。

---

## 4. ライブラリのインストール

> [!NOTE]
> CUDA のバージョン（例: `cu124`）は、お使いの GPU ドライバ・環境に合わせて
> [PyTorch 公式サイト](https://pytorch.org/get-started/locally/) で確認してください。
> 以下のバージョン指定は暫定値です。

```bash
# PyTorch（CUDA バージョンは環境に合わせる。下は CUDA 12.4 の例）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# ファインチューニング一式
pip install "transformers>=4.44" "trl>=0.9" "peft>=0.12" \
            "accelerate>=0.33" "datasets>=2.20" "bitsandbytes>=0.43" \
            sentencepiece
```

インストール後、GPU が認識されているか確認:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 5. Hugging Face の準備（Gemma 利用に必須）

1. モデルページ（例: `https://huggingface.co/google/gemma-2-2b-it`）で**利用規約に同意**する。
2. [アクセストークン発行ページ](https://huggingface.co/settings/tokens) で **read 権限**のトークンを発行する。
3. ログイン:

```bash
pip install -U huggingface_hub
hf auth login   # 発行したトークンを貼り付け
```

> [!NOTE]
> `huggingface_hub` 2.x 系では CLI コマンドが `huggingface-cli` から `hf` に変わりました。
> 旧 `huggingface-cli login` は使えません（`hf auth login` を使用）。
> 確認は `hf auth whoami`、環境変数 `HF_TOKEN` でも代用できます。

---

## 6. 使用 GPU の指定

複数 GPU 環境では、学習に使う GPU を環境変数で固定します。

```bash
# まず番号を確認
nvidia-smi -L

# VRAM の大きい GPU の番号を指定（下は例: 番号 1 を使用）
export CUDA_VISIBLE_DEVICES=1
```

---

## 7. データについて

- 学習データ: [eval/data/weather_sft](../eval/data/weather_sft)
  - `train.jsonl` / `dev.jsonl` / `test.jsonl` / `extra.jsonl`
  - 各行は `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
    形式のチャットデータ。
- この形式は TRL の `SFTTrainer` がそのまま読み込める構造で、**事前の形式変換は不要**です。
  Gemma のチャットテンプレートが自動適用されます。
- **チャットテンプレート仕様（検証済み）**: gemma4 では以下の形式でレンダリングされる。
  - 各ターンは `<|turn>{role}\n...本文...<turn|>` で囲まれる。
  - assistant は **`model` ロール**として出力される（`<|turn>model\n...<turn|>`）。
  - completion-only 学習（assistant 部分のみ損失）では、`<|turn>model\n` 以降を
    ターゲットにする想定。
- 役割分担（想定）:
  - `train.jsonl` … 学習
  - `dev.jsonl` … 検証（early stopping / ハイパラ調整）
  - `test.jsonl` … 最終評価
- データ仕様の詳細は [eval/data/WEATHER_SFT.md](../eval/data/WEATHER_SFT.md) を参照。

---

## 8. 学習スクリプトと実行手順

学習スクリプト: [util/train_lora.py](../util/train_lora.py)（QLoRA, テキスト層のみに LoRA）

### 設計上のポイント（検証済み）

- **QLoRA(4bit nf4)** で読み込み、LoRA を**テキストデコーダ層のみ**に適用する。
  - 対象: `model.language_model.layers.*` の `q/k/v/o_proj` と `gate/up/down_proj`。
  - `vision_tower` / `audio_tower` は気象テキストタスクでは使わないため除外。
- **completion-only 損失**: データを TRL の `prompt`/`completion` 形式に変換し、
  `SFTConfig(completion_only_loss=True)` で assistant(`model`)部分のみを損失対象にする。
  - 本モデルのチャットテンプレートには `{% generation %}` マーカーが無いため、
    `assistant_only_loss` ではなく prompt/completion 分割方式を採用。
  - TRL 1.17.1 では `DataCollatorForCompletionOnlyLM` は廃止済み。
- GPU は `CUDA_VISIBLE_DEVICES` で選択（単一 GPU を想定）。

### スモークテスト（少量データで配線確認）

```bash
export CUDA_VISIBLE_DEVICES=<GPU番号>
python util/train_lora.py --smoke
```

- `train` 64件 / `eval` 16件・10ステップのみ実行し、`outputs/smoke/` に保存。
- 実測: 12GB クラス GPU の単一構成で完走。loss 3.6→1.3、eval_loss≈0.82、
  token accuracy≈0.79 と学習の兆候を確認済み。

### データのトークン長（検証済み）

`train.jsonl` 3000件をチャットテンプレート適用後にトークン化した分布:

| 統計 | 値 |
|---|---|
| 最小 / 平均 / 最大 | 727 / 766 / 826 |
| p95 / p99 | 801 / 818 |

- 全サンプルが 826 トークン以下のため、`--max-length` の既定は **896**（切り捨てゼロ）。

### 8GB GPU 向け（`--low-vram`）

VRAM 8GB クラスの GPU では `--low-vram` を付けると安全に収まる。

```bash
export CUDA_VISIBLE_DEVICES=<GPU番号>
python util/train_lora.py --low-vram            # 本番
python util/train_lora.py --low-vram --smoke    # 配線確認
```

- 内容: `batch_size=1`, `grad_accum=16`, `max_length=896`（実効バッチは維持）。
- 実測: 8GB クラス GPU で **約 7.3GB / 8GB** 使用・完走。
  学習曲線は 12GB 構成とほぼ同等（loss 3.6→1.3, eval_loss≈0.82）。

### 本番学習

```bash
export CUDA_VISIBLE_DEVICES=<GPU番号>
python util/train_lora.py                        # 12GB クラス向け既定
python util/train_lora.py --low-vram             # 8GB クラス向け
```

- 既定の保存先は `outputs/gemma4-weather-qlora/`。
- 主なオプション（`--help` 参照）: `--epochs` / `--lr` / `--lora-r` /
  `--batch-size` / `--grad-accum` / `--max-length` / `--low-vram`。

---

## 9. 次のステップ

- [x] ライブラリの実インストールと GPU 認識の検証（完了）
- [x] 学習用 HF 重み `google/gemma-4-E2B-it` の取得（完了）
- [x] 4bit ロード・チャットテンプレート適用の検証（完了, VRAM 約6.7GB）
- [x] 学習スクリプト（QLoRA, completion-only）の作成（完了）
- [x] スモークテストでのパイプライン検証（完了）
- [ ] 本番学習の実行と `dev.jsonl` での検証
- [ ] `test.jsonl` での評価、推論スクリプトの作成
- [ ] 学習済み LoRA のマージと GGUF への変換（デプロイ向け）

---

## 付記

- VRAM 目安など一部に一般的な想定値が残っています。実行時の結果・修正点が判明したら、
  本ドキュメントを随時更新してください。
- 環境依存の具体値（GPU 名・ホスト名・絶対パス等）は意図的に一般化して記載しています。
