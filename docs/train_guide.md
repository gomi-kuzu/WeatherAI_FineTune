# ファインチューニング手順ガイド（train_guide）

本ドキュメントは、[eval/data/weather_sft](../eval/data/weather_sft) の SFT データを用いて、
Hugging Face の PEFT（LoRA / QLoRA）で Gemma 系 LLM をファインチューニングするための
**準備・環境構築手順**をまとめたものです。

> [!WARNING]
> **本ガイドの手順は現段階で未実施・未検証です。**
> 実際のインストール、学習、評価はまだ行っていません。
> バージョン番号・VRAM 目安・コマンドは一般的な想定に基づく暫定値であり、
> 実行時にエラーや追加設定が必要になる可能性があります。
> 検証後に内容を更新してください。

---

## 1. 前提知識・注意点

- **GPU が必要**: LoRA / QLoRA であっても学習には CUDA 対応 GPU が必要です。
- **モデルの選定**:
  - 本リポジトリ作成時点で「gemma4」という名称のモデルは確認できていません。
  - 実在する候補: `google/gemma-2-2b-it`, `google/gemma-2-9b-it`, `google/gemma-3-*` など。
  - **まずは軽量な 2B クラス（例: `gemma-2-2b-it`）で動作確認**することを推奨します。
- **VRAM の目安（暫定・未検証）**:

  | モデル規模 | 手法 | VRAM 目安 |
  |---|---|---|
  | 2B 系 | LoRA / QLoRA | 約 8 GB 〜 |
  | 7B〜9B 系 | QLoRA(4bit) | 約 16〜24 GB |

- **Gemma はライセンス同意が必要**: Hugging Face のモデルページで利用規約に同意し、
  アクセストークンを取得する必要があります（手順は後述）。

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
huggingface-cli login   # 発行したトークンを貼り付け
```

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
- 役割分担（想定）:
  - `train.jsonl` … 学習
  - `dev.jsonl` … 検証（early stopping / ハイパラ調整）
  - `test.jsonl` … 最終評価
- データ仕様の詳細は [eval/data/WEATHER_SFT.md](../eval/data/WEATHER_SFT.md) を参照。

---

## 8. 次のステップ（未着手）

以下はまだ着手していません。環境構築が完了してから進めます。

- [ ] 上記ライブラリの実インストールと GPU 認識の検証
- [ ] 学習スクリプト（LoRA / QLoRA）の作成
  - チャット形式の読み込み、assistant 部分のみを損失対象にする設定（completion-only）
- [ ] 学習の実行と `dev.jsonl` での検証
- [ ] `test.jsonl` での評価、推論スクリプトの作成

---

## 付記

- 本ガイドに記載したバージョン・VRAM 目安・コマンドは**未検証の暫定情報**です。
- 実行時の結果・修正点が判明したら、本ドキュメントを随時更新してください。
