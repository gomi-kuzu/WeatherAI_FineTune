# オフライン気象予測LLMの追加学習

山中でネット接続なしに使う軽量LLM向けの、降水量ベースの荒天リスク学習データ。
Hugging Face Transformers・PEFT（LoRA/QLoRA）・TRL（SFTTrainer）での教師あり追加学習を想定する。

## 0. データの取得と整形

学習用JSONは、次の2段階で公開観測データから作る（詳細は [データ仕様](eval/data/WEATHER_SFT.md)）。

1. **1分値の取得** — [util/download_asos_1min.py](util/download_asos_1min.py)（標準ライブラリのみ）。
   米国空港のASOS 1分値（Iowa State Univ. IEM が再配布）を、地点×年ごとに
   `eval/data/raw/asos_1min/iem_<地点>_<年>.csv` へ保存する。既定は5地点×2019〜2025年。
   気温・露点・現地気圧・1分降水などの生値（°F・inHg・inch）で、欠測は `M`。
   ```bash
   .venv/bin/python util/download_asos_1min.py
   ```
2. **サンプルCSVへの整形** — [util/make_dataset_asos.py](util/make_dataset_asos.py)（pandas・numpyが必要）。
   1分値をUTCの1分格子に並べてSI単位（℃・%・hPa・mm）へ換算し、判定時刻t0を5分刻みで選んで
   「過去60分の観測＋先60分の降水」を1行にした横長CSV `eval/data/dataset_asos/dataset_<地点>.csv` を作る。
   雨なしの時刻は間引いて層化し、年でtrain/dev/test/extraに分ける。
   ```bash
   .venv/bin/python util/make_dataset_asos.py
   ```
3. **学習用JSONへの変換** — [util/make_weather_sft.py](util/make_weather_sft.py)（§5参照）。
   上記CSVから、急な降水を重視して選別し、チャット形式のJSONLを生成する。
   ```bash
   .venv/bin/python util/make_weather_sft.py --overwrite
   ```

データ本体（`eval/data/raw/`・`eval/data/dataset_asos/`・`eval/data/weather_sft/`）は
[.gitignore](.gitignore) でリモートに含めていない。クローン後に学習を始めるには、
上記1→2→3の順にローカルで実行してデータを作る必要がある（容量の都合で数百MB〜GB規模、
取得には数十分かかる）。手順2・3は乱数シードを固定しているため、同じ入力なら再現できる。

## 1. 使用するファイル

|用途|ファイル|件数|
|---|---|---:|
|学習|[train.jsonl](eval/data/weather_sft/train.jsonl)|3,000|
|学習中の検証|[dev.jsonl](eval/data/weather_sft/dev.jsonl)|750|
|学習後の評価|[test.jsonl](eval/data/weather_sft/test.jsonl)|750|
|別枠の追加評価（2025年）|[extra.jsonl](eval/data/weather_sft/extra.jsonl)|300|

各行は `messages` に `system`・`user`・`assistant` を持つ1会話。

- **入力**：過去60分前〜5分前の12点の気温・相対湿度・現地気圧・5分気圧差。
  `user` は列指向JSONで、`station`・`t0_utc` と、`min`（-60..-5）・`T`・`RH`・`P`・`dP` の
  各12要素配列を持つ。`dP` の先頭要素のみ `null`。
- **出力**：`weather_risk`（整数）と `reason`（短い判断理由）のJSON。
- **リスク**：`min(round(今後60分の合計降水量[mm] × 10), 100)` の0〜100整数。例：0.254mm → 3。
- **判断の重み**：`reason` は気圧の上下動（5分ごとの変化のばらつき）を主指標とし、
  相対湿度・気温は補助として扱う。気圧の正味の上下（上がった/下がった）の向きだけでは荒天を判断しない。
- **履歴が1時間に満たないとき**：配列長は常に12点に固定する。まだ無い過去点は
  `T`・`RH`・`P` を最古の実測値で前方パディングし、その区間の `dP` は `0.0`（先頭のみ `null`）にする。
  気圧変動が小さく出るため、データ不足時は低リスク寄り（安全側）に倒れる。
  これは学習分布外の簡便策で精度は保証しない。`null` を `T`・`RH`・`P` に入れないこと。
- **メタデータ**：同名の `.metadata.jsonl` は正解照合・分析用。学習入力には使用しない。
  未来の降水量や選別カテゴリを入力すると情報漏洩になる。

## 2. モデルと環境の準備

GPU環境に `torch`、`transformers`、`datasets`、`peft`、`trl`、`accelerate` を用意する。
4bit QLoRAを使う場合は、GPU・実行環境に対応した `bitsandbytes` も必要。
モデルが利用制限付きの場合は、Hugging Faceで利用条件への同意と認証を済ませておく。

Gemma 4等は**実在する対象モデルの正確なID**を指定し、そのモデルカードに従って
対応するライブラリ版、モデルクラス、チャットテンプレートを選ぶこと。
以下は `AutoModelForCausalLM` に対応するテキスト生成モデルの例。
マルチモーダルモデルは専用クラス・processorが必要な場合があり、そのまま流用しないこと。

## 3. PEFT + TRLでの学習の流れ

会話を `prompt` / `completion` に分け、**assistantの回答だけ**を損失対象にする。
次は対応するTRL版で実行するLoRAの最小例（リポジトリには学習依存をインストールしていない）。

```python
import os

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

model_id = os.environ["MODEL_ID"]  # 対象の正確なHugging FaceモデルID
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

data = load_dataset("json", data_files={
    "train": "eval/data/weather_sft/train.jsonl",
    "validation": "eval/data/weather_sft/dev.jsonl",
})
data = data.map(
    lambda row: {"prompt": row["messages"][:-1],
                 "completion": row["messages"][-1:]},
    remove_columns=["messages"],
)

model = AutoModelForCausalLM.from_pretrained(model_id)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    peft_config=LoraConfig(
        task_type="CAUSAL_LM", target_modules="all-linear",
        r=16, lora_alpha=32, lora_dropout=0.05,
    ),
    args=SFTConfig(
        output_dir="outputs/weather-lora",
        completion_only_loss=True,
        max_length=4096,
        packing=False,
        learning_rate=1e-4,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        eval_strategy="steps", eval_steps=100,
        save_steps=100,
        report_to="none",
    ),
)
trainer.train()
trainer.save_model("outputs/weather-lora/final")
tokenizer.save_pretrained("outputs/weather-lora/final")
```

### 学習前の確認

- **チャット形式**：`tokenizer.apply_chat_template()` で1会話を確認する。
  systemロール非対応のテンプレートでは、system本文を先頭user本文に統合する。
  推論時にも同じ統合方法を使い、独自の区切りトークンを追加しないこと。
- **長さ**：4,096は仮の上限。テンプレート適用後の実トークン長を測り、
  入力や回答末尾のJSONが切れない長さに調整する。
- **メモリ**：上の例は非量子化LoRA。VRAM不足ならモデルカードの対応範囲で
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)` を指定してロードし、
  PEFTの `prepare_model_for_kbit_training(model)` を学習前に適用するQLoRAへ変更する。
  計算dtypeはGPUのBF16/FP16対応に合わせ、学習時に推論用の `device_map="auto"` を安易に使わないこと。
- **互換性**：TRLの引数名やモデル対応は版により異なる。動作確認した版を固定すること。
  この例は実際のGemmaモデルでの学習実行までは検証していない。

## 4. 保存・オフライン推論・評価

- 保存されるのは原則**LoRAアダプター**。ベースモデルと同一リビジョン、tokenizer、
  アダプターを端末へ事前配置する。PEFTでベースにアダプターを読み込み、
  ローカルパスと `local_files_only=True` を使えばネットなしでロードできる。
- 推論では学習時と同じsystem指示・user観測を与え、assistantの正解文は渡さない。
  テンプレートの `add_generation_prompt=True` と `do_sample=False` を基本にする。
- JSONを解析し、2項目の存在、`weather_risk` の整数型と0〜100の範囲を検査する。
- testでJSON出力成功率、リスクのMAE、急変例の見逃しを評価する。
  「高リスク」の閾値はdevで決め、testでは変更しない。testやextraを学習に混ぜないこと。

### デプロイ時のプロンプト例

system は学習時と同一文（[util/make_weather_sft.py](util/make_weather_sft.py) の `SYSTEM`）をそのまま使う。
user はセンサー列を学習と同じ列指向JSONに組み立てる。BME280などで気温・相対湿度・現地気圧を
5分間隔で12点そろえ、`dP` は直前点との気圧差を自分で計算する（先頭のみ `null`）。

```text
# system（学習時のSYSTEMをそのまま貼る）
山中オフライン用の気象予測。過去1時間・5分間隔のT(気温℃)、RH(相対湿度%)、P(現地気圧hPa)、dP(5分気圧差)から今後1時間の荒天リスクを出す。各配列は-60..-5分の順。出力はJSONで weather_risk(整数0-100) と reason(短い理由) の2項目のみ。weather_riskは今後1時間の合計降水量mm×10で100上限、降水量の指標で降水確率ではない。判断は気圧の細かな上下動(dPのばらつき)を最重視し、変動が大きいほど高リスク、平坦なら低リスク。湿度と気温は補助で気圧変動より優先しない。気圧が正味で上下した向きだけでは判断しない。雷・突風・雨雪は断定しない。reasonは観測の変化と意味を簡潔に。未来値を既知として述べず、注意書きやレーダー確認案内は不要。
```

```json
// user（1行のJSON文字列として渡す。station は任意ラベル、t0_utc はUTC）
{"station": "FIELD", "t0_utc": "2026-11-03T05:00Z", "min": [-60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5], "T": [8.21, 8.19, 8.15, 8.12, 8.08, 8.05, 8.0, 7.96, 7.93, 7.9, 7.88, 7.85], "RH": [72.4, 72.9, 73.5, 74.1, 74.8, 75.6, 76.3, 77.0, 77.8, 78.5, 79.1, 79.8], "P": [912.35, 912.41, 912.33, 912.5, 912.28, 912.47, 912.3, 912.52, 912.27, 912.55, 912.22, 912.58], "dP": [null, 0.06, -0.08, 0.17, -0.22, 0.19, -0.17, 0.22, -0.25, 0.28, -0.33, 0.36]}
```

推論コードの最小例（オフライン・LoRA読み込み）:

```python
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "/models/base"          # ベースモデルのローカルパス（学習と同一リビジョン）
adapter = "outputs/weather-lora/final"
tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True)
model = PeftModel.from_pretrained(model, adapter, local_files_only=True).eval()

system = open("prompt_system.txt", encoding="utf-8").read().strip()  # 上のsystem文
user = {"station": "FIELD", "t0_utc": "2026-11-03T05:00Z",
        "min": [-60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5],
        "T": t_list, "RH": rh_list, "P": p_list, "dP": dp_list}  # 各12要素、dP先頭はNone
messages = [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]

inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
with torch.no_grad():
    out = model.generate(inputs, max_new_tokens=256, do_sample=False)
text = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
result = json.loads(text)                 # {"weather_risk": int, "reason": str}
assert isinstance(result["weather_risk"], int) and 0 <= result["weather_risk"] <= 100
```

systemロール非対応のテンプレートでは、system本文を先頭user本文に統合する（§3の注意と同じ）。
履歴が1時間に満たないときは §1 の前方パディング規約で12点に補う。

急変優先のデータなので自然な発生頻度とは異なる。また、荒天リスクは**降水確率ではない**。
理由文は観測傾向のルールベース要約で、数値リスクの予測性能を保証する説明ではない。
山地・標高変化への適用条件や選別方法は [データ仕様](eval/data/WEATHER_SFT.md) を参照のこと。

## 5. データの再生成

生成器は [util/make_weather_sft.py](util/make_weather_sft.py)（Python標準ライブラリのみ）。
リポジトリのルートで次を実行する。

```bash
.venv/bin/python util/make_weather_sft.py --overwrite
.venv/bin/python -m unittest util.test_make_weather_sft -v
```

`--overwrite` は既存の生成データを上書きする。元のASOS CSVは変更しない。

## ライセンス

このリポジトリのコードは [Apache License 2.0](LICENSE) で提供する。
元の観測データ（ASOS 1分値）は Iowa State University IEM が再配布するもので、
本リポジトリには含めず、各自の取得・利用条件に従うこと。