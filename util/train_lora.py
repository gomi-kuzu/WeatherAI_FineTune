#!/usr/bin/env python
"""Gemma 4 (E2B) を QLoRA でファインチューニングする学習スクリプト.

- 対象データ: eval/data/weather_sft/{train,dev}.jsonl （messages 形式のチャット）
- 手法: QLoRA (4bit nf4) + LoRA（テキスト層のみ）
- 損失: prompt/completion 形式で completion（assistant=model ターン）のみを対象

使い方:
    # スモークテスト（少量データで配線確認）
    python util/train_lora.py --smoke

    # 本番学習
    python util/train_lora.py

GPU は環境変数 CUDA_VISIBLE_DEVICES で選択する（例: export CUDA_VISIBLE_DEVICES=1）。
"""
from __future__ import annotations

import argparse
import os
import subprocess

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

# テキストデコーダ（language_model）配下の線形層のみを LoRA 対象にする正規表現。
# vision_tower / audio_tower は気象テキストタスクでは使わないため除外する。
TEXT_TARGET_MODULES = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)


class GPUProgressCallback(TrainerCallback):
    """学習中の進捗と GPU メモリ使用量を表示する。"""

    def __init__(self, log_every: int = 5) -> None:
        self.log_every = log_every

    def _gpu_status(self) -> str:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            rows = []
            for line in out.strip().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    idx, util, used, total = parts[:4]
                    rows.append(f"GPU{idx}: util={util}% mem={used}/{total}MiB")
            return "; ".join(rows) if rows else "GPU status unavailable"
        except Exception:
            return "GPU status unavailable"

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % self.log_every == 0:
            print(
                f"[progress] step={state.global_step}/{state.max_steps or 'NA'} "
                f"lr={state.log_history[-1].get('learning_rate', 'N/A') if state.log_history else 'N/A'} "
                f"| {self._gpu_status()}"
            )


def detect_gpu_mem_mb() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        vals = [int(line.strip()) for line in out.splitlines() if line.strip()]
        return min(vals) if vals else None
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for Gemma 4 E2B")
    p.add_argument("--model", default="google/gemma-4-E2B-it")
    p.add_argument("--train-file", default="eval/data/weather_sft/train.jsonl")
    p.add_argument("--eval-file", default="eval/data/weather_sft/dev.jsonl")
    p.add_argument("--output-dir", default="outputs/gemma4-weather-qlora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-length", type=int, default=896)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="少量データ・少ステップで配線のみ検証する",
    )
    p.add_argument(
        "--low-vram",
        action="store_true",
        help="8GB クラスGPU向けに VRAM ピークを抑える"
        "（batch_size=1, grad_accum=16, max_length=896）。実効バッチは維持。",
    )
    args = p.parse_args()
    gpu_total_mb = detect_gpu_mem_mb()
    if gpu_total_mb is not None and gpu_total_mb <= 8192 and not args.low_vram:
        print(f"[info] 8GB GPU detected (total={gpu_total_mb}MiB). Enabling low-vram mode by default.")
        args.low_vram = True
    if args.low_vram:
        args.batch_size = 1
        args.grad_accum = 16
        args.max_length = min(args.max_length, 896)
    return args


def to_prompt_completion(example: dict) -> dict:
    """messages をTRLの会話型 prompt/completion 形式へ変換する.

    prompt には system+user、completion には assistant（= model ターン）を入れる。
    completion_only_loss=True により prompt 部分は損失計算から除外される。
    """
    messages = example["messages"]
    prompt = [m for m in messages if m["role"] != "assistant"]
    completion = [m for m in messages if m["role"] == "assistant"]
    return {"prompt": prompt, "completion": completion}


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU が見つかりません。GPU 環境で実行してください。")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data_files = {"train": args.train_file, "eval": args.eval_file}
    ds = load_dataset("json", data_files=data_files)
    ds = ds.map(to_prompt_completion, remove_columns=ds["train"].column_names)

    output_dir = args.output_dir
    if args.smoke:
        # 少量データ・数ステップのみ。本番成果物を上書きしないよう別ディレクトリへ。
        ds["train"] = ds["train"].select(range(min(64, len(ds["train"]))))
        ds["eval"] = ds["eval"].select(range(min(16, len(ds["eval"]))))
        output_dir = "outputs/smoke"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TEXT_TARGET_MODULES,
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1.0 if args.smoke else args.epochs,
        max_steps=10 if args.smoke else -1,
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5 if args.smoke else 50,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10 if args.smoke else 100,
        save_strategy="steps",
        save_steps=10 if args.smoke else 200,
        save_total_limit=3,
        report_to="none",
        seed=args.seed,
        dataset_kwargs={"add_special_tokens": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[GPUProgressCallback(log_every=5)],
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"学習完了。LoRA アダプタを {output_dir} に保存しました。")


if __name__ == "__main__":
    args = parse_args()
    print("[config] model=", args.model)
    print("[config] output_dir=", args.output_dir)
    print("[config] epochs=", args.epochs)
    print("[config] max_length=", args.max_length)
    print("[config] batch_size=", args.batch_size)
    print("[config] grad_accum=", args.grad_accum)
    print("[config] low_vram=", args.low_vram)
    print("[config] GPU_total_mb=", detect_gpu_mem_mb())
    main(args)
