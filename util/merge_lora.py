#!/usr/bin/env python
"""学習済み LoRA アダプタをベースモデルにマージして safetensors で保存する.

QLoRA で学習したアダプタ（outputs/.../adapter_model.safetensors）を、
bf16 のベース `google/gemma-4-E2B-it` に統合し、GGUF 変換用の
マージ済みチェックポイントを出力する。

使い方:
    python util/merge_lora.py \
        --adapter outputs/gemma4-weather-qlora \
        --out outputs/gemma4-weather-merged

マージは CPU 上で bf16 のまま行う（4bit 量子化は使わない）。
ベース bf16 は約 9.6GB。十分な RAM（目安 20GB 以上）を要する。
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--base", default="google/gemma-4-E2B-it")
    p.add_argument(
        "--adapter",
        default="outputs/gemma4-weather-qlora",
        help="LoRA アダプタ（adapter_config.json を含むディレクトリ）",
    )
    p.add_argument(
        "--out",
        default="outputs/gemma4-weather-merged",
        help="マージ済みモデルの保存先",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 量子化せず bf16 で読み込む（マージ精度を保つため）。CPU に載せる。
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        device_map="cpu",
    )

    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    model.save_pretrained(args.out, safe_serialization=True)

    # トークナイザ／プロセッサも同梱して GGUF 変換時に参照できるようにする。
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    try:
        AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)
    except Exception:
        pass  # プロセッサが無い場合はスキップ

    print(f"マージ完了。{args.out} に保存しました。")


if __name__ == "__main__":
    main()
