#!/usr/bin/env python
"""学習済み LoRA（QLoRA）モデルを test/dev データで評価する.

- ベースモデル + LoRA アダプタを 4bit で読み込み、各 prompt について生成する。
- 生成結果の JSON をパースして weather_risk を取り出し、正解（assistant ターン）と比較する。
- 指標: weather_risk の MAE / RMSE、荒天判定(risk>=50)の適合率・再現率・F1、JSON パース成功率。

使い方:
    # 少数で動作確認
    python util/eval_llm.py --limit 20

    # 全件
    python util/eval_llm.py

GPU は環境変数 CUDA_VISIBLE_DEVICES で選択する（例: export CUDA_VISIBLE_DEVICES=0）。
"""
from __future__ import annotations

import argparse
import json
import math
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

STORM_RISK = 50  # これ以上を「荒天」とみなす weather_risk の閾値


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate QLoRA fine-tuned Gemma 4 on weather SFT")
    p.add_argument("--base-model", default="google/gemma-4-E2B-it")
    p.add_argument("--adapter", default="outputs/gemma4-weather-qlora")
    p.add_argument("--eval-file", default="eval/data/weather_sft/test.jsonl")
    p.add_argument("--limit", type=int, default=0, help="先頭 N 件のみ評価（0=全件）")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 なら greedy。>0 ならサンプリング復号（do_sample=True）。",
    )
    p.add_argument("--top-p", type=float, default=0.9, help="サンプリング時の top-p。")
    p.add_argument("--seed", type=int, default=0, help="サンプリング時の乱数シード。")
    p.add_argument("--out", default="", help="推論結果を JSONL で書き出すパス（任意）")
    return p.parse_args()


def extract_risk(text: str) -> int | None:
    """生成テキストから weather_risk（整数）を取り出す。

    まず JSON としてパースを試み、失敗したら正規表現で拾う。
    """
    # 最初の { ... } ブロックを抜き出す
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    block = text[start : i + 1]
                    try:
                        obj = json.loads(block)
                        val = obj.get("weather_risk")
                        if isinstance(val, (int, float)):
                            return int(round(val))
                    except (json.JSONDecodeError, AttributeError):
                        pass
                    break
    m = re.search(r'"?weather_risk"?\s*[:=]\s*(-?\d+)', text)
    if m:
        return int(m.group(1))
    return None


def gold_risk(messages: list[dict]) -> int | None:
    for m in messages:
        if m["role"] == "assistant":
            r = extract_risk(m["content"])
            if r is not None:
                return r
    return None


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("[info] ベースモデルを 4bit で読み込み中 ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    print(f"[info] LoRA アダプタを読み込み中 ... {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    do_sample = args.temperature > 0.0
    if do_sample:
        torch.manual_seed(args.seed)
        print(f"[info] サンプリング復号: temperature={args.temperature} top_p={args.top_p} seed={args.seed}")
    else:
        print("[info] greedy 復号（do_sample=False）")

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    data = load_dataset("json", data_files={"eval": args.eval_file})["eval"]
    if args.limit and args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))
    n_total = len(data)
    print(f"[info] 評価件数: {n_total}")

    out_fh = open(args.out, "w", encoding="utf-8") if args.out else None

    abs_errs: list[float] = []
    sq_errs: list[float] = []
    parse_ok = 0
    # 荒天判定の混同行列
    tp = fp = fn = tn = 0

    for i, row in enumerate(data):
        messages = row["messages"]
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        gold = gold_risk(messages)

        inputs = tokenizer.apply_chat_template(
            prompt_msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                inputs,
                **gen_kwargs,
            )
        new_tokens = gen[0][inputs.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = extract_risk(text)

        if pred is not None:
            parse_ok += 1
        if pred is not None and gold is not None:
            err = abs(pred - gold)
            abs_errs.append(err)
            sq_errs.append(err * err)
            p_storm = pred >= STORM_RISK
            g_storm = gold >= STORM_RISK
            if p_storm and g_storm:
                tp += 1
            elif p_storm and not g_storm:
                fp += 1
            elif not p_storm and g_storm:
                fn += 1
            else:
                tn += 1

        if out_fh is not None:
            out_fh.write(
                json.dumps(
                    {"index": i, "gold": gold, "pred": pred, "raw": text},
                    ensure_ascii=False,
                )
                + "\n"
            )

        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            mae = sum(abs_errs) / len(abs_errs) if abs_errs else float("nan")
            print(f"[progress] {i + 1}/{n_total}  parse_ok={parse_ok}  MAE(so far)={mae:.2f}")

    if out_fh is not None:
        out_fh.close()

    print("\n==== 評価結果 ====")
    print(f"件数: {n_total}")
    print(f"JSON パース成功率: {parse_ok}/{n_total} = {parse_ok / n_total:.1%}")
    if abs_errs:
        mae = sum(abs_errs) / len(abs_errs)
        rmse = math.sqrt(sum(sq_errs) / len(sq_errs))
        print(f"weather_risk  MAE={mae:.2f}  RMSE={rmse:.2f}  （採点対象 {len(abs_errs)} 件）")
    else:
        print("weather_risk の採点対象がありません（パース失敗または正解欠損）。")

    scored = tp + fp + fn + tn
    if scored:
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else float("nan")
        acc = (tp + tn) / scored
        print(
            f"荒天判定(risk>={STORM_RISK}): "
            f"適合率={prec:.3f} 再現率={rec:.3f} F1={f1:.3f} 正解率={acc:.3f}  "
            f"(TP={tp} FP={fp} FN={fn} TN={tn})"
        )


if __name__ == "__main__":
    main()
