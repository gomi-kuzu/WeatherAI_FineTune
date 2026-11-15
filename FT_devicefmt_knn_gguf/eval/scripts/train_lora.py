#!/usr/bin/env python3
"""LoRA FineTune（提案書 016 §4-2）: unsloth/gemma-4-E2B-it に「生文面 → 整数\\n定型理由」を学習させ、マージして F16 GGUF にする。
  （⑤ gemma4-train コンテナ内）
  python eval/scripts/train_lora.py --run A --data eval/data/sft/sft_A.jsonl [--max-rows N] [--epochs 1]
出力: models/ft/<run>/lora/（アダプタ）、models/ft/<run>/merged/（HF 16bit）、models/ft/<run>/gemma-4-E2B-wr-<run>-F16.gguf、train_log.json
方針:
  - 生文面（チャットテンプレート無し）: text = prompt + completion + EOS。損失は completion 部分だけ（prompt のトークンは -100）。
  - T4（sm_75）は bf16 の行列積が遅いので fp16。ベースは 4bit（QLoRA）で読み、マージ時に 16bit に戻す（Unsloth の merged_16bit）。
  - 乱数 seed 0。ハイパーパラメータは提案書 016 の既定値。
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--base", default=str(ROOT / "models" / "hf" / "gemma-4-E2B-it"))
    ap.add_argument("--out", default=str(ROOT / "models" / "ft"))
    ap.add_argument("--max-rows", type=int, default=0); ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--r", type=int, default=16); ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--grad-acc", type=int, default=2)
    ap.add_argument("--load-4bit", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gguf", action="store_true"); ap.add_argument("--eval-rows", type=int, default=1000)
    a = ap.parse_args()
    out = pathlib.Path(a.out) / a.run; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=a.base, max_seq_length=a.max_len, dtype=None, load_in_4bit=bool(a.load_4bit), local_files_only=True)
    model = FastLanguageModel.get_peft_model(
        model, r=a.r, lora_alpha=a.alpha, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=a.seed)
    eos = tokenizer.eos_token or "<eos>"

    ds = load_dataset("json", data_files=a.data, split="train")
    if a.max_rows:
        ds = ds.select(range(min(a.max_rows, len(ds))))
    ds = ds.shuffle(seed=a.seed)
    ev = ds.select(range(min(a.eval_rows, len(ds)))); tr = ds.select(range(min(a.eval_rows, len(ds)), len(ds)))

    def to_text(ex):   # 生文面 + 回答 + EOS（損失は completion のみ: completion_only_loss）
        return {"prompt": ex["prompt"], "completion": ex["completion"] + eos}
    tr = tr.map(to_text, remove_columns=[c for c in tr.column_names if c not in ("prompt", "completion")])
    ev = ev.map(to_text, remove_columns=[c for c in ev.column_names if c not in ("prompt", "completion")])

    cfg = SFTConfig(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_acc,
        num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=50, eval_strategy="steps", eval_steps=500, save_strategy="no", seed=a.seed,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit", weight_decay=0.0, max_length=a.max_len, completion_only_loss=True, packing=False,
        dataset_num_proc=4, report_to="none")
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=tr, eval_dataset=ev, args=cfg)
    # 損失が completion だけに掛かっているかの確認（先頭 1 件のラベルで -100 の個数）
    b = trainer.train_dataset[0]
    n_lab = sum(1 for x in b["labels"] if x != -100) if "labels" in b else None
    print(f"[check] input_ids={len(b['input_ids'])} completion tokens in loss={n_lab}", flush=True)
    stats = trainer.train()
    log = {"run": a.run, "data": a.data, "n_train": len(tr), "n_eval": len(ev), "args": vars(a), "train_stats": stats.metrics,
           "log_history": trainer.state.log_history, "completion_tokens_in_loss_example": n_lab, "bf16": torch.cuda.is_bf16_supported()}
    model.save_pretrained(str(out / "lora")); tokenizer.save_pretrained(str(out / "lora"))
    model.save_pretrained_merged(str(out / "merged"), tokenizer, save_method="merged_16bit")
    log["merge_s"] = time.time() - t0
    if not a.no_gguf:
        gguf = out / f"gemma-4-E2B-wr-{a.run}-F16.gguf"
        conv = os.environ.get("LLAMA_CPP_DIR", "/opt/llama.cpp") + "/convert_hf_to_gguf.py"
        r = subprocess.run([sys.executable, conv, str(out / "merged"), "--outfile", str(gguf), "--outtype", "f16"], capture_output=True, text=True)
        log["gguf"] = {"path": str(gguf), "returncode": r.returncode, "stderr_tail": r.stderr[-2000:]}
        print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
    log["total_s"] = time.time() - t0
    json.dump(log, open(out / "train_log.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: log[k] for k in ("run", "n_train", "train_stats", "total_s")}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
