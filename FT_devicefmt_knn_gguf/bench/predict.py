"""実機（UNO Q 4GB, Cortex-A53）の速度を実機なしで見積もる簡易モデル（提案書 revB §6-2）
  1) GGUF のテンソル表から「1 トークンあたりに読む重みのバイト数 W」を量子化ごとに集計
     - 埋め込み表（token_embd, per_layer_token_embd 等）は 1 トークンにつき 1 行しか読まないので除外
     - MoE ではない E2B は残り全部を毎トークン読む
  2) 予測 decode tok/s ≒ B / W   （B: 実効メモリ帯域。公開実測 Gemma 3 1B Q4_K_M 722MB → 5.5 tok/s から B≒4.0 GB/s）
  3) 1 判定の所要秒 ≒ prompt_tokens / prefill_tps + completion_tokens / decode_tps
使い方（① または ② コンテナ内）:
  python bench/predict.py models/*.gguf [--bandwidth-gbs 4.0] [--prefill-tps 4.5] [--prompt-tokens 100,300,1700] [--completion-tokens 50]
実機入手後は --bandwidth-gbs / --prefill-tps を llama-bench の実測で校正する（bench/calibration.yaml に記録）。
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np
from gguf import GGUFReader

EMBED_HINTS = ("token_embd", "per_layer_token_embd", "output.weight")  # output.weight は tied の場合 token_embd と同一

def tensor_bytes(path: pathlib.Path):
    r = GGUFReader(str(path))
    rows = []
    for t in r.tensors:
        nbytes = int(t.n_bytes)
        is_embed = any(h in t.name for h in EMBED_HINTS)
        rows.append((t.name, t.tensor_type.name, nbytes, is_embed))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ggufs", nargs="+")
    ap.add_argument("--bandwidth-gbs", type=float, default=4.0, help="実効メモリ帯域 [GB/s]（既定: UNO Q 公開実測からの逆算）")
    ap.add_argument("--prefill-tps", type=float, default=4.5, help="prefill tok/s の仮定（既定: Gemma3 1B 9.7 から E2B 向けに半減）")
    ap.add_argument("--prompt-tokens", default="100,300,1700", help="表現方式 c/b/a のデータ部トークン数")
    ap.add_argument("--completion-tokens", type=int, default=50)
    ap.add_argument("--csv", help="結果 CSV の出力先")
    a = ap.parse_args()
    ptoks = [int(x) for x in a.prompt_tokens.split(",")]
    out = []
    print(f"{'file':44s} {'file_GB':>8s} {'embed_GB':>9s} {'W_GB/tok':>9s} {'decode_tps':>10s} " + " ".join(f"{'sec@'+str(p):>9s}" for p in ptoks))
    for g in a.ggufs:
        p = pathlib.Path(g)
        rows = tensor_bytes(p)
        total = sum(r[2] for r in rows); emb = sum(r[2] for r in rows if r[3]); W = total - emb
        dec = a.bandwidth_gbs * 1e9 / W
        secs = [pt / a.prefill_tps + a.completion_tokens / dec for pt in ptoks]
        print(f"{p.name:44s} {total/1e9:8.2f} {emb/1e9:9.2f} {W/1e9:9.2f} {dec:10.2f} " + " ".join(f"{s:9.1f}" for s in secs))
        out.append({"file": p.name, "file_gb": total / 1e9, "embed_gb": emb / 1e9, "w_gb_per_tok": W / 1e9, "decode_tps_pred": dec,
                    **{f"sec_at_{pt}": s for pt, s in zip(ptoks, secs)}})
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        print("wrote", a.csv)

if __name__ == "__main__":
    main()
