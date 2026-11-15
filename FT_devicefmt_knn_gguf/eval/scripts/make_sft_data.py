#!/usr/bin/env python3
"""LoRA FineTune 用の学習データを train 分割から作る（提案書 016 §4-1）。
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/w -w /w gemma4-eval python3 -I eval/scripts/make_sft_data.py
入力: eval/data/baseline/train_feats.csv（013 で生成。train 分割 543,157 行の 5 分平均 × 12 × 3・要約量・真値）
出力: eval/data/sft/
  sft_A.jsonl   P2' 文面 → risk2 そのもの          {"prompt", "completion", "sample_id", "label", "risk2"}
  sft_B.jsonl   P2' 文面 → 似た入力 200 件の risk2 平均（kNN、同地点 ±6 時間を除外）
  sft_C.jsonl   P0 文面  → risk2 そのもの
  sft_summary.json
■ 層化: wet（risk2 > 0）は全部、dry は N_DRY 件を無作為（seed 固定）。
■ completion = "<整数>\\n<定型の理由>"。理由は要約量のしきい値から機械的に選ぶ（LLM は使わない）。EOS は学習側で付ける。
■ 正解 B の kNN: 要約量 8 個を標準化し、train 全体（dry 間引き後の 1:3 の母集団）の中で近い順に探す。平均は重み無し
  （= 学習母集団での期待値。母集団重みで戻すと値が 0〜5 に潰れるため）。
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_prompt import BUILDERS

FEATS8 = ["P_now", "dP60", "dP30", "RH_now", "dRH60", "dRH30", "T_now", "dT60"]
N_DRY = 45000; SEED = 20261015; K = 200; EXCL_MIN = 6 * 60


def reason(r) -> str:
    """定型の理由（6 パターン）。入力の要約量と矛盾しない。"""
    hum = f"湿度が {r.RH_now:.0f}% と高く" if r.RH_now >= 90 else f"湿度が {r.RH_now:.0f}% で"
    if r.dP60 <= -0.5:
        pres = f"気圧が 60 分で {abs(r.dP60):.1f} hPa 下がっているため"
    elif r.dP60 >= 0.5:
        pres = f"気圧が 60 分で {r.dP60:.1f} hPa 上がっているため"
    else:
        pres = "気圧が安定しているため"
    return f"{hum}、{pres}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="eval/data/baseline/train_feats.csv"); ap.add_argument("--out", default="eval/data/sft")
    ap.add_argument("--n-dry", type=int, default=N_DRY); ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
    T = pd.read_csv(a.train); T["t"] = pd.to_datetime(T.t0_utc).astype("int64") // 60_000_000_000   # 分
    wet = T[T.risk2_lin > 0]; dry = T[T.risk2_lin == 0].sample(n=a.n_dry, random_state=a.seed)
    S = pd.concat([wet, dry]).sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    print(f"train {len(T)} → wet {len(wet)} + dry {len(dry)} = {len(S)}", flush=True)
    # 正解 B: kNN（train 全体から。同地点 ±6h を除外）
    sc = StandardScaler().fit(T[FEATS8]); X_all = sc.transform(T[FEATS8]); X_q = sc.transform(S[FEATS8])
    nn = NearestNeighbors(n_neighbors=a.k + 300, algorithm="kd_tree").fit(X_all)
    dist, ind = nn.kneighbors(X_q)
    t_all, st_all, y_all = T.t.to_numpy(), T.station.to_numpy(), T.risk2_lin.to_numpy(float)
    t_q, st_q = S.t.to_numpy(), S.station.to_numpy()
    B = np.zeros(len(S)); nused = np.zeros(len(S), int)
    for i in range(len(S)):
        c = ind[i]; ok = ~((st_all[c] == st_q[i]) & (np.abs(t_all[c] - t_q[i]) < EXCL_MIN)); sel = c[ok][: a.k]
        B[i] = y_all[sel].mean(); nused[i] = len(sel)
    S["label_B"] = np.floor(B + 0.5).astype(int); S["label_A"] = S.risk2_lin.astype(int)
    S["reason"] = S.apply(reason, axis=1)
    cols = {"A": ("p2h", "label_A"), "B": ("p2h", "label_B"), "C": ("p0", "label_A")}
    stats = {"n": int(len(S)), "n_wet": int(len(wet)), "n_dry": int(len(dry)), "k": a.k, "nused_min": int(nused.min()), "seed": a.seed,
             "label_A_mean": float(S.label_A.mean()), "label_B_mean": float(S.label_B.mean()),
             "label_A_nunique": int(S.label_A.nunique()), "label_B_nunique": int(S.label_B.nunique()),
             "label_B_quantiles": {q: float(S.label_B.quantile(q)) for q in (0.1, 0.5, 0.9, 0.99)},
             "corr_A_B_spearman": float(S.label_A.corr(S.label_B, method="spearman")),
             "reason_counts": {k: int(v) for k, v in S.reason.str.replace(r"[0-9.]+", "N", regex=True).value_counts().items()}}
    for name, (style, lab) in cols.items():
        build = BUILDERS[style]; out = f"{a.out}/sft_{name}.jsonl"; chars = []
        with open(out, "w", encoding="utf-8") as f:
            for _, r in S.iterrows():
                t = [r[f"T_b{k:02d}"] for k in range(12)]; h = [r[f"RH_b{k:02d}"] for k in range(12)]; p = [r[f"P_b{k:02d}"] for k in range(12)]
                prompt = build(t, h, p); comp = f"{int(r[lab])}\n{r.reason}"; chars.append(len(prompt))
                f.write(json.dumps({"prompt": prompt, "completion": comp, "sample_id": r.sample_id, "label": int(r[lab]), "risk2": int(r.risk2_lin)}, ensure_ascii=False) + "\n")
        stats[f"sft_{name}_prompt_chars_mean"] = float(np.mean(chars)); print(f"wrote {out}", flush=True)
    json.dump(stats, open(f"{a.out}/sft_summary.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
