#!/usr/bin/env python3
"""実装I の repo の主指標（気圧の上下動）と、こちらの P2' の手がかり（気圧の低下・高湿度）を、こちらの dev 層化 2,500 行で突き合わせる。
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/w -w /w gemma4-eval python3 -I report/261105_実装Iのrepositoryと比較/check/check_signals.py
入力の 12 点はこちらの 5 分平均（実機と同じ）。実装I の repo は 1 分値の 5 分おきの瞬間値なので、値の大きさ（帯 0.06/0.14 hPa）は比べられない。順位（AUC）だけ比べる。
"""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
d = pd.read_csv("eval/data/baseline/dev_sample.csv")
P = d[[f"P_b{k:02d}" for k in range(12)]].to_numpy(); H = d[[f"RH_b{k:02d}" for k in range(12)]].to_numpy()
dp = np.diff(P, axis=1)
sig = {"気圧の上下動（5 分差の標準偏差）": dp.std(axis=1),
       "気圧の 60 分低下（−dP60）": -(P[:, -1] - P[:, 0]),
       "現在の湿度": H[:, -1]}
tg = {"event（先 60 分 ≥ 0.5 mm）": (d.R60_mm >= 0.5).to_numpy(), "荒天（risk2 ≥ 50 = 5 mm 以上）": (d.risk2_lin >= 50).to_numpy()}
w = d["w_pop"].to_numpy() if "w_pop" in d else None
rows = []
for sn, s in sig.items():
    r = {"指標": sn}
    for tn, y in tg.items():
        r[tn] = round(roc_auc_score(y, s), 3); r[tn + "（母集団重み）"] = round(roc_auc_score(y, s, sample_weight=w), 3) if w is not None else None
    rows.append(r)
out = pd.DataFrame(rows); print(f"n={len(d)} event={int(tg['event（先 60 分 ≥ 0.5 mm）'].sum())} storm={int(tg['荒天（risk2 ≥ 50 = 5 mm 以上）'].sum())}")
print(out.to_string(index=False))
out.to_csv("report/261105_実装Iのrepositoryと比較/check/signal_auc.csv", index=False)
