#!/usr/bin/env python3
"""教師ルール（気圧変動を主指標）の素のベースライン性能を計測する。

LLM を学習する前の下限の目安。metadata JSONL から特徴と正解を読み、
気圧変動（P_volatility_hPa）と相対湿度で荒天・降水をどれだけ当てられるかを出す。
Python 標準ライブラリのみ。
"""
import argparse
import json
from pathlib import Path

# make_weather_sft.VOLATILITY_BANDS と対応する気圧変動の帯 [hPa]。
LOW, HIGH = 0.06, 0.14
STORM_RISK = 50   # これ以上を「荒天」とみなす weather_risk の閾値


def load(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            f, o = r["features"], r["outcome"]
            records.append({"vol": f["P_volatility_hPa"], "rh": f["RH"][-1],
                            "risk": o["weather_risk"], "rain": o["R60_mm"] > 0,
                            "storm": o["weather_risk"] >= STORM_RISK})
    return records


def auc(scores, labels):
    """同順位を平均ランクで扱う Mann-Whitney U ベースの AUC。"""
    paired = sorted(zip(scores, labels))
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[k] = rank
        i = j
    n_pos = sum(1 for _, label in paired if label)
    n_neg = len(paired) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = sum(rank for rank, (_, label) in zip(ranks, paired) if label)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def threshold_report(data, thr, name):
    tp = sum(1 for d in data if d["vol"] >= thr and d["storm"])
    fp = sum(1 for d in data if d["vol"] >= thr and not d["storm"])
    fn = sum(1 for d in data if d["vol"] < thr and d["storm"])
    tn = sum(1 for d in data if d["vol"] < thr and not d["storm"])
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else float("nan")
    acc = (tp + tn) / len(data)
    print(f"  [{name}] 荒天判定: 適合率{prec:.2f} 再現率{rec:.2f} F1={f1:.2f} 正解率{acc:.2f}")


def evaluate(split, data):
    n = len(data)
    storm = [d["storm"] for d in data]
    rain = [d["rain"] for d in data]
    vol = [d["vol"] for d in data]
    rh = [d["rh"] for d in data]
    print(f"\n==== {split} (n={n}) ====")
    print(f"荒天(risk>={STORM_RISK})率: {sum(storm)/n:.1%}  降水率: {sum(rain)/n:.1%}")
    print(f"AUC  気圧変動→荒天: {auc(vol, storm):.3f}   気圧変動→降水: {auc(vol, rain):.3f}")
    print(f"AUC  湿度→荒天   : {auc(rh, storm):.3f}   湿度→降水   : {auc(rh, rain):.3f}")
    threshold_report(data, HIGH, f"変動大>={HIGH}(主ルール)")
    threshold_report(data, LOW, f"やや変動>={LOW}")
    bands = (("静穏<{}".format(LOW), lambda v: v < LOW),
             ("やや変動{}-{}".format(LOW, HIGH), lambda v: LOW <= v < HIGH),
             ("変動大>={}".format(HIGH), lambda v: v >= HIGH))
    for label, cond in bands:
        sub = [d for d in data if cond(d["vol"])]
        if sub:
            storm_rate = sum(d["storm"] for d in sub) / len(sub)
            mean_risk = sum(d["risk"] for d in sub) / len(sub)
            print(f"  {label}: n={len(sub):4d}  荒天率{storm_rate:.1%}  平均risk{mean_risk:.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("eval/data/weather_sft"),
                        help="metadata JSONL があるディレクトリ")
    parser.add_argument("--splits", nargs="+", default=["test", "dev"],
                        help="評価する split 名")
    args = parser.parse_args()
    for split in args.splits:
        path = args.data / f"{split}.metadata.jsonl"
        if not path.exists():
            print(f"(skip) {path} がありません")
            continue
        evaluate(split, load(path))


if __name__ == "__main__":
    main()
