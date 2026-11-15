#!/usr/bin/env python3
"""baseline 評価用データの抽出と文面生成（提案書 012 §3-1・§3-3）。

入力: eval/data/dataset_asos/dataset_<地点>.csv（010 で確定した 5 地点）
出力: <out>/
  dev_sample.csv      層化抽出した dev 2,500 行（真値 3 種・重み・5 分平均 × 12 × 3・要約量 8 個）
  dev_prompts.jsonl   同じ行の実機文面（P0 = build_prompt() と同一。1 行 1 サンプル）
  train_feats.csv     train 分割（2019〜2022）の全行の 5 分平均 × 12 × 3・要約量・真値（参照モデルの学習用。文面は作らない）
  strata.json         段ごとの dev 行数・抜いた行数・重み、除外件数、重み付き事象率の検算
実行例（評価コンテナ）:
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/w -w /w gemma4-eval \
      python3 -I eval/scripts/make_baseline_prompts.py

■ 5 分平均の作り方: 過去 60 分の 1 分値を 5 分ずつ 12 区切り（k 番目 = オフセット -60+5k … -56+5k）にし、
  欠測を除いて平均（実機の BucketAccumulator と同じ）。12 区切りのうち 1 つでも全欠測なら除外。
■ 層化: risk2_lin を 5 段（0 / 1-10 / 11-30 / 31-60 / 61-100）に分け、段ごとに N_PER_BIN 行を無作為に抜く。
  w_pop = dataset の weight × (段の dev 行数 ÷ 抜いた行数) で 2023 年の母集団（5 分ごと判定）に戻せる。
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_prompt import build_prompt_p0, vendored_build_prompt, BUILDERS

STATIONS = ["DSM", "IAH", "SEA", "ATL", "BOS"]
BINS = [("b0", -1, 0), ("b1", 0, 10), ("b2", 10, 30), ("b3", 30, 60), ("b4", 60, 100)]   # (名前, 下限(含まず), 上限(含む))
N_PER_BIN = {"b0": 800, "b1": 600, "b2": 500, "b3": 350, "b4": 250}
SEED = 20261015
VARS = ["T", "RH", "P"]
META = ["sample_id", "station", "t0_utc", "split", "weight", "n_past_ok", "R60_mm", "risk1_bin", "risk2_lin", "risk3_peak"]


def bucket_means(df: pd.DataFrame):
    """1 分値 60 列 × 3 変数 → 5 分平均 12 × 3（欠測は除いて平均）。返り値: (DataFrame[36 列], 全欠測バケット数の配列)"""
    out = {}; n_empty = np.zeros(len(df), int)
    for v in VARS:
        arr = df[[f"{v}_{o}" for o in range(-60, 0)]].to_numpy(dtype=float).reshape(len(df), 12, 5)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(arr, axis=2)
        n_empty += np.isnan(m).sum(axis=1)
        for k in range(12):
            out[f"{v}_b{k:02d}"] = m[:, k]
    return pd.DataFrame(out, index=df.index), n_empty


def summary_feats(b: pd.DataFrame) -> pd.DataFrame:
    """revD §3-4 の要約量 8 個（現在値・60/30 分変化）。参照モデルと層別に使う。"""
    f = pd.DataFrame(index=b.index)
    f["P_now"] = b["P_b11"]; f["dP60"] = b["P_b11"] - b["P_b00"]; f["dP30"] = b["P_b11"] - b["P_b05"]
    f["RH_now"] = b["RH_b11"]; f["dRH60"] = b["RH_b11"] - b["RH_b00"]; f["dRH30"] = b["RH_b11"] - b["RH_b05"]
    f["T_now"] = b["T_b11"]; f["dT60"] = b["T_b11"] - b["T_b00"]
    return f


def strat_bin(r2: pd.Series) -> pd.Series:
    s = pd.Series("", index=r2.index)
    for name, lo, hi in BINS:
        s[(r2 > lo) & (r2 <= hi)] = name
    return s


def make_variant(a):
    """既存の層化サンプルから別の文面（P1/P2/P2'/P3）を作る。行の集合・順序は dev_prompts.jsonl と同一。"""
    sample = pd.read_csv(f"{a.out}/{a.split}_sample.csv"); build = BUILDERS[a.style]; out = f"{a.out}/{a.split}_prompts_{a.style}.jsonl"; n = 0
    with open(out, "w", encoding="utf-8") as f:
        for _, r in sample.iterrows():
            t = [r[f"T_b{k:02d}"] for k in range(12)]; h = [r[f"RH_b{k:02d}"] for k in range(12)]; p = [r[f"P_b{k:02d}"] for k in range(12)]
            f.write(json.dumps({"sample_id": r.sample_id, "station": r.station, "t0_utc": r.t0_utc, "strat_bin": r.strat_bin,
                                "risk1": int(r.risk1_bin), "risk2": int(r.risk2_lin), "risk3": int(r.risk3_peak), "R60_mm": float(r.R60_mm),
                                "weight": float(r.weight), "w_pop": float(r.w_pop), "prompt": build(t, h, p, t0_utc=r.t0_utc, station=r.station)}, ensure_ascii=False) + "\n"); n += 1
    print(f"{out}: {n} rows, mean chars {sample.shape[0] and round(sum(len(build([r[f'T_b{k:02d}'] for k in range(12)], [r[f'RH_b{k:02d}'] for k in range(12)], [r[f'P_b{k:02d}'] for k in range(12)], t0_utc=r.t0_utc, station=r.station)) for _, r in sample.head(200).iterrows()) / 200)} (first 200)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="eval/data/dataset_asos")
    ap.add_argument("--out", default="eval/data/baseline")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--style", default="p0", choices=list(BUILDERS), help="p0 以外: 既存の <split>_sample.csv（同じ 2,500 行）から <split>_prompts_<style>.jsonl だけを作る")
    ap.add_argument("--split", default="dev", choices=["dev", "test"], help="test: 最終確認用（016 で最良 1 本だけ 1 回使う。台帳に記録）")
    a = ap.parse_args()
    if a.style != "p0":
        make_variant(a); return
    os.makedirs(a.out, exist_ok=True)
    cols = META + [f"{v}_{o}" for v in VARS for o in range(-60, 0)]
    dev_parts, train_parts, excluded = [], [], {}
    for st in STATIONS:
        df = pd.read_csv(f"{a.dataset_dir}/dataset_{st}.csv", usecols=cols)
        df = df[df.split.isin([a.split, "train"])]
        b, n_empty = bucket_means(df)
        keep = n_empty == 0
        excluded[st] = {a.split: int(((~keep) & (df.split == a.split)).sum()), "train": int(((~keep) & (df.split == "train")).sum())}
        d = pd.concat([df[META], b, summary_feats(b)], axis=1)[keep]
        d["month"] = pd.to_datetime(d.t0_utc).dt.month
        dev_parts.append(d[d.split == a.split]); train_parts.append(d[d.split == "train"])
        print(f"{st}: {a.split} {int((d.split==a.split).sum())} train {int((d.split=='train').sum())} excluded {excluded[st]}", flush=True)
    dev = pd.concat(dev_parts, ignore_index=True); train = pd.concat(train_parts, ignore_index=True)
    dev["strat_bin"] = strat_bin(dev.risk2_lin)
    # 層化抽出
    rng = np.random.default_rng(a.seed); picks = []; strata = {}
    for name, _, _ in BINS:
        pool = dev[dev.strat_bin == name]; n = min(N_PER_BIN[name], len(pool))
        idx = rng.choice(pool.index.to_numpy(), size=n, replace=False); picks.append(idx)
        strata[name] = {"dev_rows": int(len(pool)), "sampled": int(n), "w_strat": len(pool) / n,
                        "by_station": {st: int((pool.loc[idx].station == st).sum()) for st in STATIONS}}
    sample = dev.loc[np.concatenate(picks)].copy()
    sample["w_strat"] = sample.strat_bin.map({k: v["w_strat"] for k, v in strata.items()})
    sample["w_pop"] = sample.weight * sample.w_strat
    sample = sample.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)   # 順序を無作為化（先頭 500 行を部分集合に使う）
    # 文面（P0）と vendored 原本の一致検査
    prompts = []; mism = 0
    for _, r in sample.iterrows():
        t = [r[f"T_b{k:02d}"] for k in range(12)]; h = [r[f"RH_b{k:02d}"] for k in range(12)]; p = [r[f"P_b{k:02d}"] for k in range(12)]
        txt = build_prompt_p0(t, h, p)
        if txt != vendored_build_prompt(t, h, p):
            mism += 1
        prompts.append({"sample_id": r.sample_id, "station": r.station, "t0_utc": r.t0_utc, "strat_bin": r.strat_bin,
                        "risk1": int(r.risk1_bin), "risk2": int(r.risk2_lin), "risk3": int(r.risk3_peak), "R60_mm": float(r.R60_mm),
                        "weight": float(r.weight), "w_pop": float(r.w_pop), "prompt": txt})
    assert mism == 0, f"build_prompt mismatch in {mism} rows"
    # 検算: 重み付き事象率（地点別）が dataset_summary の dev 値に戻るか
    summ = pd.read_csv(f"{a.dataset_dir}/dataset_summary.csv"); summ = summ[summ.split == a.split].set_index("station")
    check = {}
    for st in STATIONS:
        s = sample[sample.station == st]; d = dev[dev.station == st]
        check[st] = {"event_rate_sample_wpop_pct": round(100 * float(np.average(s.risk1_bin == 100, weights=s.w_pop)), 2),
                     "event_rate_split_weight_pct": round(100 * float(np.average(d.risk1_bin == 100, weights=d.weight)), 2),
                     "event_rate_summary_pct": float(summ.loc[st, "event_rate_weighted_pct"]),
                     "risk2_mean_sample_wpop": round(float(np.average(s.risk2_lin, weights=s.w_pop)), 3),
                     "risk2_mean_split_weight": round(float(np.average(d.risk2_lin, weights=d.weight)), 3)}
    sample.to_csv(f"{a.out}/{a.split}_sample.csv", index=False, float_format="%.4f")
    with open(f"{a.out}/{a.split}_prompts.jsonl", "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if a.split == "dev":
        train.to_csv(f"{a.out}/train_feats.csv", index=False, float_format="%.3f")
    info = {"split": a.split, "seed": a.seed, "n_sample": int(len(sample)), "n_dev_after_exclusion": int(len(dev)), "n_train_after_exclusion": int(len(train)),
            "excluded_empty_bucket": excluded, "strata": strata, "vendored_mismatch": mism, "weighted_check": check,
            "prompt_chars_mean": float(np.mean([len(p["prompt"]) for p in prompts]))}
    json.dump(info, open(f"{a.out}/strata{'' if a.split == 'dev' else '_' + a.split}.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(info, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
