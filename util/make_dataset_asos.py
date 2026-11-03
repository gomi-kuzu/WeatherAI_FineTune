#!/usr/bin/env python3
"""ASOS 1 分値 CSV から、学習・評価用データセット（案 2 横長: 1 行 = 1 サンプル）を作る。

依存: pandas, numpy
入力: <raw>/iem_<地点>_<年>.csv（download_asos_1min.py の出力）
出力: <out>/dataset_<地点>.csv（地点ごと 1 ファイル）、<out>/dataset_summary.csv（件数の一覧）
実行例（評価コンテナ）:
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/w -w /w gemma4-eval \
      python3 -I eval/scripts/make_dataset_asos.py
  ローカルに pandas があれば  python3 eval/scripts/make_dataset_asos.py  でも同じ。

■ 1 サンプルの形
  判定時刻 t0（UTC、5 分刻み）を挟む ±60 分の 1 分値。
    入力側: 過去 60 分 [t0-60, t0)  の 温度 T [°C]・相対湿度 RH [%]・現地気圧 P [hPa]・降水 R [mm/分]
    正解側: 先 60 分 (t0, t0+60] の 降水 R [mm/分] と、そこから作る降水危険度 3 種（いずれも 0〜100 の整数。四捨五入）
      risk1_bin  : R60（先 60 分の合計雨量 [mm]）が 0.5 mm 以上なら 100、それ以外 0
      risk2_lin  : R60 × 10（100 を超えたら 100）
      risk3_peak : 「最も強く降った区間の強さ」。幅 w 分の区間（1 分ずらしで全部）の雨量の最大値を
                   「w 分で RISK3_INTENSITY × w mm（= 12 mm/h）なら 100」に換算して上限 100 とし、
                   それを RISK3_WINDOWS の各幅（既定 5, 10, 15, 30 分）で求めて平均する。
                   短時間に集中した雨は 5 分の点が高く、降り続く雨は 30 分の点も高くなる。
                   （009 版の risk3_max5 = 5 分固定区間の最大 × 100 は、雨量計の刻み 0.254 mm のため
                     25/51/76/100 の 4 段しか取れなかった。幅を複数持って平均すると段差が細かくなる。
                     RISK3_WINDOWS = (5,) にすれば 009 版とほぼ同じ、(10,) にすれば「10 分最大」になる）
  列の並び:
    sample_id, station, t0_utc, split, weight, n_past_ok, n_fut_ok, R60_mm, risk1_bin, risk2_lin, risk3_peak,
    T_-60 … T_-1, RH_-60 … RH_-1, P_-60 … P_-1, R_-60 … R_-1, R_+1 … R_+60
  欠測は空欄。危険度の計算では先 60 分の欠測分を 0 mm として扱う。

■ サンプルの選び方（層化）
  ・有効条件: 過去 60 分のうち 48 分以上に T/RH/P が揃い、先 60 分のうち 54 分以上に R がある t0 だけ使う
  ・先 60 分に雨が記録された t0（R60 > 0、wet）は全部採る
  ・雨が無い t0（R60 = 0、dry）は wet の DRY_RATIO 倍だけ無作為に採る（乱数の種 SEED で再現可）
  ・weight 列: 元の比率に戻すための重み（wet = 1、dry = 「元の dry 件数 ÷ 採った dry 件数」）。
    平均や割合を出すときに weight で重み付けすると、間引く前の母集団の値になる
■ 分割 split: 年で決める（SPLIT_BY_YEAR）。2025 年はアーカイブの欠測が多いので "extra" として分けておく
"""
import argparse
import os
import numpy as np
import pandas as pd

STATIONS = ["DSM", "IAH", "SEA", "ATL", "BOS"]
YEARS = list(range(2019, 2026))
SPLIT_BY_YEAR = {2019: "train", 2020: "train", 2021: "train", 2022: "train", 2023: "dev", 2024: "test", 2025: "extra"}
STRIDE_MIN = 5          # 判定時刻 t0 の刻み [分]
PAST_MIN_OK = 48        # 過去 60 分のうち T/RH/P が揃う分数の下限
FUT_MIN_OK = 54         # 先 60 分のうち R がある分数の下限
DRY_RATIO = 3           # dry（雨なし）を wet（雨あり）の何倍まで採るか
SEED = 20261014
RISK3_WINDOWS = (5, 10, 15, 30)   # risk3_peak で使う区間の幅 [分]
RISK3_INTENSITY = 0.2             # risk3_peak で 100 点とする降水強度 [mm/分]（0.2 mm/分 = 12 mm/h。5 分なら 1.0 mm）
PAST = np.arange(-60, 0)   # 過去 60 分のオフセット [分]
FUT = np.arange(1, 61)     # 先 60 分のオフセット [分]


def load_station(raw_dir, st):
    """1 地点の全年を読み、1 分格子（UTC）に並べて SI 単位に換算する。列: T, RH, P, R"""
    frames = []
    for y in YEARS:
        p = os.path.join(raw_dir, f"iem_{st}_{y}.csv")
        if not os.path.exists(p):
            print(f"  missing {p}")
            continue
        # ptype 列に区切り文字が混ざる行がごく稀にある（IAH 2025 で 35 行）。その行は警告を出して飛ばす
        df = pd.read_csv(p, usecols=["valid(UTC)", "tmpf", "dwpf", "pres1", "precip"], na_values=["M"], on_bad_lines="warn")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.index = pd.to_datetime(df.pop("valid(UTC)"), utc=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated()].sort_index()
    grid = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1] + 1}-01-01", freq="1min", tz="UTC", inclusive="left")
    df = df.reindex(grid)
    T = (df["tmpf"] - 32) / 1.8                       # °F → °C
    Td = (df["dwpf"] - 32) / 1.8
    RH = 100 * np.exp(17.625 * Td / (243.04 + Td)) / np.exp(17.625 * T / (243.04 + T))   # Magnus 式。露点 → 相対湿度
    return pd.DataFrame({"T": T, "RH": RH.clip(upper=100), "P": df["pres1"] * 33.8639, "R": df["precip"] * 25.4}, index=grid)


def choose_t0(m, rng):
    """有効な判定時刻の位置（1 分格子上のインデックス）を層化して選ぶ。返り値: (選んだ位置, 各行の weight, 統計)"""
    ok3 = (m["T"].notna() & m["RH"].notna() & m["P"].notna()).astype(int)
    past_ok = ok3.rolling(60, min_periods=1).sum().shift(1)                       # [t0-60, t0) の揃い分数
    okR = m["R"].notna().astype(int)
    fut_ok = okR[::-1].rolling(60, min_periods=1).sum()[::-1].shift(-1)           # (t0, t0+60] の揃い分数
    R60 = m["R"].fillna(0)[::-1].rolling(60, min_periods=1).sum()[::-1].shift(-1)
    valid = (past_ok >= PAST_MIN_OK) & (fut_ok >= FUT_MIN_OK) & (m.index.minute % STRIDE_MIN == 0)
    idx = np.flatnonzero(valid.values)
    idx = idx[(idx >= 60) & (idx < len(m) - 60)]                                   # 端で窓がはみ出す位置は除く
    wet = idx[R60.values[idx] > 0]
    dry = idx[R60.values[idx] == 0]
    n_dry = min(len(dry), DRY_RATIO * len(wet))
    dry_keep = rng.choice(dry, n_dry, replace=False)
    sel = np.sort(np.concatenate([wet, dry_keep]))
    w_dry = len(dry) / n_dry if n_dry else 1.0
    weight = np.where(R60.values[sel] > 0, 1.0, w_dry)
    stats = dict(n_valid_t0=len(idx), n_wet_all=len(wet), n_dry_all=len(dry), n_dry_kept=n_dry, weight_dry=round(w_dry, 3))
    return sel, weight, stats


def to_int100(x):
    """四捨五入して 0〜100 の整数にする。先に小数 6 桁に丸めて、浮動小数の誤差で 63.5 が 63.4999… になる等のぶれを防ぐ"""
    return np.clip(np.floor(np.round(x, 6) + 0.5), 0, 100).astype(int)


def risk3_peak(Rf0):
    """Rf0: 先 60 分の降水 (件数, 60) [mm/分]（欠測は 0）→ 0〜100 の実数（丸めは呼び出し側）"""
    cs = np.concatenate([np.zeros((len(Rf0), 1)), np.cumsum(Rf0, axis=1)], axis=1)   # 累積和（先頭に 0）
    scores = []
    for w in RISK3_WINDOWS:
        max_w = (cs[:, w:] - cs[:, :-w]).max(axis=1)                                 # 幅 w 分の区間雨量の最大 [mm]
        scores.append(np.minimum(max_w / (RISK3_INTENSITY * w) * 100, 100))
    return np.mean(scores, axis=0)


def build_rows(m, st, sel, weight):
    """選んだ t0 ごとに ±60 分を切り出して横長の DataFrame にする。"""
    win = lambda col, offs: m[col].values[sel[:, None] + offs[None, :]]           # (件数, 60)
    T, RH, P, Rp = win("T", PAST), win("RH", PAST), win("P", PAST), win("R", PAST)
    Rf = win("R", FUT)
    Rf0 = np.nan_to_num(Rf)                                                        # 危険度の計算では欠測を 0 mm 扱い
    R60 = Rf0.sum(axis=1)
    risk1 = np.where(R60 >= 0.5, 100, 0)
    risk2 = to_int100(R60 * 10)
    risk3 = to_int100(risk3_peak(Rf0))
    t0 = m.index[sel]
    meta = pd.DataFrame({
        "sample_id": [f"{st}_{t.strftime('%Y%m%d%H%M')}" for t in t0],
        "station": st,
        "t0_utc": t0.strftime("%Y-%m-%dT%H:%MZ"),
        "split": [SPLIT_BY_YEAR.get(y, "other") for y in t0.year],
        "weight": np.round(weight, 3),
        "n_past_ok": (~np.isnan(T) & ~np.isnan(RH) & ~np.isnan(P)).sum(axis=1),
        "n_fut_ok": (~np.isnan(Rf)).sum(axis=1),
        "R60_mm": np.round(R60, 3),
        "risk1_bin": risk1,
        "risk2_lin": risk2,
        "risk3_peak": risk3,
    })
    cols = {}
    cols.update({f"T_{o:d}": v for o, v in zip(PAST, np.round(T, 2).T)})
    cols.update({f"RH_{o:d}": v for o, v in zip(PAST, np.round(RH, 1).T)})
    cols.update({f"P_{o:d}": v for o, v in zip(PAST, np.round(P, 2).T)})
    cols.update({f"R_{o:d}": v for o, v in zip(PAST, np.round(Rp, 3).T)})
    cols.update({f"R_+{o:d}": v for o, v in zip(FUT, np.round(Rf, 3).T)})
    return pd.concat([meta, pd.DataFrame(cols)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="eval/data/raw/asos_1min")
    ap.add_argument("--out", default="eval/data/dataset_asos")
    ap.add_argument("--stations", nargs="+", default=STATIONS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary = []
    for st in a.stations:
        print(f"[{st}] loading", flush=True)
        m = load_station(a.raw, st)
        sel, weight, stats = choose_t0(m, rng)
        df = build_rows(m, st, sel, weight)
        path = os.path.join(a.out, f"dataset_{st}.csv")
        df.to_csv(path, index=False, na_rep="")
        print(f"[{st}] {len(df):,} rows, {df.shape[1]} cols -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)  {stats}", flush=True)
        for split, g in df.groupby("split"):
            summary.append(dict(station=st, split=split, rows=len(g), wet=int((g["R60_mm"] > 0).sum()), dry=int((g["R60_mm"] == 0).sum()),
                                risk1_ge0_5=int((g["risk1_bin"] == 100).sum()),
                                event_rate_weighted_pct=round(100 * np.average(g["risk1_bin"] == 100, weights=g["weight"]), 2),
                                risk2_mean=round(float(g["risk2_lin"].mean()), 2), risk3_mean=round(float(g["risk3_peak"].mean()), 2), **stats))
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(a.out, "dataset_summary.csv"), index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
