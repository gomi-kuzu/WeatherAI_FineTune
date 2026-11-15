#!/usr/bin/env python3
"""baseline 評価の採点（提案書 012 §3-4・§3-5）: 順位（AUC・Spearman）/ 値（MAE・信頼性表）/ 形式（整数率・出力長）
+ 参照モデル（常に 0・地点平均・ロジスティック回帰・勾配ブースティング）+ ブートストラップ CI + 層別 + 誤り現物 + 温度 0.2 のばらつき + 図。
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/w -w /w gemma4-eval python3 -I eval/scripts/score_baseline.py
入力: eval/data/baseline/{dev_sample.csv, dev_prompts.jsonl, train_feats.csv}, eval/results/baseline/*.jsonl
出力: eval/results/baseline/{summary.csv, summary.json, slices.csv, reliability.csv, error_samples.csv, noise.json, predictions_*.csv}
      eval/results/figs/*.png
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE = "#4C78A8"; GREY = "#94a3b8"; ORANGE = "#E76F51"; GREEN = "#2A9D8F"
FEATS8 = ["P_now", "dP60", "dP30", "RH_now", "dRH60", "dRH30", "T_now", "dT60"]
BUCKETS = [f"{v}_b{k:02d}" for v in ("T", "RH", "P") for k in range(12)]
BIN_ORDER = ["b0", "b1", "b2", "b3", "b4"]
BIN_LABEL = {"b0": "0", "b1": "1-10", "b2": "11-30", "b3": "31-60", "b4": "61-100"}
SEASON = {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春", 6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}


def setup_font():
    p = "eval/data/raw/fonts/NotoSansJP-Regular.ttf"
    if os.path.exists(p):
        fm.fontManager.addfont(p); plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()


def wmean(x, w):
    return float(np.average(x, weights=w))


def metrics(y2, y1, pred, w):
    """y2 = risk2, y1 = event(0/1), pred = 出力(0-100), w = 母集団重み。NaN の pred は除外済みであること。"""
    m = {}
    m["auc_event_w"] = float(roc_auc_score(y1, pred, sample_weight=w)) if len(set(y1)) > 1 else np.nan
    m["auc_event"] = float(roc_auc_score(y1, pred)) if len(set(y1)) > 1 else np.nan
    heavy = (y2 >= 30).astype(int)
    m["auc_heavy"] = float(roc_auc_score(heavy, pred)) if len(set(heavy)) > 1 else np.nan
    m["rho_all"] = float(spearmanr(pred, y2).correlation) if np.std(pred) > 0 else 0.0
    wet = y2 > 0
    m["rho_wet"] = float(spearmanr(pred[wet], y2[wet]).correlation) if wet.sum() > 2 and np.std(pred[wet]) > 0 else 0.0
    m["mae_w"] = wmean(np.abs(pred - y2), w); m["mae"] = float(np.mean(np.abs(pred - y2)))
    m["mae_wet"] = float(np.mean(np.abs(pred[wet] - y2[wet]))) if wet.any() else np.nan
    m["bias_w"] = wmean(pred - y2, w)
    alarm = pred >= 50
    tp = float(np.sum(w * (alarm & (y1 == 1)))); fp = float(np.sum(w * (alarm & (y1 == 0)))); fn = float(np.sum(w * (~alarm & (y1 == 1))))
    m["prec50_w"] = tp / (tp + fp) if tp + fp > 0 else np.nan; m["rec50_w"] = tp / (tp + fn) if tp + fn > 0 else np.nan
    m["f1_50_w"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else np.nan
    m["alarm_rate_w"] = wmean(alarm.astype(float), w)
    # dev 最良閾値（参考値。dev で選ぶので甘い）
    best = (np.nan, -1.0)
    for th in range(0, 101, 5):
        al = pred >= th; tp = float(np.sum(w * (al & (y1 == 1)))); fp = float(np.sum(w * (al & (y1 == 0)))); fn = float(np.sum(w * (~al & (y1 == 1))))
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
        if f1 > best[1]:
            best = (th, f1)
    m["best_th_dev"], m["best_f1_dev_w"] = best
    m["pred_mean"] = float(np.mean(pred)); m["pred_std"] = float(np.std(pred)); m["pred_nunique"] = int(len(np.unique(pred)))
    m["pred_mode_share"] = float(pd.Series(pred).value_counts(normalize=True).iloc[0])
    return m


def bootstrap(y2, y1, pred, w, n=1000, seed=0):
    rng = np.random.default_rng(seed); keys = ["auc_event_w", "rho_all", "rho_wet", "mae_w"]; acc = {k: [] for k in keys}
    N = len(y2)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(set(y1[idx])) < 2:
            continue
        mm = metrics(y2[idx], y1[idx], pred[idx], w[idx])
        for k in keys:
            acc[k].append(mm[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in acc.items()}


def reliability(y2, y1, pred, w):
    b = np.clip(pred // 10, 0, 9).astype(int); rows = []
    for k in range(10):
        s = b == k
        if s.sum() == 0:
            rows.append({"pred_bin": f"{k*10}-{k*10+9 if k < 9 else 100}", "n": 0}); continue
        rows.append({"pred_bin": f"{k*10}-{k*10+9 if k < 9 else 100}", "n": int(s.sum()), "pred_mean": float(pred[s].mean()),
                     "event_rate_w": wmean(y1[s], w[s]), "event_rate": float(y1[s].mean()), "risk2_mean_w": wmean(y2[s], w[s]),
                     "risk2_mean": float(y2[s].mean()), "risk2_median": float(np.median(y2[s]))})
    return rows


def per_true_bin(df, pred_col):
    rows = []
    for b in BIN_ORDER:
        s = df[df.strat_bin == b][pred_col].dropna()
        rows.append({"true_bin": BIN_LABEL[b], "n": int(len(s)), "mean": float(s.mean()), "median": float(s.median()),
                     "q25": float(s.quantile(.25)), "q75": float(s.quantile(.75)), "share_ge50": float((s >= 50).mean())})
    return rows


def load_results(resdir):
    out = {}
    for f in sorted(glob.glob(f"{resdir}/*.jsonl")):
        name = os.path.basename(f)[:-6]
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        d = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last").set_index("sample_id")
        d["prompt_tokens"] = d.usage.map(lambda u: u.get("prompt_tokens") if u else np.nan)
        d["completion_tokens"] = d.usage.map(lambda u: u.get("completion_tokens") if u else np.nan)
        d["prompt_ms"] = d.timings.map(lambda t: t.get("prompt_ms") if t else np.nan)
        d["predicted_ms"] = d.timings.map(lambda t: t.get("predicted_ms") if t else np.nan)
        out[name] = d
    return out


def reason_check(d, feats):
    """理由文の有無と、「気圧が下が/上が」の向きが dP60 の符号と合う割合。"""
    txt = d.raw.fillna(""); has = txt.str.contains("\n") & (txt.str.split("\n").str[1:].str.join("").str.strip() != "")
    claims = []
    for sid, t in txt.items():
        body = "".join(t.split("\n")[1:])
        if "気圧" in body and ("下が" in body or "低下" in body):
            claims.append((sid, -1))
        elif "気圧" in body and ("上が" in body or "上昇" in body):
            claims.append((sid, +1))
    ok = sum(1 for sid, sgn in claims if np.sign(feats.loc[sid, "dP60"]) == sgn)
    return {"reason_present_rate": float(has.mean()), "n_pressure_claims": len(claims),
            "pressure_direction_match_rate": (ok / len(claims)) if claims else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/data/baseline"); ap.add_argument("--results", default="eval/results/baseline")
    ap.add_argument("--figdir", default="eval/results/figs"); ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--cond-filter", default="", help="LLM 条件名の正規表現（図と表をこの条件に絞る。参照モデルは常に含む）")
    ap.add_argument("--summary-dir", default="", help="summary.csv 等の出力先（既定 --results と同じ）")
    ap.add_argument("--sample", default="dev", choices=["dev", "test"], help="採点に使う層化サンプル（test は最終確認用。016 で 1 回だけ）")
    a = ap.parse_args(); setup_font(); os.makedirs(a.figdir, exist_ok=True)
    if a.summary_dir:
        os.makedirs(a.summary_dir, exist_ok=True)
    S = pd.read_csv(f"{a.data}/{a.sample}_sample.csv").set_index("sample_id")
    S["event"] = (S.risk1_bin == 100).astype(int); S["season"] = S.month.map(SEASON)
    prompts = {json.loads(l)["sample_id"]: json.loads(l)["prompt"] for l in open(f"{a.data}/{a.sample}_prompts.jsonl", encoding="utf-8")}
    y2 = S.risk2_lin.to_numpy(float); y1 = S.event.to_numpy(int); w = S.w_pop.to_numpy(float)

    # ---- 参照モデル（train で学習、同じ 2,500 行で採点）----
    T = pd.read_csv(f"{a.data}/train_feats.csv"); T["event"] = (T.risk1_bin == 100).astype(int)
    preds = {}
    preds["ref: 常に 0"] = np.zeros(len(S))
    st_mean = T.groupby("station").apply(lambda g: np.average(g.risk2_lin, weights=g.weight))
    preds["ref: 地点平均"] = S.station.map(st_mean).to_numpy(float)
    sc = StandardScaler().fit(T[FEATS8]); lr = LogisticRegression(max_iter=2000).fit(sc.transform(T[FEATS8]), T.event, sample_weight=T.weight)
    preds["ref: ロジスティック回帰×100"] = 100 * lr.predict_proba(sc.transform(S[FEATS8]))[:, 1]
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0)
    hgb.fit(T[FEATS8 + BUCKETS], T.risk2_lin, sample_weight=T.weight)
    preds["ref: 勾配ブースティング回帰"] = np.clip(hgb.predict(S[FEATS8 + BUCKETS]), 0, 100)
    lr_coef = dict(zip(FEATS8, [round(float(c), 3) for c in lr.coef_[0]]))

    # ---- LLM の結果 ----
    R = load_results(a.results); table = []
    if a.cond_filter:
        R = {k: v for k, v in R.items() if re.search(a.cond_filter, k)}
    outdir = a.summary_dir or a.results; detail = {}; rel_rows = []; slice_rows = []; err_rows = []
    llm_preds = {}
    for name, d in R.items():
        d = d.reindex(S.index)
        n = int(d.raw.notna().sum()); parsed = d.risk.notna()
        row = {"condition": name, "n": int(len(d)), "n_done": n, "error_rate": float(d.error.notna().mean()),
               "int_rate": float(parsed.sum() / max(n, 1)), "lenient_rate": float(d.risk_lenient.notna().sum() / max(n, 1)),
               "cap_hit_rate": float((d.stop.isin(["limit", "length"])).sum() / max(n, 1)),
               "prompt_tokens_mean": float(d.prompt_tokens.mean()), "completion_tokens_mean": float(d.completion_tokens.mean()),
               "completion_tokens_p90": float(d.completion_tokens.quantile(.9)), "prompt_ms_mean": float(d.prompt_ms.mean()),
               "predicted_ms_mean": float(d.predicted_ms.mean()), "wall_s_mean": float(d.wall_s.mean())}
        if parsed.sum() >= 50:
            idx = parsed.to_numpy(); p = d.risk.to_numpy(float)[idx]
            row |= metrics(y2[idx], y1[idx], p, w[idx])
            if n >= 2000:   # 本番条件だけ CI・信頼性表・層別・誤り現物
                row["ci"] = bootstrap(y2[idx], y1[idx], p, w[idx], a.nboot)
                for r in reliability(y2[idx], y1[idx], p, w[idx]):
                    rel_rows.append({"condition": name} | r)
                S[f"pred::{name}"] = d.risk; llm_preds[name] = d.risk.to_numpy(float)
                detail[name] = {"per_true_bin": per_true_bin(S, f"pred::{name}"), "pred_value_counts": {int(k): int(v) for k, v in d.risk.value_counts().sort_index().items()},
                                "reason": reason_check(d[parsed], S)}
                for axis, col in [("station", S.station), ("season", S.season), ("true_bin", S.strat_bin),
                                  ("dP60_sign", np.where(S.dP60 < 0, "気圧低下", "気圧上昇/不変")), ("RH_now", np.where(S.RH_now >= 90, "湿度≥90", "湿度<90"))]:
                    col = pd.Series(np.asarray(col), index=S.index)
                    for v in sorted(col.unique()):
                        s = (col == v).to_numpy() & idx
                        if s.sum() < 30:
                            continue
                        mm = metrics(y2[s], y1[s], d.risk.to_numpy(float)[s], w[s])
                        slice_rows.append({"condition": name, "axis": axis, "slice": str(v), "n": int(s.sum()), "event_rate": float(y1[s].mean()),
                                           "auc_event": mm["auc_event"], "rho_all": mm["rho_all"], "mae": mm["mae"], "pred_mean": mm["pred_mean"]})
                # 誤りの現物: 真値 ≥ 60 で出力 ≤ 10 / 真値 0 で出力が最大
                dd = S.assign(pred=d.risk).dropna(subset=["pred"])
                miss = dd[(dd.risk2_lin >= 60)].nsmallest(5, "pred"); fa = dd[dd.risk2_lin == 0].nlargest(5, "pred")
                for kind, part in [("見逃し(真値≥60で出力が最小)", miss), ("空振り(真値0で出力が最大)", fa)]:
                    for sid, r in part.iterrows():
                        err_rows.append({"condition": name, "kind": kind, "sample_id": sid, "station": r.station, "t0_utc": r.t0_utc, "risk2": int(r.risk2_lin),
                                         "pred": int(r.pred), "dP60": round(r.dP60, 2), "RH_now": round(r.RH_now, 1), "raw": d.loc[sid, "raw"], "prompt": prompts[sid]})
        table.append(row)
    for name, p in preds.items():
        row = {"condition": name, "n": len(S), "n_done": len(S), "int_rate": 1.0} | metrics(y2, y1, p, w)
        row["ci"] = bootstrap(y2, y1, p, w, a.nboot)
        for r in reliability(y2, y1, p, w):
            rel_rows.append({"condition": name} | r)
        S[f"pred::{name}"] = p; detail[name] = {"per_true_bin": per_true_bin(S, f"pred::{name}")}
        table.append(row)
    summ = pd.DataFrame(table); ci = summ.pop("ci") if "ci" in summ else None
    summ.to_csv(f"{outdir}/summary.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(f"{outdir}/reliability.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(f"{outdir}/slices.csv", index=False)
    pd.DataFrame(err_rows).to_csv(f"{outdir}/error_samples.csv", index=False)
    S.filter(regex="^pred::|^risk|^station$|^strat_bin$|^w_pop$").to_csv(f"{outdir}/predictions_all.csv")

    # ---- 温度 0.2 × 3 回のばらつき（N1）----
    noise = {}
    seeds = {k: v for k, v in R.items() if "_t0.2_" in k}
    if len(seeds) >= 2:
        P = pd.DataFrame({k: v.risk for k, v in seeds.items()}).dropna()
        base = R.get("BF16_raw_t0_s0")
        noise = {"n_rows": int(len(P)), "row_std_mean": float(P.std(axis=1).mean()), "all_agree_rate": float((P.nunique(axis=1) == 1).mean()),
                 "range_mean": float((P.max(axis=1) - P.min(axis=1)).mean())}
        if base is not None:
            b = base.risk.reindex(P.index); noise["agree_with_temp0_rate"] = float((P.eq(b, axis=0)).mean().mean())
        per_seed = {}
        for k in P.columns:
            idx = P.index; mm = metrics(S.loc[idx, "risk2_lin"].to_numpy(float), S.loc[idx, "event"].to_numpy(int), P[k].to_numpy(float), S.loc[idx, "w_pop"].to_numpy(float))
            per_seed[k] = {kk: mm[kk] for kk in ("auc_event_w", "rho_all", "mae_w", "pred_mean")}
        noise["per_seed"] = per_seed
        noise["metric_sd_across_seeds"] = {kk: float(np.std([v[kk] for v in per_seed.values()])) for kk in ("auc_event_w", "rho_all", "mae_w")}
        if base is not None:
            idx = P.index; mm = metrics(S.loc[idx, "risk2_lin"].to_numpy(float), S.loc[idx, "event"].to_numpy(int), b.to_numpy(float), S.loc[idx, "w_pop"].to_numpy(float))
            noise["temp0_same_rows"] = {kk: mm[kk] for kk in ("auc_event_w", "rho_all", "mae_w", "pred_mean")}
    json.dump({"summary": table, "detail": detail, "noise": noise, "lr_coef": lr_coef, "hgb": {"n_train": int(len(T)), "max_iter": 300}},
              open(f"{outdir}/summary.json", "w"), indent=1, ensure_ascii=False, default=float)

    # ---- 図 ----
    main_conds = [c for c in llm_preds] + list(preds)
    # fig1: 入力データの現物確認（段 × 地点の件数、risk2 の分布: 抜いた 2,500 行 vs dev 全体）
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ct = pd.crosstab(S.strat_bin, S.station).reindex(BIN_ORDER); bottom = np.zeros(len(ct))
    for st, col in zip(ct.columns, [BASE, GREEN, ORANGE, GREY, "#8E6C8A"]):
        ax[0].bar([BIN_LABEL[b] for b in ct.index], ct[st], bottom=bottom, label=st, color=col); bottom += ct[st].to_numpy()
    ax[0].set_xlabel("risk2 の段（真値）"); ax[0].set_ylabel("行数"); ax[0].set_title("抜いた 2,500 行の段 × 地点"); ax[0].legend(fontsize=8)
    ax[1].hist(S.risk2_lin, bins=np.arange(0, 105, 5), color=BASE, alpha=.8, label="抜いた 2,500 行")
    ax[1].set_yscale("log"); ax[1].set_xlabel("risk2"); ax[1].set_ylabel("行数（対数）"); ax[1].set_title("risk2 の分布（層化で高い側を厚くしてある）"); ax[1].legend()
    for x in ax: x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{a.figdir}/fig1_sample.png", dpi=110); plt.close(fig)
    # fig3: 出力値の分布（LLM 条件と参照）
    nr = max(1, -(-len(main_conds) // 4)); fig, axes = plt.subplots(nr, 4, figsize=(16, 3.2 * nr)); axes = np.atleast_1d(axes).ravel()
    for ax_ in axes[len(main_conds):]: ax_.axis("off")
    for i, c in enumerate(main_conds):
        p = S[f"pred::{c}"].dropna(); axes[i].hist(p, bins=np.arange(0, 105, 5), color=BASE if c in llm_preds else GREY)
        axes[i].set_title(c, fontsize=9); axes[i].set_xlabel("出力"); axes[i].grid(alpha=.3)
    fig.suptitle("出力値の分布（2,500 行）"); fig.tight_layout(); fig.savefig(f"{a.figdir}/fig3_pred_dist.png", dpi=110); plt.close(fig)
    # fig4: 真値の段 × 出力の箱ひげ
    fig, axes = plt.subplots(1, len(main_conds), figsize=(2.6 * len(main_conds), 4), sharey=True)
    for axx, c in zip(np.atleast_1d(axes), main_conds):
        data = [S[S.strat_bin == b][f"pred::{c}"].dropna() for b in BIN_ORDER]
        axx.boxplot(data, tick_labels=[BIN_LABEL[b] for b in BIN_ORDER], showfliers=False, medianprops={"color": ORANGE})
        axx.set_title(c, fontsize=8); axx.set_xlabel("真値の段"); axx.grid(alpha=.3); axx.tick_params(axis="x", labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("出力（0〜100）"); fig.suptitle("真値の段ごとの出力の分布（箱 = 四分位、線 = 中央値）")
    fig.tight_layout(); fig.savefig(f"{a.figdir}/fig4_pred_by_truebin.png", dpi=110); plt.close(fig)
    # fig5: 信頼性（出力 10 刻み → 実際の事象率・risk2 平均）
    rel = pd.DataFrame(rel_rows); fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for c, col in zip(main_conds, [BASE, GREEN, ORANGE, "#8E6C8A", GREY, "#555", "#999", "#bbb"]):
        r = rel[(rel.condition == c) & (rel.n > 0)]
        if r.empty: continue
        ax[0].plot(r.pred_mean, r.event_rate_w, "o-", color=col, label=c, ms=4); ax[1].plot(r.pred_mean, r.risk2_mean_w, "o-", color=col, label=c, ms=4)
    ax[0].set_xlabel("出力（10 刻みの平均）"); ax[0].set_ylabel("実際に 0.5 mm 以上降った割合（母集団重み）"); ax[0].set_title("信頼性: 出力 → 事象率")
    ax[1].set_xlabel("出力（10 刻みの平均）"); ax[1].set_ylabel("risk2 の平均（母集団重み）"); ax[1].set_title("信頼性: 出力 → risk2 平均"); ax[1].plot([0, 100], [0, 100], "--", color=GREY, lw=1)
    for x in ax: x.grid(alpha=.3); x.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{a.figdir}/fig5_reliability.png", dpi=110); plt.close(fig)
    # fig6: 主指標の比較（AUC・ρ・MAE、CI 付き）
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2)); names = [r["condition"] for r in table if "auc_event_w" in r]
    for k, axx, title in [("auc_event_w", ax[0], "AUC（event、母集団重み）↑"), ("rho_all", ax[1], "Spearman ρ（risk2、全体）↑"), ("mae_w", ax[2], "MAE（母集団重み）↓")]:
        vals = [r[k] for r in table if "auc_event_w" in r]; lo = [r[k] - r["ci"][k][0] if "ci" in r else 0 for r in table if "auc_event_w" in r]
        hi = [r["ci"][k][1] - r[k] if "ci" in r else 0 for r in table if "auc_event_w" in r]
        cols = [BASE if n in llm_preds else GREY for n in names]
        axx.barh(names, vals, xerr=[lo, hi], color=cols, capsize=3); axx.set_title(title); axx.grid(alpha=.3); axx.tick_params(axis="y", labelsize=7); axx.invert_yaxis()
    ax[0].axvline(0.5, color=ORANGE, ls="--", lw=1); fig.tight_layout(); fig.savefig(f"{a.figdir}/fig6_metrics.png", dpi=110); plt.close(fig)
    # fig7: 層別（地点・季節）
    sl = pd.DataFrame(slice_rows)
    if not sl.empty:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for axx, axis in zip(ax, ["station", "season"]):
            sub = sl[sl.axis == axis]; conds = sub.condition.unique(); xs = sorted(sub.slice.unique()); wdt = 0.8 / max(len(conds), 1)
            for i, c in enumerate(conds):
                v = sub[sub.condition == c].set_index("slice").reindex(xs)
                axx.bar(np.arange(len(xs)) + i * wdt, v.auc_event, wdt, label=c)
            axx.set_xticks(np.arange(len(xs)) + 0.4 - wdt / 2); axx.set_xticklabels(xs); axx.axhline(0.5, color=ORANGE, ls="--", lw=1)
            axx.set_ylabel("AUC（event）"); axx.set_title(f"層別: {axis}"); axx.grid(alpha=.3); axx.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"{a.figdir}/fig7_slices.png", dpi=110); plt.close(fig)
    # fig8: 温度 0.2 の揺れ
    if seeds:
        P = pd.DataFrame({k: v.risk for k, v in seeds.items()}).dropna(); fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(P.max(axis=1) - P.min(axis=1), bins=np.arange(0, 60, 5), color=BASE); ax[0].set_xlabel("同じ行の 3 回の出力の幅（最大 − 最小）"); ax[0].set_ylabel("行数"); ax[0].set_title("温度 0.2 × 3 回の揺れ")
        b = R["BF16_raw_t0_s0"].risk.reindex(P.index) if "BF16_raw_t0_s0" in R else None
        if b is not None:
            ax[1].scatter(b + np.random.default_rng(0).normal(0, .6, len(b)), P.mean(axis=1), s=6, alpha=.4, color=BASE); ax[1].plot([0, 100], [0, 100], "--", color=GREY)
            ax[1].set_xlabel("温度 0 の出力"); ax[1].set_ylabel("温度 0.2 の 3 回平均"); ax[1].set_title("温度 0 と 0.2 の対応")
        for x in ax: x.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{a.figdir}/fig8_noise.png", dpi=110); plt.close(fig)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(summ[[c for c in ["condition", "n_done", "int_rate", "completion_tokens_mean", "auc_event_w", "auc_event", "auc_heavy", "rho_all", "rho_wet", "mae_w", "mae_wet", "f1_50_w", "pred_mean", "pred_nunique", "pred_mode_share"] if c in summ]].to_string(index=False))
    print(json.dumps(noise, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
