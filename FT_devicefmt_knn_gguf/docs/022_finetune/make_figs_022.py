#!/usr/bin/env python3
"""022（017 revB）用の模式図・学習曲線を描く。
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/w -w /w gemma4-eval python3 -I report/261008_auduino検証/022_finetune/make_figs_022.py
出力: report/261008_auduino検証/022_finetune/figs/figR1..R5.png、out/label_ab_table.csv
"""
import json, os, struct, collections, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = f"{HERE}/figs"; OUT = f"{HERE}/out"
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
BASE = "#4C78A8"; GREY = "#94a3b8"; ORANGE = "#E76F51"; GREEN = "#2A9D8F"; INK = "#0f172a"; LIGHT = "#e2e8f0"
p = "eval/data/raw/fonts/NotoSansJP-Regular.ttf"
if os.path.exists(p):
    fm.fontManager.addfont(p); plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
plt.rcParams["axes.unicode_minus"] = False

# ---------- figR1: LoRA の模式図（1 つの重み行列） ----------
fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.7); ax.axis("off")
# 入力 x
ax.text(0.45, 2.6, "入力 x\n(1,536 個の数)", ha="center", va="center", fontsize=9, color=INK)
ax.add_patch(Rectangle((0.3, 1.4), 0.3, 2.4, fc=LIGHT, ec=INK, lw=0.8))
# 元の重み W（凍結・4bit）
ax.add_patch(Rectangle((1.6, 0.6), 3.0, 4.0, fc="#cbd5e1", ec=INK, lw=1.0))
ax.text(3.1, 2.6, "元の重み W\n6,144 × 1,536\n= 944 万個の数\n\n凍結（変えない）\n4 ビットに圧縮して保持", ha="center", va="center", fontsize=10, color=INK)
# +
ax.text(5.05, 2.6, "＋", ha="center", va="center", fontsize=22, color=INK)
# B (6144 x 16)
ax.add_patch(Rectangle((5.6, 0.6), 0.45, 4.0, fc=ORANGE, ec=INK, lw=1.0, alpha=0.9))
ax.text(5.83, 4.85, "B", ha="center", va="center", fontsize=12, color=INK, fontweight="bold")
ax.text(5.83, 0.35, "6,144 × 16", ha="center", va="center", fontsize=8.5, color=INK)
ax.text(6.35, 2.6, "×", ha="center", va="center", fontsize=16, color=INK)
# A (16 x 1536)
ax.add_patch(Rectangle((6.7, 2.4), 3.0, 0.4, fc=ORANGE, ec=INK, lw=1.0, alpha=0.9))
ax.text(8.2, 3.1, "A   16 × 1,536", ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
ax.text(8.2, 1.75, "A と B だけを学習する\n合わせて 12.3 万個の数（W の 1.3%）\n初期値: A は乱数、B は 0\n（学習前は B×A = 0 で元のモデルと同じ）", ha="center", va="center", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff7ed", ec=ORANGE, lw=0.8))
# 出力
ax.text(5.5, 5.4, "出力 = W·x ＋ (α/r)·B·A·x      （r = 16, α = 16 なので係数は 1）", ha="center", va="center", fontsize=11, color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc="#eef3f9", ec=BASE, lw=1.0))
ax.text(5.5, 0.08, "例は MLP の gate_proj（入力 1,536 → 出力 6,144）。同じ仕組みを 35 層 × 7 箇所（q/k/v/o + gate/up/down）に付け、学習する数は合計 2,416 万個（ベース 51 億個の 0.5%）。", ha="center", va="bottom", fontsize=8.5, color="#475569")
fig.tight_layout(); fig.savefig(f"{FIG}/figR1_lora.png", dpi=120); plt.close(fig)

# ---------- figR2: 学習データ 1 件の構造（損失のかかる範囲） ----------
rows = [json.loads(l) for i, l in zip(range(3), open("eval/data/sft/sft_B.jsonl", encoding="utf-8"))]
r = rows[1]
fig, ax = plt.subplots(figsize=(11, 5.0)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
import textwrap
pl = [w for ln in r["prompt"].rstrip("\n").split("\n") for w in textwrap.wrap(ln, 44) or [""]]
ax.add_patch(Rectangle((0.3, 1.5), 6.2, 2.85, fc="#f1f5f9", ec=GREY, lw=1.0))
ax.text(0.45, 4.2, "文面（prompt）: 229 トークン → 読むだけ。損失に入れない（ラベル = −100）", fontsize=9.5, color="#475569", va="top", fontweight="bold")
ax.text(0.45, 3.85, "\n".join(pl), fontsize=7.2, color=INK, va="top", family=plt.rcParams["font.family"])
ax.add_patch(Rectangle((6.8, 1.5), 3.9, 2.85, fc="#eef3f9", ec=BASE, lw=1.4))
ax.text(6.95, 4.2, "回答（completion）: 31 トークン → ここだけ当てさせる", fontsize=9.5, color=BASE, va="top", fontweight="bold")
comp = r["completion"].split("\n")
ax.text(6.95, 3.7, f"「{comp[0]}」", fontsize=16, color=INK, va="top", fontweight="bold")
ax.text(6.95, 2.95, "↑ 正解 B（似た入力 200 件の risk2 平均）\n   この行の実測 risk2 は " + str(r["risk2"]), fontsize=8.5, color="#475569", va="top")
ax.text(6.95, 2.3, "\n".join(["「" + comp[1][:22], "  " + comp[1][22:] + "」", "＋ <eos>（終わりの印）"]), fontsize=8.5, color=INK, va="top")
ax.text(5.5, 1.1, "学習は「文面を読んだ状態で、回答の次の 1 トークンを当てる」を回答の 31 トークンぶん繰り返し、外れた分（交差エントロピー）を小さくするように A・B を更新する。", ha="center", fontsize=9, color=INK, wrap=True)
ax.text(5.5, 0.6, "チャット用の飾り（<start_of_turn> など）は付けない＝実機で llama_ros に渡す生文面と同じ形で学習する。", ha="center", fontsize=9, color=INK)
ax.text(5.5, 0.15, f"実例: sft_B.jsonl の 2 件目（{r['sample_id']}）。1 件あたりの長さは 260 トークン前後（文面 229 + 回答 31）。", ha="center", fontsize=8, color="#475569")
fig.tight_layout(); fig.savefig(f"{FIG}/figR2_sample.png", dpi=120); plt.close(fig)

# ---------- figR3: 学習曲線 ----------
fig, ax = plt.subplots(figsize=(9, 4.2))
cols = {"A": GREY, "B": BASE, "C": ORANGE}; names = {"A": "FT-A（risk2 そのまま・P2'）", "B": "FT-B（期待値ラベル・P2'）", "C": "FT-C（risk2 そのまま・P0）"}
curve = {}
for run in "ABC":
    d = json.load(open(f"models/ft/{run}/train_log.json"))
    tr = [(x["step"], x["loss"]) for x in d["log_history"] if "loss" in x]; ev = [(x["step"], x["eval_loss"]) for x in d["log_history"] if "eval_loss" in x]
    curve[run] = {"train": tr, "eval": ev}
    ax.plot([s for s, _ in tr], [l for _, l in tr], "-", color=cols[run], lw=2, label=f"{names[run]}  検証損失 {ev[-1][1]:.3f}")
    ax.plot([s for s, _ in ev], [l for _, l in ev], "o", color=cols[run], ms=8, mec="white", mew=1.5)
ax.set_yscale("log"); ax.set_xlabel("学習ステップ（1 ステップ = 16 件。982 ステップ = 15,700 件 × 1 周）"); ax.set_ylabel("損失（回答トークンの交差エントロピー、対数軸）")
ax.set_title("学習曲線: 線 = 学習データの損失（50 ステップごと）、● = 検証 300 件の損失（500 / 982 ステップ）", fontsize=10)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8.5, loc="upper right")
ax.axvline(100, color=GREY, ls=":", lw=1); ax.text(105, 0.6, "100 ステップで形式（整数＋定型文）を覚える", fontsize=8, color="#475569")
fig.tight_layout(); fig.savefig(f"{FIG}/figR3_loss.png", dpi=120); plt.close(fig)
json.dump(curve, open(f"{OUT}/loss_curves.json", "w"), indent=1)

# ---------- figR4: 量子化の模式図 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]; ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
rng = np.random.default_rng(0); w = rng.normal(0, 0.02, 6)
ax.text(6, 4.75, "重み 1 個の表し方（ブロック 32 個のうち 6 個を例示）", ha="center", fontsize=10, color=INK, fontweight="bold")
ax.text(0.1, 3.95, "F16（16 ビット）\n各値をそのまま", fontsize=8.5, color=INK, va="center")
for i, v in enumerate(w):
    ax.add_patch(Rectangle((2.8 + i * 1.05, 3.6), 1.0, 0.7, fc="#cbd5e1", ec=INK, lw=0.6)); ax.text(3.3 + i * 1.05, 3.95, f"{v:+.4f}", ha="center", va="center", fontsize=7.5, color=INK)
ax.text(9.4, 3.95, "16 bit × 32 個\n= 512 bit", fontsize=8, color="#475569", va="center", ha="left")
ax.annotate("", xy=(3.3, 2.75), xytext=(3.3, 3.45), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
ax.text(3.6, 3.1, "ブロックごとに「目盛りの幅 d」と「最小値 m」を決め、\n各値を 0〜15 の整数 q に丸める（値 ≈ d·q + m）", fontsize=8, color=INK, va="center")
ax.text(0.1, 2.2, "Q4_K_M（4 ビット）\n整数 0〜15 ＋ 共通の目盛り", fontsize=8.5, color=INK, va="center")
d = (w.max() - w.min()) / 15; m = w.min(); q = np.round((w - m) / d).astype(int)
for i, v in enumerate(q):
    ax.add_patch(Rectangle((2.8 + i * 1.05, 1.85), 1.0, 0.7, fc="#eef3f9", ec=BASE, lw=0.8)); ax.text(3.3 + i * 1.05, 2.2, f"{v}", ha="center", va="center", fontsize=10, color=BASE, fontweight="bold")
ax.text(9.4, 2.2, "4 bit × 32 個 ＋ d, m\n≈ 4.5 bit/個（約 1/3.5）", fontsize=8, color="#475569", va="center", ha="left")
ax.text(0.1, 1.1, f"復元すると: {', '.join(f'{m + d * v:+.4f}' for v in q[:4])} …（元の値との差が「量子化の誤差」）", fontsize=8, color=INK, va="center")
ax.text(0.1, 0.55, "imatrix 有り: 学習文面 2,000 件を流して「よく使う重み」を測り、そこの誤差が小さくなるように d・m を選ぶ。", fontsize=8, color=INK, va="center")
ax.text(0.1, 0.1, "本書の数字: 誤差の指標 KL（F16 との出力分布のずれ）は imatrix 無し 0.185 → 有り 0.114。", fontsize=8, color=INK, va="center")
ax = axes[1]
labels = ["ベース HF\n（BF16）", "LoRA\nアダプタ", "マージ後\nF16 GGUF", "Q4_K_M\n(imatrix)"]; sizes = [10.2, 0.097, 9.31, 3.43]
bars = ax.bar(labels, sizes, color=[GREY, ORANGE, GREY, BASE], width=0.6)
for b, s in zip(bars, sizes):
    ax.text(b.get_x() + b.get_width() / 2, s + 0.15, f"{s:.2f} GB" if s < 1 else f"{s:.1f} GB", ha="center", fontsize=9, color=INK)
ax.set_ylabel("ファイルの大きさ（GB）"); ax.set_title("各段階のファイルの大きさ", fontsize=10); ax.set_ylim(0, 11); ax.grid(axis="y", alpha=.3); ax.tick_params(axis="x", labelsize=8.5)
ax.text(1.5, 10.3, "実機（空き約 2.9 GB + mmap）に載るのは右端だけ", ha="center", fontsize=8, color="#475569")
fig.tight_layout(); fig.savefig(f"{FIG}/figR4_quant.png", dpi=120); plt.close(fig)

# ---------- figR5: 正解 A（実測）と正解 B（期待値）の関係 ----------
S = pd.read_json("eval/data/sft/sft_B.jsonl", lines=True)[["label", "risk2"]]
bins = [-1, 0, 10, 30, 100]; names_bin = ["0", "1〜10", "11〜30", "31〜100"]
S["abin"] = pd.cut(S.risk2, bins=bins, labels=names_bin)
S["bbin"] = pd.cut(S.label, bins=[-1, 0, 5, 10, 15, 20, 30, 100], labels=["0", "1〜5", "6〜10", "11〜15", "16〜20", "21〜30", "31〜"])
ct = pd.crosstab(S.bbin, S.abin, normalize="index") * 100; n = S.bbin.value_counts().reindex(ct.index)
ct.to_csv(f"{OUT}/label_ab_table.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]; bottom = np.zeros(len(ct)); shades = ["#dbe4f0", "#9fb9d6", "#4C78A8", "#1e3a5f"]
for col, c in zip(ct.columns, shades):
    ax.bar(ct.index.astype(str), ct[col], bottom=bottom, color=c, label=f"実測 risk2 = {col}", width=0.7, edgecolor="white", linewidth=1); bottom += ct[col].to_numpy()
for i, (idx, cnt) in enumerate(n.items()):
    ax.text(i, 101, f"{cnt:,} 件", ha="center", fontsize=7.5, color="#475569")
ax.set_xlabel("正解 B（似た入力 200 件の risk2 平均）"); ax.set_ylabel("その B の値を持つ学習データの内訳（%）"); ax.set_ylim(0, 108)
ax.set_title("同じ B の値でも、実測 risk2（正解 A）はばらついている", fontsize=10); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False); ax.grid(axis="y", alpha=.3)
ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.text(5, 9.4, "ある入力（湿度 85%・気圧 −0.9 hPa …）に対して", ha="center", fontsize=9.5, color=INK, fontweight="bold")
ax.add_patch(Rectangle((0.4, 5.2), 4.2, 3.6, fc="#f1f5f9", ec=GREY, lw=1)); ax.text(2.5, 8.4, "正解 A: 実測そのまま", ha="center", fontsize=9, color=INK, fontweight="bold")
ax.text(2.5, 6.9, "似た入力の正解が\n0, 0, 100, 0, 46, 0, 3 …\nとばらつく\n→ 温度 0 の生成は最頻値（0 や 3）に寄る", ha="center", va="center", fontsize=8, color=INK)
ax.add_patch(Rectangle((5.4, 5.2), 4.2, 3.6, fc="#eef3f9", ec=BASE, lw=1.4)); ax.text(7.5, 8.4, "正解 B: 期待値", ha="center", fontsize=9, color=BASE, fontweight="bold")
ax.text(7.5, 6.9, "似た入力 200 件の平均\n= 12（1 つに定まる）\n→ 入力が変われば値も\n連続的に変わる", ha="center", va="center", fontsize=8, color=INK)
ax.text(5, 4.3, "結果（017 §5-B）", ha="center", fontsize=9.5, color=INK, fontweight="bold")
ax.text(5, 2.6, "A・C: 出力が 0 と 3 の 2 値に崩れた（AUC 0.79 / 0.73）\nB: 出力 0〜30 の 23 種類（AUC 0.87）", ha="center", va="center", fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=0.4", fc="#fff7ed", ec=ORANGE, lw=0.8))
ax.text(5, 0.6, "B と A の順位相関は 0.55（sft_summary.json）。B は A を「平均でならした」値。", ha="center", fontsize=8, color="#475569")
fig.tight_layout(); fig.savefig(f"{FIG}/figR5_label.png", dpi=120); plt.close(fig)
print(ct.round(1)); print(n.to_dict()); print("rows", len(S), "corr", round(S.label.corr(S.risk2, method="spearman"), 3)); print("done")
