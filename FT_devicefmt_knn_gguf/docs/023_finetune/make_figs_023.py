#!/usr/bin/env python3
"""023（017 revC）用の図: モデルの全体像と各部位の役割、1 層の中身と LoRA の位置、35 層 × 7 箇所の LoRA 地図。
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/w -w /w gemma4-eval python3 -I report/261008_auduino検証/023_finetune/make_figs_023.py
入力: models/hf/gemma-4-E2B-it/{config.json,model.safetensors}（ヘッダだけ読む）、models/ft/B/lora/adapter_model.safetensors（ヘッダだけ読む）
出力: figs/figR6_model_overview.png、figs/figR8_layer_lora.png、figs/figR9_lora_map.png、out/{parts.csv,lora_by_module.csv,lora_by_layer.csv}
"""
import json, os, re, struct, math, collections
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = f"{HERE}/figs"; OUT = f"{HERE}/out"
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
BASE = "#4C78A8"; GREY = "#94a3b8"; ORANGE = "#E76F51"; GREEN = "#2A9D8F"; INK = "#0f172a"; LIGHT = "#e2e8f0"; FROZEN = "#cbd5e1"; DATA = "#eef3f9"
p = "eval/data/raw/fonts/NotoSansJP-Regular.ttf"
if os.path.exists(p):
    fm.fontManager.addfont(p); plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
plt.rcParams["axes.unicode_minus"] = False


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    h.pop("__metadata__", None); return h


# ---------- 現物から数える ----------
cfg = json.load(open("models/hf/gemma-4-E2B-it/config.json")); tc = cfg["text_config"]
L = tc["num_hidden_layers"]; H = tc["hidden_size"]; V = tc["vocab_size"]; KVS = tc["num_kv_shared_layers"]
layer_types = tc["layer_types"]
base = st_header("models/hf/gemma-4-E2B-it/model.safetensors")
parts = collections.Counter(); wshape = {}
for k, v in base.items():
    c = math.prod(v["shape"]) if v["shape"] else 1
    if "vision" in k: g = "画像の入口（vision tower）"
    elif "audio" in k: g = "音声の入口（audio tower）"
    elif "embed_tokens_per_layer" in k: g = "層ごとの埋め込み表（PLE）"
    elif "embed_tokens" in k: g = "トークン埋め込み表（出力層と共有）"
    elif re.search(r"layers\.\d+\.self_attn\.(q|k|v|o)_proj", k): g = "35 層: 注意機構 q/k/v/o"
    elif re.search(r"layers\.\d+\.mlp\.(gate|up|down)_proj", k): g = "35 層: MLP gate/up/down"
    elif "per_layer" in k: g = "PLE の射影"
    else: g = "正規化など"
    parts[g] += c
    m = re.search(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)\.weight", k)
    if m: wshape[(int(m.group(1)), m.group(3))] = tuple(v["shape"])
total = sum(parts.values())
ad = st_header("models/ft/B/lora/adapter_model.safetensors")
lora_layer = collections.defaultdict(dict); lora_mod = collections.Counter(); lora_total = 0
for k, v in ad.items():
    m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_(A|B)", k)
    l, mod, ab = int(m.group(1)), m.group(2), m.group(3); c = math.prod(v["shape"])
    lora_layer[l][(mod, ab)] = tuple(v["shape"]); lora_mod[mod] += c; lora_total += c
has = {(l, mod): (mod, "A") in lora_layer[l] for l in range(L) for mod in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
per_layer = {l: sum(math.prod(s) for s in d.values()) for l, d in lora_layer.items()}
# LoRA を付けた W の合計（付いているものだけ）
w_target = sum(math.prod(wshape[(l, mod)]) for (l, mod), on in has.items() if on)
pd.DataFrame([(g, c, c / total) for g, c in parts.items()], columns=["part", "params", "share"]).to_csv(f"{OUT}/parts.csv", index=False)
pd.DataFrame([(m, c, c / lora_total) for m, c in lora_mod.items()], columns=["module", "lora_params", "share"]).to_csv(f"{OUT}/lora_by_module.csv", index=False)
pd.DataFrame([(l, layer_types[l], l >= L - KVS, per_layer[l], wshape[(l, "gate_proj")][0]) for l in range(L)],
             columns=["layer", "attention", "kv_shared", "lora_params", "mlp_width"]).to_csv(f"{OUT}/lora_by_layer.csv", index=False)
print({g: f"{c:,}" for g, c in parts.items()}, "total", f"{total:,}")
print({m: f"{c:,}" for m, c in lora_mod.items()}, "lora total", f"{lora_total:,}", "share of model", f"{lora_total/total:.2%}", "share of targeted W", f"{lora_total/w_target:.2%}", "W targeted", f"{w_target:,}")
print("per-layer lora params", sorted(set(per_layer.values())))
oku = lambda c: f"{c/1e8:.1f} 億"
man = lambda c: f"{c/1e4:,.0f} 万"


def box(ax, x, y, w, h, text, fc=FROZEN, ec=INK, lw=1.0, fs=9, color=INK, ls="-", bold=False, rounded=True):
    if rounded:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw, ls=ls))
    else:
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.2, style="-|>", cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12, color=color, lw=lw, connectionstyle=cs))


def lora_tag(ax, x, y, text="＋B·A", fs=7.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=ORANGE, ec=ORANGE, lw=0.5))


# ---------- figR6: モデルの全体像（入力 → 出力）と LoRA の位置 ----------
fig, ax = plt.subplots(figsize=(12.5, 7.6)); ax.set_xlim(0, 12.5); ax.set_ylim(0, 7.6); ax.axis("off")
Y = 4.6; HB = 1.5
box(ax, 0.2, Y, 1.45, HB, "入力の文面\n（P2'、231 トークン）\n\n『ATL 2023-04-20 …\n湿度 79% …』", fc=DATA, ec=BASE, fs=8)
box(ax, 1.95, Y, 1.35, HB, "① トークナイザ\n\n文字列を 26 万種の\n番号の列に区切る\n（重み無し）", fc="#f8fafc", ec=GREY, fs=8)
box(ax, 3.6, Y, 1.55, HB, f"② トークン埋め込み表\n{V:,} × {H:,}\n= {oku(parts['トークン埋め込み表（出力層と共有）'])}個\n\n番号 → 1,536 個の数\n凍結", fc=FROZEN, fs=8)
# 35 layers block
ax.add_patch(FancyBboxPatch((5.45, Y - 0.35), 3.3, HB + 0.7, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#fff7ed", ec=ORANGE, lw=1.6))
ax.text(7.1, Y + HB + 0.15, f"③ 35 層の積み重ね（{oku(parts['35 層: 注意機構 q/k/v/o'] + parts['35 層: MLP gate/up/down'])}個）  ← LoRA はここ", ha="center", va="center", fontsize=9, color=INK, fontweight="bold")
for i in range(5):
    x0 = 5.6 + i * 0.62
    box(ax, x0, Y + 0.12, 0.52, 1.2, "", fc=FROZEN, rounded=False)
    ax.text(x0 + 0.26, Y + 1.0, "注意", ha="center", va="center", fontsize=7, color=INK)
    ax.text(x0 + 0.26, Y + 0.42, "MLP", ha="center", va="center", fontsize=7, color=INK)
    ax.add_patch(Rectangle((x0 + 0.04, Y + 0.78), 0.44, 0.07, fc=ORANGE, ec="none"))
    ax.add_patch(Rectangle((x0 + 0.04, Y + 0.2), 0.44, 0.07, fc=ORANGE, ec="none"))
ax.text(8.6, Y + 0.75, "…", ha="center", va="center", fontsize=14, color=INK)
ax.text(7.1, Y - 0.2, f"各層 = 注意機構（q/k/v/o）＋ MLP（gate/up/down）。橙の線 = 7 箇所の行列に横付けした LoRA（A・B）", ha="center", va="center", fontsize=7.5, color=INK)
box(ax, 9.0, Y, 1.25, HB, "④ 正規化 →\n出力層\n（②の表を再利用）\n\n26 万種それぞれの\n『次に来る確率』", fc=FROZEN, fs=8)
box(ax, 10.5, Y, 1.8, HB, "⑤ 1 トークン選ぶ\n温度 0 = 一番高い確率\n\n『1』『4』『\\n』\n『湿度』『が』…『<eos>』", fc=DATA, ec=BASE, fs=8)
for x0, x1 in [(1.65, 1.95), (3.3, 3.6), (5.15, 5.45), (8.75, 9.0), (10.25, 10.5)]:
    arrow(ax, x0, Y + HB / 2, x1, Y + HB / 2)
# loop arrow
arrow(ax, 11.4, Y, 11.4, Y - 0.6, color=BASE, style="-")
arrow(ax, 11.4, Y - 0.6, 1.0, Y - 0.6, color=BASE, style="-")
arrow(ax, 1.0, Y - 0.6, 1.0, Y - 0.05, color=BASE)
ax.text(6.2, Y - 0.78, "選んだトークンを文面の末尾に足して ①〜⑤ を繰り返す（1 周 = 出力 1 トークン。平均 24 周で <eos> が出て止まる）。学習時は⑤の代わりに「正解トークンの確率」から損失を計算する", ha="center", va="center", fontsize=8, color=BASE)
# PLE
box(ax, 5.45, Y + HB + 0.55, 3.3, 0.75, f"層ごとの埋め込み表（PLE）{V:,} × 35 × 256 = {oku(parts['層ごとの埋め込み表（PLE）'])}個 ＋ 射影 {man(parts['PLE の射影'])}個\n各層に「このトークン専用の数」を 256 個ずつ渡す。凍結・LoRA 無し", fc=FROZEN, fs=7.5)
arrow(ax, 7.1, Y + HB + 0.55, 7.1, Y + HB + 0.4, color=GREY)
# vision/audio
box(ax, 0.2, Y + HB + 0.55, 3.1, 0.75, f"画像の入口 {oku(parts['画像の入口（vision tower）'])}個・音声の入口 {oku(parts['音声の入口（audio tower）'])}個\n本用途では使わない（GGUF 変換で落ちる）。凍結・LoRA 無し", fc="#f8fafc", ec=GREY, ls="--", fs=7.5, color="#475569")
# bottom: parameter bar
by = 1.9; bx0 = 0.6; bw = 11.3
order = ["層ごとの埋め込み表（PLE）", "35 層: MLP gate/up/down", "トークン埋め込み表（出力層と共有）", "音声の入口（audio tower）", "35 層: 注意機構 q/k/v/o", "画像の入口（vision tower）", "PLE の射影", "正規化など"]
cols = {"層ごとの埋め込み表（PLE）": "#94a3b8", "35 層: MLP gate/up/down": "#fdba74", "トークン埋め込み表（出力層と共有）": "#cbd5e1", "音声の入口（audio tower）": "#e2e8f0",
        "35 層: 注意機構 q/k/v/o": "#fb923c", "画像の入口（vision tower)": "#e2e8f0", "画像の入口（vision tower）": "#e2e8f0", "PLE の射影": "#cbd5e1", "正規化など": "#cbd5e1"}
x = bx0
ax.text(bx0, by + 0.95, f"重みの内訳（全 {oku(total)}個 = {total/1e9:.2f} B）。橙系 = LoRA を横付けした部位（35 層の注意機構と MLP）、灰 = 一切変えない部位", fontsize=9, color=INK, fontweight="bold")
for g in order:
    w = bw * parts[g] / total
    ax.add_patch(Rectangle((x, by), w, 0.55, fc=cols[g], ec="white", lw=0.8))
    if w > 0.8:
        ax.text(x + w / 2, by + 0.275, f"{g}\n{oku(parts[g])}", ha="center", va="center", fontsize=7.5, color=INK)
    x += w
def seg_center(g):
    return bx0 + bw * (sum(parts[gg] for gg in order[:order.index(g)]) + parts[g] / 2) / total
ax.text(seg_center("音声の入口（audio tower）"), by - 0.2, "音声 3.1 億", ha="center", va="top", fontsize=7, color="#475569")
ax.text(seg_center("35 層: 注意機構 q/k/v/o"), by - 0.45, f"35 層: 注意機構 {oku(parts['35 層: 注意機構 q/k/v/o'])}（LoRA あり）", ha="center", va="top", fontsize=7, color=INK)
ax.text(seg_center("画像の入口（vision tower）") + 0.45, by - 0.2, "画像 1.7 億・PLE 射影 0.4 億", ha="center", va="top", fontsize=7, color="#475569")
# LoRA bar
ly = 0.75
ax.text(bx0, ly + 0.62, f"学習した数（LoRA の A・B）: {man(lora_total)}個 = 全体の {lora_total/total:.2%}。LoRA を付けた行列（{oku(w_target)}個）に対しては {lora_total/w_target:.1%}", fontsize=9, color=INK, fontweight="bold")
lw_ = bw * lora_total / total
ax.add_patch(Rectangle((bx0, ly), bw, 0.3, fc="#f1f5f9", ec=GREY, lw=0.6))
ax.add_patch(Rectangle((bx0, ly), max(lw_, 0.06), 0.3, fc=ORANGE, ec="none"))
ax.text(bx0 + max(lw_, 0.06) + 0.1, ly + 0.15, f"← この幅（全体の {lora_total/total:.2%}）が学習で動かした数。残り {1-lora_total/total:.1%} は凍結", va="center", fontsize=8, color=INK)
ax.text(6.25, 0.12, "数は config.json と model.safetensors / adapter_model.safetensors のヘッダから数えた実物。文面・出力の例は §5-F の実物（DSM 2023-01-11 05:45 → 『14 / 湿度が 79% で、気圧が 60 分で 2.1 hPa 下がっているため』）。", ha="center", va="bottom", fontsize=7.5, color="#475569")
fig.tight_layout(); fig.savefig(f"{FIG}/figR6_model_overview.png", dpi=120); plt.close(fig)

# ---------- figR7: 1 層の中身と LoRA の位置 ----------
fig, ax = plt.subplots(figsize=(12.5, 7.8)); ax.set_xlim(0, 12.5); ax.set_ylim(0, 7.8); ax.axis("off")
s0 = {m: wshape[(0, m)] for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
l0 = lora_layer[0]
def dims(m):  # (out, in)
    o, i = s0[m]; return f"{i:,} → {o:,}"
def lcount(m):
    return math.prod(l0[(m, "A")]) + math.prod(l0[(m, "B")])
# main flow y
ym = 4.9
ax.text(0.25, ym + 2.65, "1 層の中身（第 0 層の実寸。入力・出力とも 1,536 個の数 × トークン数）", fontsize=10.5, color=INK, fontweight="bold")
box(ax, 0.2, ym - 0.35, 0.7, 0.7, "x\n前の層\nから", fc=DATA, ec=BASE, fs=7.5)
# attention block
ax.add_patch(FancyBboxPatch((1.2, ym - 1.75), 5.1, 3.35, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#f8fafc", ec=GREY, lw=1.0))
ax.text(3.75, ym + 1.4, "注意機構（attention）: 文面の中の「どのトークンを見るか」を決めて情報を集める", ha="center", fontsize=8.5, color=INK, fontweight="bold")
box(ax, 1.35, ym - 0.3, 0.75, 0.6, "正規化", fc="#f1f5f9", ec=GREY, fs=7.5)
arrow(ax, 0.9, ym, 1.35, ym)
for i, (m, lab) in enumerate([("q_proj", "q: 何を探すか"), ("k_proj", "k: 各トークンの見出し"), ("v_proj", "v: 各トークンの中身")]):
    yy = ym + 0.75 - i * 0.75
    box(ax, 2.4, yy - 0.26, 1.55, 0.52, f"{lab}\n{dims(m)}", fc=FROZEN, fs=7)
    arrow(ax, 2.1, ym, 2.4, yy, cs="arc3,rad=0")
    lora_tag(ax, 3.95, yy + 0.26, f"＋B·A {lcount(m)/1e4:.1f} 万")
box(ax, 4.3, ym - 0.5, 1.0, 1.0, "注意の計算\n（重み無し）\n8 ヘッド", fc="#f1f5f9", ec=GREY, fs=7)
for i in range(3):
    arrow(ax, 3.95, ym + 0.75 - i * 0.75, 4.3, ym + 0.3 - i * 0.3)
box(ax, 5.45, ym - 0.3, 0.75, 0.6, f"o\n{dims('o_proj')}", fc=FROZEN, fs=7)
arrow(ax, 5.3, ym, 5.45, ym)
lora_tag(ax, 6.2, ym + 0.3, f"＋B·A {lcount('o_proj')/1e4:.1f} 万")
# residual add
ax.add_patch(plt.Circle((6.75, ym), 0.17, fc="white", ec=INK, lw=1.0)); ax.text(6.75, ym, "＋", ha="center", va="center", fontsize=11, color=INK)
arrow(ax, 6.2, ym, 6.58, ym)
arrow(ax, 0.55, ym + 0.35, 0.55, ym + 2.2, style="-", color=GREY); arrow(ax, 0.55, ym + 2.2, 6.75, ym + 2.2, style="-", color=GREY); arrow(ax, 6.75, ym + 2.2, 6.75, ym + 0.17, color=GREY)
ax.text(3.5, ym + 2.3, "残差（入力をそのまま足す。層を重ねても情報が消えない仕組み）", ha="center", va="bottom", fontsize=7, color="#475569")
# MLP block
ax.add_patch(FancyBboxPatch((7.2, ym - 1.75), 4.1, 3.35, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#f8fafc", ec=GREY, lw=1.0))
ax.text(9.25, ym + 1.4, "MLP: 集めた情報を変換する（「知識」の大部分）", ha="center", fontsize=8.5, color=INK, fontweight="bold")
box(ax, 7.35, ym - 0.3, 0.75, 0.6, "正規化", fc="#f1f5f9", ec=GREY, fs=7.5)
arrow(ax, 6.92, ym, 7.35, ym)
box(ax, 8.35, ym + 0.2, 1.25, 0.52, f"gate\n{dims('gate_proj')}", fc=FROZEN, fs=7)
box(ax, 8.35, ym - 0.72, 1.25, 0.52, f"up\n{dims('up_proj')}", fc=FROZEN, fs=7)
arrow(ax, 8.1, ym, 8.35, ym + 0.46); arrow(ax, 8.1, ym, 8.35, ym - 0.46)
lora_tag(ax, 9.6, ym + 0.72, f"＋B·A {lcount('gate_proj')/1e4:.1f} 万"); lora_tag(ax, 9.6, ym - 0.2, f"＋B·A {lcount('up_proj')/1e4:.1f} 万")
ax.add_patch(plt.Circle((9.95, ym), 0.17, fc="white", ec=INK, lw=1.0)); ax.text(9.95, ym, "×", ha="center", va="center", fontsize=11, color=INK)
arrow(ax, 9.6, ym + 0.46, 9.83, ym + 0.1); arrow(ax, 9.6, ym - 0.46, 9.83, ym - 0.1)
ax.text(9.95, ym - 0.4, "GELU", ha="center", va="top", fontsize=6.5, color="#475569")
box(ax, 10.3, ym - 0.3, 0.9, 0.6, f"down\n{dims('down_proj')}", fc=FROZEN, fs=7)
arrow(ax, 10.12, ym, 10.3, ym)
lora_tag(ax, 11.2, ym + 0.3, f"＋B·A {lcount('down_proj')/1e4:.1f} 万")
ax.add_patch(plt.Circle((11.65, ym), 0.17, fc="white", ec=INK, lw=1.0)); ax.text(11.65, ym, "＋", ha="center", va="center", fontsize=11, color=INK)
arrow(ax, 11.2, ym, 11.48, ym)
arrow(ax, 6.75, ym - 0.17, 6.75, ym - 1.95, style="-", color=GREY); arrow(ax, 6.75, ym - 1.95, 11.65, ym - 1.95, style="-", color=GREY); arrow(ax, 11.65, ym - 1.95, 11.65, ym - 0.17, color=GREY)
ax.text(9.2, ym - 2.05, "残差", ha="center", va="top", fontsize=7, color="#475569")
box(ax, 11.85, ym - 0.35, 0.6, 0.7, "次の\n層へ", fc=DATA, ec=BASE, fs=7.5)
arrow(ax, 11.82, ym, 11.85, ym)
# PLE input
box(ax, 7.4, ym - 1.6, 2.7, 0.5, "＋ 層ごとの埋め込み（PLE、256 個 → 1,536 個）凍結", fc=FROZEN, fs=6.5)
arrow(ax, 10.1, ym - 1.35, 11.55, ym - 0.3, color=GREY, cs="arc3,rad=-0.25")
# legend / notes
ax.add_patch(Rectangle((0.3, 0.2), 0.4, 0.3, fc=FROZEN, ec=INK, lw=0.8)); ax.text(0.8, 0.35, "灰の箱 = 元の重み行列（凍結・4 ビットで保持）。「1,536 → 2,048」は入力の数 → 出力の数", va="center", fontsize=8, color=INK)
lora_tag(ax, 0.5, 0.75, "＋B·A"); ax.text(0.8, 0.75, "= その行列に LoRA（B·A）を横付けして出力に足す。数字は学習する数（A と B の合計）", va="center", fontsize=8, color=INK)
ax.add_patch(Rectangle((0.3, 1.0), 0.4, 0.3, fc="#f1f5f9", ec=GREY, lw=0.8)); ax.text(0.8, 1.15, "薄い箱 = 重みが無い、または正規化（1,536 個のスケール値のみ）。LoRA は付けない", va="center", fontsize=8, color=INK)
nk = sum(1 for l in range(L) if (l, "k_proj") in [(ll, m) for (ll, m), on in has.items() if on])
ax.text(6.4, 1.95, (f"この層（第 0 層）の LoRA: 7 箇所・{per_layer[0]/1e4:.1f} 万個。層によって行列の幅が違う: "
                    f"第 4・9・14・19・24・29・34 層（全体注意）は q/k/v/o が 2 倍幅、"
                    f"第 15〜34 層は MLP が 2 倍幅（12,288）で、k・v を第 0〜14 層から借りる（KV 共有）ため k・v の行列は使われず LoRA も付かない。"),
        ha="center", va="center", fontsize=7.8, color=INK, wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff7ed", ec=ORANGE, lw=0.8))
fig.tight_layout(); fig.savefig(f"{FIG}/figR8_layer_lora.png", dpi=120); plt.close(fig)

# ---------- figR8: 35 層 × 7 箇所の LoRA 地図 + 部位別の学習数 ----------
mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
labels = {"q_proj": "注意 q", "k_proj": "注意 k", "v_proj": "注意 v", "o_proj": "注意 o", "gate_proj": "MLP gate", "up_proj": "MLP up", "down_proj": "MLP down"}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [2.4, 1]})
ax = axes[0]
for l in range(L):
    for j, m in enumerate(mods):
        on = has[(l, m)]
        n = (math.prod(lora_layer[l][(m, "A")]) + math.prod(lora_layer[l][(m, "B")])) if on else 0
        big = on and n > 1e5
        ax.add_patch(Rectangle((l, 6 - j), 1, 1, fc=(ORANGE if on else "#e2e8f0"), ec="white", lw=1.0, alpha=(1.0 if big else 0.6) if on else 1))
        if not on: ax.text(l + 0.5, 6 - j + 0.5, "—", ha="center", va="center", fontsize=6, color="#64748b")
ax.set_xlim(0, L); ax.set_ylim(0, 7); ax.set_yticks([i + 0.5 for i in range(7)]); ax.set_yticklabels([labels[m] for m in mods][::-1], fontsize=8.5)
ax.set_xticks([i + 0.5 for i in range(0, L, 1)]); ax.set_xticklabels([str(i) for i in range(L)], fontsize=6.5)
ax.set_xlabel("層の番号（0 = 入力に近い側 … 34 = 出力に近い側）", fontsize=8.5)
ax.annotate("", xy=(L - KVS, 7.2), xytext=(L, 7.2), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
ax.text(L - KVS / 2, 7.32, f"第 {L-KVS}〜{L-1} 層: k・v を前の層から借りる（KV 共有）→ k・v に LoRA 無し。MLP は 2 倍幅", ha="center", va="bottom", fontsize=7.5, color=INK)
for l in range(L):
    if layer_types[l] == "full_attention":
        ax.annotate("", xy=(l + 0.5, 7.85), xytext=(l + 0.5, 8.3), arrowprops=dict(arrowstyle="-|>", color=BASE, lw=1.0))
ax.text(L / 2, 8.4, "▼ = 全体注意の層（第 4・9・…・34 層。他は 512 トークン幅の局所注意）", ha="center", va="bottom", fontsize=7.5, color=BASE)
ax.set_ylim(0, 8.9); ax.spines[:].set_visible(False); ax.tick_params(length=0)
ax.set_title(f"LoRA を付けた場所（橙）: 35 層 × 7 箇所のうち {sum(has.values())} 箇所、行列 {len(ad)} 個。濃い橙 = 1 箇所 10 万個以上（MLP 側）", fontsize=9)
ax = axes[1]
vals = [lora_mod[m] for m in mods]
ax.barh(range(7), vals[::-1], color=[ORANGE if m.startswith(("gate", "up", "down")) else "#fb923c" for m in mods[::-1]])
ax.set_yticks(range(7)); ax.set_yticklabels([labels[m] for m in mods][::-1], fontsize=8.5)
for i, v in enumerate(vals[::-1]):
    ax.text(v + lora_total * 0.005, i, f"{v/1e4:,.0f} 万（{v/lora_total:.0%}）", va="center", fontsize=7.5, color=INK)
ax.set_xlim(0, max(vals) * 1.45); ax.set_xlabel("学習する数（A と B の合計、万個）", fontsize=8.5)
ax.set_xticks([0, 2e6, 4e6, 6e6, 8e6]); ax.set_xticklabels(["0", "200 万", "400 万", "600 万", "800 万"], fontsize=8)
ax.set_title(f"部位別の学習数: 合計 {lora_total/1e4:,.0f} 万個\nMLP 側が {sum(lora_mod[m] for m in mods[4:])/lora_total:.0%}、注意機構側が {sum(lora_mod[m] for m in mods[:4])/lora_total:.0%}", fontsize=9)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIG}/figR9_lora_map.png", dpi=120); plt.close(fig)
print("done")
