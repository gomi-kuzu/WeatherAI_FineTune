#!/usr/bin/env python3
"""baseline 評価: 実機文面を llama-server に流し、応答・1 行目の整数・トークン数・所要を記録する（提案書 012 §3-2）。
  python run_baseline.py --url http://quant:8080 --tag BF16 --mode raw  --temp 0   --seed 0
  python run_baseline.py --url http://quant:8080 --tag BF16 --mode chat --temp 0   --seed 0
  python run_baseline.py --url http://quant:8080 --tag BF16 --mode raw  --temp 0.2 --seed 1 --limit 500
出力: results/baseline/<tag>_<mode>_t<temp>_s<seed>.jsonl（1 行 1 サンプル。再開可能）
  - mode raw : /completion に文面をそのまま渡す（チャットテンプレート無し。実機の llama_ros と同じ渡し方）
  - mode chat: /v1/chat/completions に user 発言として渡す（テンプレート付き、思考モード OFF）
  - 1 行目の読み取りは weather_risk_setter と同じ（改行までを int()、0〜100 にクランプ）。読めない場合は risk=None。
    参考として「行頭の数字だけを拾う」寛容な読み方（risk_lenient）も記録する。
  - --parse fmt4（追加パターン ①②）: 【降雨確率の目安】の % を risk に、【降雨危険度】の 4 段階を risk_level に記録する（範囲表記は平均）。
  - --parse fmtidx（追加パターン ①'、021）: 【降雨指数】の整数（0〜100）を risk に、【降水量予測】の mm を risk_mm に記録する。
    指数が読めず mm だけ読めた行は、mm × 10（100 上限）を risk_lenient に入れる。
"""
from __future__ import annotations
import argparse, json, pathlib, re, time
import requests
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_first_line(text: str):
    head, _, _ = (text or "").partition("\n"); head = head.strip()
    try:
        v = int(head); return max(0, min(100, v))
    except ValueError:
        return None


def parse_lenient(text: str):
    m = re.search(r"-?\d+", (text or "").partition("\n")[0])
    return max(0, min(100, int(m.group()))) if m else None


LEVELS = ("危険", "警戒", "注意", "安全")


def parse_fmt4(text: str):
    """書式付き回答（追加パターン ①②）の読み取り。返り値: (risk_pct, level)。
    - risk_pct: 「降雨確率の目安」の後ろにある % 値（キー語が複数回出る場合は数値を含む最初の行）。「60〜70%」のような範囲は平均。
      キー語の行に無ければ本文中で最初に現れる % 値。
    - level: 「降雨危険度」の後ろにある 4 段階語（前置き文「…降雨危険度を評価します」のようにキー語の後ろに段階語が無い行は飛ばす）。
      フォーマットの復唱 [危険 / 警戒 / 注意 / 安全]（3 語以上）は無効。どの行にも無ければ、キー語「降雨危険度」を除いた本文で最初に現れる段階語。"""
    t = (text or "").replace("％", "%").replace("：", ":").replace("~", "〜").replace("～", "〜")
    pct = None
    segs = [m.group(1) for m in re.finditer(r"降雨確率の目安[】\]]*\s*:?\s*([^\n]*)", t)] or [t]
    for seg in segs:
        rng = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*[〜\-–]\s*(\d+(?:\.\d+)?)\s*%", seg)
        if rng:
            nums = [float(rng.group(1)), float(rng.group(2))]
        else:
            nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", seg)][:1]
            if not nums and seg is not t:   # 「約70」のように % 無し
                nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", seg)][:1]
        if nums:
            pct = max(0, min(100, int(round(sum(nums) / len(nums))))); break
    level = None
    for seg2 in [m.group(1) for m in re.finditer(r"降雨危険度[】\]]*\s*:?\s*([^\n]*)", t)]:
        hits = sorted((seg2.find(l), l) for l in LEVELS if l in seg2)
        if 1 <= len(hits) <= 2:
            level = hits[0][1]; break
    if level is None:
        body = re.sub(r"\[\s*危険\s*/\s*警戒\s*/\s*注意\s*/\s*安全\s*\]", "", t.replace("降雨危険度", ""))
        hits = sorted((body.find(l), l) for l in LEVELS if l in body)
        level = hits[0][1] if hits else None
    return pct, level


def parse_fmtidx(text: str):
    """書式付き回答（追加パターン ①'）の読み取り。返り値: (idx, mm, idx_from_mm)。
    - idx: 「降雨指数」の後ろにある最初の整数（0〜100 にクランプ。「約 25」「**25**」も可。範囲「20〜30」は平均）。
    - mm: 「降水量予測」の後ろにある最初の数値（範囲は平均。「0.5mm未満」は 0.5 と読む）。
    - idx_from_mm: mm × 10 を四捨五入し 100 で頭打ちした値（指数が無い行の代替、および指数と mm の整合の確認用）。"""
    t = (text or "").replace("：", ":").replace("~", "〜").replace("～", "〜").replace("，", ",")
    t = re.sub(r"\[\s*0\s*〜\s*100\s*の整数\s*\]|\[\s*◯\.◯\s*mm\s*\]", "", t)   # フォーマットの復唱は無効
    def first_num(segs):
        for seg in segs:
            seg = seg.replace("*", "")
            rng = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*[〜\-–]\s*(\d+(?:\.\d+)?)", seg)
            if rng: return (float(rng.group(1)) + float(rng.group(2))) / 2
            m = re.search(r"(\d+(?:\.\d+)?)", seg)
            if m: return float(m.group(1))
        return None
    idx = first_num([m.group(1) for m in re.finditer(r"降雨指数[】\]]*\s*:?\s*([^\n]*)", t)])
    mm = first_num([m.group(1) for m in re.finditer(r"降水量予測[】\]]*\s*:?\s*([^\n]*)", t)])
    idx = None if idx is None else max(0, min(100, int(round(idx))))
    idx_from_mm = None if mm is None else max(0, min(100, int(round(mm * 10 + 1e-9))))
    return idx, mm, idx_from_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["raw", "chat"], required=True)
    ap.add_argument("--temp", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompts", default=str(ROOT / "data" / "baseline" / "dev_prompts.jsonl"))
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--label", default="", help="文面の種類（p1/p2/p2h/p3）。出力ファイル名を <tag>-<label>_... にする")
    ap.add_argument("--concurrency", type=int, default=4); ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--parse", choices=["first_line", "fmt4", "fmtidx"], default="first_line", help="回答の読み取り方（p* は first_line、追加パターン q1/q2 は fmt4、q3 は fmtidx）")
    ap.add_argument("--seed-mode", choices=["fixed", "per-row"], default="per-row",
                    help="fixed: 全リクエストに同じ seed（乱数列が毎回同じになり、温度 > 0 でも行ごとの独立な揺れにならない）/ per-row: seed*100000 + 行番号")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.prompts, encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[: a.limit]
    name = f"{a.tag}-{a.label}" if a.label else a.tag
    out = ROOT / "results" / "baseline" / f"{name}_{a.mode}_t{a.temp:g}_s{a.seed}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["sample_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [it for it in items if it["sample_id"] not in done]
    print(f"{out.name}: {len(todo)} to run ({len(done)} done)", flush=True)

    row_no = {it["sample_id"]: i for i, it in enumerate(items)}

    def one(it):
        t0 = time.time(); rec = {"raw": None, "error": None, "usage": None, "timings": None, "stop": None}
        seed = a.seed if a.seed_mode == "fixed" else a.seed * 100000 + row_no[it["sample_id"]]
        try:
            if a.mode == "raw":
                body = {"prompt": it["prompt"], "n_predict": a.max_tokens, "temperature": a.temp, "seed": seed, "cache_prompt": True}
                r = requests.post(f"{a.url}/completion", json=body, timeout=a.timeout); r.raise_for_status(); j = r.json()
                rec["raw"] = j.get("content")
                rec["usage"] = {"prompt_tokens": j.get("tokens_evaluated"), "completion_tokens": j.get("tokens_predicted")}
                rec["timings"] = j.get("timings"); rec["stop"] = j.get("stop_type") or ("limit" if j.get("stopped_limit") else "eos")
            else:
                body = {"model": a.tag, "messages": [{"role": "user", "content": it["prompt"]}], "temperature": a.temp, "seed": seed,
                        "max_tokens": a.max_tokens, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
                r = requests.post(f"{a.url}/v1/chat/completions", json=body, timeout=a.timeout); r.raise_for_status(); j = r.json()
                ch = j["choices"][0]; rec["raw"] = ch["message"]["content"]
                rec["usage"] = j.get("usage"); rec["timings"] = j.get("timings"); rec["stop"] = ch.get("finish_reason")
        except Exception as e:
            rec["error"] = str(e)[:300]
        rec["wall_s"] = round(time.time() - t0, 3); rec["seed_used"] = seed
        if a.parse == "fmt4":
            pct, level = parse_fmt4(rec["raw"]) if rec["raw"] else (None, None)
            rec["risk"] = pct; rec["risk_lenient"] = pct; rec["risk_level"] = level
        elif a.parse == "fmtidx":
            idx, mm, idx_mm = parse_fmtidx(rec["raw"]) if rec["raw"] else (None, None, None)
            rec["risk"] = idx; rec["risk_lenient"] = idx if idx is not None else idx_mm; rec["risk_mm"] = mm; rec["risk_from_mm"] = idx_mm
        else:
            rec["risk"] = parse_first_line(rec["raw"]) if rec["raw"] else None
            rec["risk_lenient"] = parse_lenient(rec["raw"]) if rec["raw"] else None
        return {k: it[k] for k in it if k != "prompt"} | rec

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for i, rec in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} wall={rec['wall_s']}s risk={rec['risk']} stop={rec['stop']}", flush=True)


if __name__ == "__main__":
    main()
