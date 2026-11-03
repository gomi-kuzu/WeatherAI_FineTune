#!/usr/bin/env python3
"""ASOS 横長 CSV を、急な降雨を重視した日本語チャット学習 JSONL に変換する。

標準ライブラリのみ。既定では各地点 train=600/dev=150/test=150/extra=60 件を上限に採る。
出力は weather_risk（未来60分の降水量mm×10を整数化、上限100）と reason（観測根拠の要約）。
"""
import argparse
import csv
import hashlib
import heapq
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFSETS = tuple(range(-60, 0, 5))
SPLITS = {2019: "train", 2020: "train", 2021: "train", 2022: "train",
          2023: "dev", 2024: "test", 2025: "extra"}
LIMITS = {"train": 600, "dev": 150, "test": 150, "extra": 60}
BUCKETS = {"sudden_onset": 0.25, "rapid_intensification": 0.20,
           "rain_onset": 0.15, "ongoing_rain": 0.10, "light_rain": 0.05,
           "sensor_alert_no_rain": 0.15, "dry": 0.10}
# 気圧変動（5 分変化の標準偏差）の帯。学習年の四分位（約 0.053 / 0.084 / 0.143 hPa）を基に設定。
# 変動が大きいほど荒天率・平均リスクが単調に上がることを dev/test で確認済み（荒天 AUC 0.75）。
VOLATILITY_BANDS = (0.06, 0.14)   # [hPa] 未満: 静穏 / 以上 VOLATILITY_BANDS[1] 未満: やや変動 / 以上: 変動大
SYSTEM = (
    "山中オフライン用の気象予測。過去1時間・5分間隔のT(気温℃)、RH(相対湿度%)、"
    "P(現地気圧hPa)、dP(5分気圧差)から今後1時間の荒天リスクを出す。各配列は-60..-5分の順。"
    "出力はJSONで weather_risk(整数0-100) と reason(短い理由) の2項目のみ。"
    "weather_riskは今後1時間の合計降水量mm×10で100上限、降水量の指標で降水確率ではない。"
    "判断は気圧の細かな上下動(dPのばらつき)を最重視し、変動が大きいほど高リスク、"
    "平坦なら低リスク。湿度と気温は補助で気圧変動より優先しない。"
    "気圧が正味で上下した向きだけでは判断しない。雷・突風・雨雪は断定しない。"
    "reasonは観測の変化と意味を簡潔に。未来値を既知として述べず、注意書きやレーダー確認案内は不要。"
)


def number(row, key):
    try:
        value = float(row[key])
    except (ValueError, KeyError, TypeError):
        return None
    return value if math.isfinite(value) else None


def max_window(values, width):
    total = sum(values[:width])
    maximum = total
    for i in range(width, len(values)):
        total += values[i] - values[i - width]
        maximum = max(maximum, total)
    return maximum


def pressure_volatility(pressures):
    """5 分ごとの気圧変化の標準偏差 [hPa]。正味の増減ではなく「揺れの大きさ」を測る。"""
    diffs = [b - a for a, b in zip(pressures, pressures[1:])]
    mean = sum(diffs) / len(diffs)
    return math.sqrt(sum((d - mean) ** 2 for d in diffs) / len(diffs))


def weather_risk(r60_mm):
    """未来60分の降水量[mm]×10を整数に丸め、上限100とする。"""
    if not math.isfinite(r60_mm) or r60_mm < 0:
        raise ValueError("合計降水量は0以上の有限値で指定してください")
    return min(100, int(math.floor(r60_mm * 10 + 0.5)))


def extract(row):
    """未来の値はラベル専用。入力特徴は過去のセンサー値だけから作る。"""
    t0 = datetime.strptime(row["t0_utc"], "%Y-%m-%dT%H:%MZ")
    if row["split"] != SPLITS.get(t0.year):
        raise ValueError(f"split と年が不一致: {row['sample_id']}")
    # 年境界を跨ぐ窓を除き、隣接 split 間で観測・正解が重ならないようにする。
    if (t0 - timedelta(minutes=60)).year != t0.year or (t0 + timedelta(minutes=60)).year != t0.year:
        return None, "year_boundary"
    series = {name: [number(row, f"{name}_{o}") for o in OFFSETS]
              for name in ("T", "RH", "P")}
    past_r = [number(row, f"R_{o}") for o in range(-30, 0)]
    future = [number(row, f"R_+{o}") for o in range(1, 61)]
    if any(v is None for values in (*series.values(), past_r, future) for v in values):
        return None, "missing"
    if any(not -90 <= v <= 60 for v in series["T"]) or any(not 0 <= v <= 100 for v in series["RH"]):
        return None, "invalid_sensor"
    if any(not 450 <= v <= 1100 for v in series["P"]) or any(v < 0 for v in past_r + future):
        return None, "invalid_sensor"
    dp = [None] + [round(b - a, 2) for a, b in zip(series["P"], series["P"][1:])]
    features = {**series, "dP_5min": dp}
    features["delta_T_55min"] = round(series["T"][-1] - series["T"][0], 2)
    features["delta_RH_55min"] = round(series["RH"][-1] - series["RH"][0], 1)
    features["delta_P_55min"] = round(series["P"][-1] - series["P"][0], 2)
    features["P_volatility_hPa"] = round(pressure_volatility(series["P"]), 3)   # 主指標: 気圧の上下動
    alert = features["delta_P_55min"] <= -1.0 or (
        series["RH"][-1] >= 85 and features["delta_RH_55min"] >= 10)
    r60 = round(sum(future), 3)
    peak5 = round(max_window(future, 5), 3)
    past5 = round(sum(past_r[-5:]), 3)
    first_rain = next((i + 1 for i, value in enumerate(future) if value > 0), None)
    was_dry = sum(past_r) <= 0.254 + 1e-8
    if was_dry and first_rain is not None and first_rain <= 30 and peak5 >= 1.0:
        bucket = "sudden_onset"
    elif not was_dry and peak5 >= 1.0 and peak5 - past5 >= 1.0:
        bucket = "rapid_intensification"
    elif was_dry and r60 >= 0.5:
        bucket = "rain_onset"
    elif r60 >= 0.5:
        bucket = "ongoing_rain"
    elif r60 > 0:
        bucket = "light_rain"
    else:
        bucket = "sensor_alert_no_rain" if alert else "dry"
    result = {"sample_id": row["sample_id"], "station": row["station"],
              "t0_utc": row["t0_utc"], "split": row["split"], "category": bucket,
              "source_weight": number(row, "weight"), "features": features,
              "outcome": {"R60_mm": r60, "weather_risk": weather_risk(r60), "max5_mm": peak5,
                          "past30_mm": round(sum(past_r), 3), "past5_mm": past5,
                          "first_rain_min": first_rain}}
    return result, None


def explain(record):
    """リスクは実測教師値。理由は過去観測だけから作り、気圧の上下動を主・湿度/気温を従として述べる。"""
    f = record["features"]
    vol, net = f["P_volatility_hPa"], f["delta_P_55min"]
    rh, t = f["RH"][-1], f["T"][-1]
    low, high = VOLATILITY_BANDS
    # 1) 主指標: 気圧の上下動の大きさ
    if vol >= high:
        parts = [f"現地気圧は5分ごとに大きく上下しており（変化のばらつき{vol:.2f}hPa）、"
                 "大気が静穏でなく擾乱の影響を受けている兆候で、荒天リスクを高める主因となる。"]
    elif vol >= low:
        parts = [f"現地気圧は5分ごとに小さく上下しており（変化のばらつき{vol:.2f}hPa）、"
                 "やや不安定な場を示すが、変動は中程度にとどまる。"]
    else:
        parts = [f"現地気圧はほぼ平坦で5分ごとの揺れが小さく（変化のばらつき{vol:.2f}hPa）、"
                 "静穏な場を示し、荒天リスクを下げる方向に働く。"]
    parts.append(f"なお55分の正味変化は{net:+.2f}hPaだが、上下の向きだけでは荒天は判断しない。")
    # 2) 補助指標: 湿度（気圧変動より優先しない）
    if rh >= 90:
        parts.append(f"補助的に見ると、相対湿度は{rh:.1f}%と飽和に近く、冷却されれば凝結が起こりやすい。")
    elif rh >= 75:
        parts.append(f"補助的に見ると、相対湿度は{rh:.1f}%とやや高く、飽和に近づきやすい。")
    else:
        parts.append(f"補助的に見ると、相対湿度は{rh:.1f}%で飽和から離れており、強い湿りの兆候は弱い。")
    # 3) さらに弱い補助: 気温と湿度上昇の同時変化
    if f["delta_RH_55min"] >= 10 and f["delta_T_55min"] < 0:
        parts.append(f"気温は{t:.1f}℃で、55分で{f['delta_T_55min']:+.1f}℃の低下と"
                     f"{f['delta_RH_55min']:+.1f}ポイントの湿度上昇が重なり、空気が飽和へ近づいている。")
    return {"weather_risk": weather_risk(record["outcome"]["R60_mm"]), "reason": "".join(parts)}


def chat_record(record):
    f = record["features"]
    user = {"station": record["station"], "t0_utc": record["t0_utc"],
            "min": list(OFFSETS), "T": f["T"], "RH": f["RH"], "P": f["P"], "dP": f["dP_5min"]}
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                         {"role": "assistant", "content": json.dumps(explain(record), ensure_ascii=False)}]}


def priority(sample_id, seed):
    return int.from_bytes(hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()[:8], "big")


def update_tendencies(tendencies, record):
    # 評価年を使って特徴の傾向や閾値を調整しない。
    if record["split"] != "train":
        return
    counts = tendencies.setdefault(record["category"], Counter())
    f = record["features"]
    counts["count"] += 1
    counts["pressure_volatility_ge_high"] += f["P_volatility_hPa"] >= VOLATILITY_BANDS[1]
    counts["rh_ge85_pct"] += f["RH"][-1] >= 85
    counts["rh_rise_ge10_points"] += f["delta_RH_55min"] >= 10
    counts["sum_P_volatility_hPa"] += f["P_volatility_hPa"]
    counts["sum_delta_RH_55min_points"] += f["delta_RH_55min"]


def remember(heap, record, capacity, seed):
    if capacity <= 0:
        return
    item = (-priority(record["sample_id"], seed), record["sample_id"], record)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def select(pools, limit, spacing):
    """各層の目標比率を優先。足りない層は他層で補充し、近接時刻を抑える。"""
    selected, used_times, used_ids = [], [], set()

    def take(record):
        if record["sample_id"] in used_ids:
            return False
        t0 = datetime.strptime(record["t0_utc"], "%Y-%m-%dT%H:%MZ")
        if any(abs((t0 - previous).total_seconds()) < spacing * 60 for previous in used_times):
            return False
        selected.append(record)
        used_times.append(t0)
        used_ids.add(record["sample_id"])
        return True

    for bucket, ratio in BUCKETS.items():
        count = 0
        for _, _, record in sorted(pools.get(bucket, []), reverse=True):
            if count >= int(limit * ratio) or len(selected) >= limit:
                break
            count += take(record)
    remaining = sorted((item for pool in pools.values() for item in pool), reverse=True)
    for _, _, record in remaining:
        if len(selected) >= limit:
            break
        take(record)
    return sorted(selected, key=lambda record: record["t0_utc"])


def generate(source, output, limits, seed=20261014, spacing=60, stations=None, overwrite=False):
    paths = sorted(source.glob("dataset_*.csv"))
    paths = [p for p in paths if p.stem != "dataset_summary" and
             (stations is None or p.stem.removeprefix("dataset_") in stations)]
    if not paths:
        raise ValueError("入力の地点CSVがありません")
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / f"{split}{suffix}" for split in limits for suffix in (".jsonl", ".metadata.jsonl")]
    destinations.append(output / "manifest.json")
    if not overwrite and any(path.exists() for path in destinations):
        raise FileExistsError("既存の出力は上書きしません。別の --out または --overwrite を指定してください")
    manifest = {"seed": seed, "per_station_limits": limits, "spacing_min": spacing,
                "output_schema": {"weather_risk": "integer: min(round(R60_mm * 10), 100), 0..100",
                                  "reason": "string: 気圧の上下動を主指標とした短い判断理由"},
                "primary_signal": {"name": "P_volatility_hPa",
                                   "definition": "過去55分の5分ごと気圧変化の標準偏差",
                                   "bands_hPa": VOLATILITY_BANDS,
                                   "validation": "荒天(risk>=50) AUC≈test 0.75 > RH 0.62 > 正味気圧変化 0.42"},
                "target_category_ratios": BUCKETS, "stations": {}, "train_tendencies": {},
                "notes": ["教師文はルールベースの根拠要約であり、専門家による個別監修済みではない。",
                          "reasonは気圧の上下動（P_volatility_hPa）を最優先し、湿度・気温は補助として扱う。",
                          "正味の気圧増減（上がった/下がった）はこの粒度では予測力がないため向きには著さない。",
                          "正解はt0直後の60分間の降水。雨雪判別・雷・突風・雲量は正解に含まれない。",
                          "未来の観測はweather_riskの教師値と別ファイルのメタデータにのみ使用。reasonは過去観測だけから生成。",
                          "急変優先の選別で自然頻度は変わる。source_weightは追加選別を補正しない。",
                          "dev/testも急変重点のチャレンジセット。母集団性能・確率較正は元CSVで別途評価する。",
                          "予測可能な前兆が常にあるとは限らない。観測との関連は因果関係の証明ではない。"]}
    handles = {split: ((output / f"{split}.jsonl").open("w", encoding="utf-8"),
                       (output / f"{split}.metadata.jsonl").open("w", encoding="utf-8")) for split in limits}
    tendencies = {}
    try:
        for path in paths:
            station = path.stem.removeprefix("dataset_")
            pools = {split: {bucket: [] for bucket in BUCKETS} for split in limits}
            available, skipped = Counter(), Counter()
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["station"] != station:
                        raise ValueError(f"ファイル名と地点が不一致: {path.name}")
                    record, reason = extract(row)
                    if reason:
                        skipped[reason] += 1
                        continue
                    split, bucket = record["split"], record["category"]
                    if split not in limits:
                        continue
                    update_tendencies(tendencies, record)
                    available[f"{split}/{bucket}"] += 1
                    # 各層に上限の2倍を候補として保管し、メモリを一定に抑える。
                    remember(pools[split][bucket], record, limits[split] * 2, seed)
            counts = Counter()
            for split, limit in limits.items():
                for record in select(pools[split], limit, spacing):
                    chat, metadata = handles[split]
                    chat.write(json.dumps(chat_record(record), ensure_ascii=False) + "\n")
                    metadata.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[f"{split}/{record['category']}"] += 1
            manifest["stations"][station] = {"available": dict(available), "selected": dict(counts), "skipped": dict(skipped)}
            print(f"{station}: {sum(counts.values())} selected; {dict(counts)}", flush=True)
    finally:
        for pair in handles.values():
            for handle in pair:
                handle.close()
    for bucket, counts in tendencies.items():
        n = counts["count"]
        manifest["train_tendencies"][bucket] = {
            "eligible_source_rows": n,
            "pressure_volatility_ge_high_pct": round(100 * counts["pressure_volatility_ge_high"] / n, 2),
            "rh_ge85_pct": round(100 * counts["rh_ge85_pct"] / n, 2),
            "rh_rise_ge10_points_pct": round(100 * counts["rh_rise_ge10_points"] / n, 2),
            "mean_P_volatility_hPa": round(counts["sum_P_volatility_hPa"] / n, 3),
            "mean_delta_RH_55min_points": round(counts["sum_delta_RH_55min_points"] / n, 3),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "eval/data/dataset_asos")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/data/weather_sft")
    parser.add_argument("--stations", nargs="+")
    parser.add_argument("--overwrite", action="store_true", help="生成済みJSONLとmanifestの上書きを明示的に許可")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--spacing-min", type=int, default=60)
    for split, limit in LIMITS.items():
        parser.add_argument(f"--{split}-per-station", type=int, default=limit)
    args = parser.parse_args()
    limits = {split: getattr(args, f"{split}_per_station") for split in LIMITS}
    if any(v < 0 for v in limits.values()) or args.spacing_min < 0:
        parser.error("件数と間隔は0以上にしてください")
    generate(args.source, args.out, limits, args.seed, args.spacing_min, args.stations, args.overwrite)


if __name__ == "__main__":
    main()