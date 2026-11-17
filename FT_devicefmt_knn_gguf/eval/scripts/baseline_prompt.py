"""実機（bme280_ros_with_uno_q）の build_prompt() と同じ文面を作る関数と、vendored 原本との一致検査。

- build_prompt_p0(): 参照リポジトリ prompt_maker_for_bme280.py の build_prompt() を、ROS 依存を外して写したもの（P0 = 現行文面）。
  以後のプロンプト調整（P1 丸め・P2 要約など）はこのファイルに build_prompt_p1() … として足す。
- vendored_build_prompt(): eval/vendor/ の原本を rclpy 等のスタブ付きで import し、そのまま呼ぶ（検査用）。
"""
from __future__ import annotations
import importlib.util, pathlib, sys, types
from typing import List

VENDOR = pathlib.Path(__file__).resolve().parents[1] / "vendor" / "bme280_ros_with_uno_q" / "prompt_maker_for_bme280.py"
BUCKET_SECONDS = 300          # 実機の既定 bucket_seconds（5 分）
HISTORY_BUCKETS = 12          # 実機の既定 history_buckets（1 時間）


def build_prompt_p0(temperatures: List[float], humidities: List[float], pressures: List[float],
                    bucket_seconds: float = BUCKET_SECONDS, **meta) -> str:
    """参照リポジトリ build_prompt() の写し（P0: 現行文面）。"""
    def fmt(vs: List[float]) -> str:
        return ', '.join(f'{v:.2f}' for v in vs)
    n = len(temperatures)
    span_minutes = bucket_seconds * n / 60.0
    return (
        f'以下は、温湿度・気圧センサの直近 {span_minutes:g} 分間の測定値を、'
        f'{bucket_seconds:g} 秒ごとに取得したものです '
        f'(古い順、各 {n} 個)。\n'
        f'温度 [C]: {fmt(temperatures)}\n'
        f'湿度 [%]: {fmt(humidities)}\n'
        f'気圧 [hPa]: {fmt(pressures)}\n'
        'この情報を元にこの先1時間以内に天候が悪化する可能性を0から100の範囲で評価してください。\n'
        'あくまで推測でかまいません。回答は数値だけを端的に返したのち、改行文字を挟んで、その後短く理由を教えてください\n'
    )


def _load_vendored():
    """rclpy / ROS メッセージをスタブにして原本を import する。"""
    for name in ("rclpy", "rclpy.node", "bme280_sensor_msgs", "bme280_sensor_msgs.msg", "std_msgs", "std_msgs.msg"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            if name == "rclpy.node":
                m.Node = object
            if name.endswith(".msg"):
                m.Bme280 = object; m.String = object
            sys.modules[name] = m
    spec = importlib.util.spec_from_file_location("vendored_prompt_maker", VENDOR)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod   # dataclass が sys.modules を参照するため先に登録
    spec.loader.exec_module(mod)
    return mod


def vendored_build_prompt(temperatures, humidities, pressures, bucket_seconds: float = BUCKET_SECONDS) -> str:
    mod = _load_vendored()
    fake = types.SimpleNamespace(_bucket_seconds=float(bucket_seconds))
    return mod.PromptMakerForBme280.build_prompt(fake, list(temperatures), list(humidities), list(pressures))


# ---------------------------------------------------------------------------
# プロンプト調整の候補（提案書 014 §4）。入出力の仕様（直近 60 分 → 0〜100 の整数 1 行目 + 理由）は P0 と同じ。
# ---------------------------------------------------------------------------
_TAIL = ('この情報を元にこの先1時間以内に天候が悪化する可能性を0から100の範囲で評価してください。\n'
         'あくまで推測でかまいません。回答は数値だけを端的に返したのち、改行文字を挟んで、その後短く理由を教えてください\n')
HINT = '一般に、湿度が高く（90% 以上）、気圧が下がっている時ほど、1 時間以内に雨が降りやすいとされます。\n'
SCALE = '0から100の整数で答えてください。10刻みでなくてかまいません（0 = 降らない、100 = 確実に降る）。\n'


def build_prompt_p1(temperatures, humidities, pressures, bucket_seconds: float = BUCKET_SECONDS, **meta) -> str:
    """P1 丸め: P0 と同じ構成で値を整数に丸める（トークン削減だけの効果を分離する対照）。"""
    def fmt(vs):
        return ', '.join(f'{v:.0f}' for v in vs)
    n = len(temperatures); span_minutes = bucket_seconds * n / 60.0
    return (f'以下は、温湿度・気圧センサの直近 {span_minutes:g} 分間の測定値を、{bucket_seconds:g} 秒ごとに取得したものです (古い順、各 {n} 個)。\n'
            f'温度 [C]: {fmt(temperatures)}\n湿度 [%]: {fmt(humidities)}\n気圧 [hPa]: {fmt(pressures)}\n' + _TAIL)


def summary_lines(temperatures, humidities, pressures) -> str:
    """P2 系の要約部分: 現在値・60/30 分の変化・60 分の範囲（12 個の 5 分平均から計算）。"""
    T, H, P = list(temperatures), list(humidities), list(pressures); n = len(T); h = n // 2
    return (f'以下は、温湿度・気圧センサの直近 60 分間の測定値の要約です。\n'
            f'現在: 温度 {T[-1]:.1f} C、湿度 {H[-1]:.0f} %、気圧 {P[-1]:.1f} hPa\n'
            f'60 分間の変化: 気圧 {P[-1]-P[0]:+.1f} hPa、湿度 {H[-1]-H[0]:+.0f} ポイント、温度 {T[-1]-T[0]:+.1f} C\n'
            f'直近 30 分間の変化: 気圧 {P[-1]-P[h-1]:+.1f} hPa、湿度 {H[-1]-H[h-1]:+.0f} ポイント\n'
            f'60 分間の範囲: 気圧 {min(P):.1f}〜{max(P):.1f} hPa、湿度 {min(H):.0f}〜{max(H):.0f} %\n')


def build_prompt_p2(temperatures, humidities, pressures, bucket_seconds: float = BUCKET_SECONDS, **meta) -> str:
    """P2 要約: 36 値の列挙を要約量に置き換える。"""
    return summary_lines(temperatures, humidities, pressures) + _TAIL


def build_prompt_p2h(temperatures, humidities, pressures, bucket_seconds: float = BUCKET_SECONDS, **meta) -> str:
    """P2' 要約 + 判断の手がかり 1 文。"""
    return summary_lines(temperatures, humidities, pressures) + HINT + _TAIL


def build_prompt_p3(temperatures, humidities, pressures, bucket_seconds: float = BUCKET_SECONDS, **meta) -> str:
    """P3 = P2' + 尺度の指示（10 刻みでなくてよい）。"""
    return summary_lines(temperatures, humidities, pressures) + HINT + SCALE + _TAIL



# ---------------------------------------------------------------------------
# 追加プロンプトパターン ①・②（018_instruction.md）。役割付与・表形式の観測データ・4 段階判定・出力フォーマット指定。
# 指示書との差分: 「計13時点」→「計12時点」（実機の履歴は 5 分平均 × 12 区切りで 13 時点は無い）。
# 時刻は各 5 分区切りの終端を地点の地方時（zoneinfo）で表す。最後の行が「現在」。値の桁は指示書の例に合わせる（気圧 0.1、湿度 1、気温 0.1）。
# ---------------------------------------------------------------------------
STATION_TZ = {"DSM": "America/Chicago", "IAH": "America/Chicago", "SEA": "America/Los_Angeles", "ATL": "America/New_York", "BOS": "America/New_York"}

Q_HEAD = ('あなたは気象データ解析の専門家AIです。\n'
          '以下に提供される「過去1時間（5分おき、計12時点）」の気圧・湿度・気温の観測データを分析し、今後1時間の【降雨危険度】を評価・予測してください。\n'
          '\n'
          '### 判定指示\n'
          '1. 過去1時間における各要素（気圧の低下傾向、湿度の増加傾向、気温の急変）の変化量を算出して分析してください。\n'
          '2. 今後1時間以内に雨が降る可能性を、以下の4段階で判定してください。\n'
          '   - 【危険】（80%以上）：極めて高い確率で降雨または立チ昇る雨雲が発生する\n'
          '   - 【警戒】（50〜70%）：降雨の可能性が高く、急な雨に備える必要がある\n'
          '   - 【注意】（20〜40%）：気象条件が変化しつつあるが、直ちに降る可能性は低い\n'
          '   - 【安全】（20%未満）：降雨の兆候は見られない\n'
          '3. 判断の根拠となった気象データの変化パターンを箇条書きで簡潔に説明してください。\n'
          '\n'
          '### 観測データ（過去1時間 / 5分間隔）\n')
Q_TAIL = ('\n'
          '### 出力フォーマット\n'
          '以下のフォーマットに従って回答を出力してください。\n'
          '\n'
          '【降雨危険度】：[危険 / 警戒 / 注意 / 安全]\n'
          '【降雨確率の目安】：[◯◯%]\n'
          '【分析サマリー】：\n'
          '- 気圧の推移：\n'
          '- 湿度の推移：\n'
          '- 気温の推移：\n'
          '【判定理由・今後の見通し】：\n'
          '（ここに100〜200文字程度で解説を入力）\n')


def bucket_clock_labels(n: int, t0_utc: str | None, station: str | None, bucket_seconds: float = BUCKET_SECONDS) -> List[str]:
    """各区切りの終端時刻（地方時 HH:MM）。t0 が無ければ相対表記（-55分 … 現在）。"""
    from datetime import datetime, timedelta, timezone
    step = timedelta(seconds=bucket_seconds)
    if t0_utc and station in STATION_TZ:
        from zoneinfo import ZoneInfo
        t0 = datetime.strptime(t0_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc).astimezone(ZoneInfo(STATION_TZ[station]))
        return [(t0 - step * (n - 1 - k)).strftime("%H:%M") for k in range(n)]
    return [f"{-int(step.total_seconds() // 60) * (n - 1 - k)}分" if k < n - 1 else "現在" for k in range(n)]


def build_prompt_q1(temperatures, humidities, pressures, bucket_seconds: float = BUCKET_SECONDS, **meta) -> str:
    """①: 行 = 時刻、列 = 気圧・湿度・気温（指示書の表と同じ向き）。"""
    labels = bucket_clock_labels(len(temperatures), meta.get("t0_utc"), meta.get("station"), bucket_seconds)
    rows = "\n".join(f"{lb} | {p:.1f} | {h:.0f} | {t:.1f}" for lb, t, h, p in zip(labels, temperatures, humidities, pressures))
    return Q_HEAD + "時刻 | 気圧 (hPa) | 湿度 (%) | 気温 (°C)\n---|---|---|---\n" + rows + "\n" + Q_TAIL


def build_prompt_q2(temperatures, humidities, pressures, bucket_seconds: float = BUCKET_SECONDS, **meta) -> str:
    """②: ① の表を縦横入れ替え（行 = 要素、列 = 時刻）。"""
    labels = bucket_clock_labels(len(temperatures), meta.get("t0_utc"), meta.get("station"), bucket_seconds)
    head = "要素 | " + " | ".join(labels) + "\n" + "---|" + "|".join("---" for _ in labels) + "\n"
    body = ("気圧 (hPa) | " + " | ".join(f"{p:.1f}" for p in pressures) + "\n"
            "湿度 (%) | " + " | ".join(f"{h:.0f}" for h in humidities) + "\n"
            "気温 (°C) | " + " | ".join(f"{t:.1f}" for t in temperatures) + "\n")
    return Q_HEAD + head + body + Q_TAIL


# ---------------------------------------------------------------------------
# 追加プロンプトパターン ①'（021_instruction.md）: ① の判定指示 2 を「4 段階」から
# 「今後 1 時間の降水量予測に 10 をかけて整数化（100 以上は 100）」= 正解 risk2 と同じ定義 に差し替え、出力フォーマットも合わせる。
# 表の向き（行=時刻）・役割付与・分析手順・解説の書式は ① と同じ。
# ---------------------------------------------------------------------------
Q3_HEAD = ('あなたは気象データ解析の専門家AIです。\n'
           '以下に提供される「過去1時間（5分おき、計12時点）」の気圧・湿度・気温の観測データを分析し、今後1時間の【降水量】を評価・予測してください。\n'
           '\n'
           '### 判定指示\n'
           '1. 過去1時間における各要素（気圧の低下傾向、湿度の増加傾向、気温の急変）の変化量を算出して分析してください。\n'
           '2. 今後1時間の降水量予測（mm）に10をかけて整数化したものを【降雨指数】として判定してください。100以上の場合は100にまとめてください。\n'
           '3. 判断の根拠となった気象データの変化パターンを箇条書きで簡潔に説明してください。\n'
           '\n'
           '### 観測データ（過去1時間 / 5分間隔）\n')
Q3_TAIL = ('\n'
           '### 出力フォーマット\n'
           '以下のフォーマットに従って回答を出力してください。\n'
           '\n'
           '【降水量予測】：[◯.◯ mm]\n'
           '【降雨指数】：[0〜100の整数]\n'
           '【分析サマリー】：\n'
           '- 気圧の推移：\n'
           '- 湿度の推移：\n'
           '- 気温の推移：\n'
           '【判定理由・今後の見通し】：\n'
           '（ここに100〜200文字程度で解説を入力）\n')


def build_prompt_q3(temperatures, humidities, pressures, bucket_seconds: float = BUCKET_SECONDS, **meta) -> str:
    """①': ① と同じ表（行 = 時刻）で、判定を 4 段階でなく「降水量 × 10 の整数（0〜100）」にしたもの。"""
    labels = bucket_clock_labels(len(temperatures), meta.get("t0_utc"), meta.get("station"), bucket_seconds)
    rows = "\n".join(f"{lb} | {p:.1f} | {h:.0f} | {t:.1f}" for lb, t, h, p in zip(labels, temperatures, humidities, pressures))
    return Q3_HEAD + "時刻 | 気圧 (hPa) | 湿度 (%) | 気温 (°C)\n---|---|---|---\n" + rows + "\n" + Q3_TAIL


BUILDERS = {"p0": build_prompt_p0, "p1": build_prompt_p1, "p2": build_prompt_p2, "p2h": build_prompt_p2h, "p3": build_prompt_p3,
            "q1": build_prompt_q1, "q2": build_prompt_q2, "q3": build_prompt_q3}
# 回答の読み取り方: p* は 1 行目の整数（first_line）、q1/q2 は書式付き（fmt4: 【降雨確率の目安】の % と【降雨危険度】の 4 段階）、
# q3 は fmtidx（【降雨指数】の整数と【降水量予測】の mm）
PARSER = {k: ("fmtidx" if k == "q3" else "fmt4" if k.startswith("q") else "first_line") for k in BUILDERS}


if __name__ == "__main__":
    t = [23.91, 23.88, 23.9, 23.85, 23.8, 23.82, 23.79, 23.75, 23.8, 23.78, 23.77, 23.8]
    h = [90.12, 91.04, 92.5, 93.1, 94.0, 93.5, 92.8, 92.0, 91.5, 91.2, 91.0, 91.3]
    p = [978.01, 977.95, 977.8, 977.6, 977.2, 976.9, 976.5, 976.1, 975.9, 976.3, 976.9, 977.3]
    a, b = build_prompt_p0(t, h, p), vendored_build_prompt(t, h, p)
    assert a == b, "mismatch"
    print(a); print("vendored match: OK")
    for k, f in BUILDERS.items():
        print(f"===== {k} =====\n" + f(t, h, p, t0_utc="2023-02-08T20:50Z", station="IAH"))
