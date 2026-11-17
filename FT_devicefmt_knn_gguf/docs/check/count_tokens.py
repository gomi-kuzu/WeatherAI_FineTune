#!/usr/bin/env python3
"""実装I の repo の入出力 1 件と、こちらの FT-B の入出力 1 件のトークン数を Gemma 4 のトークナイザで数える（実機所要の見積り用）。
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/w -w /w gemma4-train python /w/report/261105_実装Iのrepositoryと比較/check/count_tokens.py
実装I 側の文字列は、README の user 例と make_weather_sft.py の SYSTEM・explain() の文面（2026-11-03 commit 9c6dcf3）をこのファイルに写したもの（外部コードは実行しない）。
"""
import json
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/w/models/hf/gemma-4-E2B-it/tokenizer.json")
n = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
SYSTEM = ("山中オフライン用の気象予測。過去1時間・5分間隔のT(気温℃)、RH(相対湿度%)、"
    "P(現地気圧hPa)、dP(5分気圧差)から今後1時間の荒天リスクを出す。各配列は-60..-5分の順。"
    "出力はJSONで weather_risk(整数0-100) と reason(短い理由) の2項目のみ。"
    "weather_riskは今後1時間の合計降水量mm×10で100上限、降水量の指標で降水確率ではない。"
    "判断は気圧の細かな上下動(dPのばらつき)を最重視し、変動が大きいほど高リスク、"
    "平坦なら低リスク。湿度と気温は補助で気圧変動より優先しない。"
    "気圧が正味で上下した向きだけでは判断しない。雷・突風・雨雪は断定しない。"
    "reasonは観測の変化と意味を簡潔に。未来値を既知として述べず、注意書きやレーダー確認案内は不要。")
user = {"station": "FIELD", "t0_utc": "2026-11-03T05:00Z", "min": [-60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5],
        "T": [8.21, 8.19, 8.15, 8.12, 8.08, 8.05, 8.0, 7.96, 7.93, 7.9, 7.88, 7.85],
        "RH": [72.4, 72.9, 73.5, 74.1, 74.8, 75.6, 76.3, 77.0, 77.8, 78.5, 79.1, 79.8],
        "P": [912.35, 912.41, 912.33, 912.5, 912.28, 912.47, 912.3, 912.52, 912.27, 912.55, 912.22, 912.58],
        "dP": [None, 0.06, -0.08, 0.17, -0.22, 0.19, -0.17, 0.22, -0.25, 0.28, -0.33, 0.36]}
U = json.dumps(user, ensure_ascii=False)
# explain() の最短（静穏・湿度中・気温文なし）と最長（変動大・湿度高・気温文あり）
short = ("現地気圧はほぼ平坦で5分ごとの揺れが小さく（変化のばらつき0.04hPa）、静穏な場を示し、荒天リスクを下げる方向に働く。"
         "なお55分の正味変化は+0.12hPaだが、上下の向きだけでは荒天は判断しない。"
         "補助的に見ると、相対湿度は80.1%とやや高く、飽和に近づきやすい。")
long_ = ("現地気圧は5分ごとに大きく上下しており（変化のばらつき0.21hPa）、大気が静穏でなく擾乱の影響を受けている兆候で、荒天リスクを高める主因となる。"
         "なお55分の正味変化は-0.85hPaだが、上下の向きだけでは荒天は判断しない。"
         "補助的に見ると、相対湿度は93.4%と飽和に近く、冷却されれば凝結が起こりやすい。"
         "気温は21.3℃で、55分で-2.4℃の低下と+12.5ポイントの湿度上昇が重なり、空気が飽和へ近づいている。")
A = lambda r: json.dumps({"weather_risk": 30, "reason": r}, ensure_ascii=False)
# Gemma 4 のチャット枠（train_guide §7: <|turn>{role}\n…<turn|>）。system 非対応時の統合などの差は数トークン
chat_in = f"<|turn>system\n{SYSTEM}<turn|>\n<|turn>user\n{U}<turn|>\n<|turn>model\n"
sft = [json.loads(l) for l in open("/w/eval/data/sft/sft_B.jsonl", encoding="utf-8")][:200]
res = {"inoue_system": n(SYSTEM), "inoue_user": n(U), "inoue_input_total(chat)": n(chat_in),
       "inoue_output_short": n(A(short)), "inoue_output_long": n(A(long_)),
       "mine_FT-B_input_mean(200)": round(sum(n(r["prompt"]) for r in sft) / len(sft), 1),
       "mine_FT-B_output_mean(200)": round(sum(n(r["completion"]) for r in sft) / len(sft), 1)}
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open("/w/report/261105_実装Iのrepositoryと比較/check/token_counts.json", "w"), ensure_ascii=False, indent=1)
