"""実測未来値の入力混入、欠測、急変定義、再現性の回帰テスト。"""
import json
import tempfile
import unittest
from pathlib import Path

from util.make_weather_sft import (
    chat_record, explain, extract, generate, pressure_volatility, remember, select, weather_risk,
)


def row_at(time="2019-06-01T12:00Z"):
    row = {"sample_id": "DSM_" + time, "station": "DSM", "split": "train", "t0_utc": time, "weight": "1"}
    for o in range(-60, 0):
        row.update({f"T_{o}": "25", f"RH_{o}": "80", f"P_{o}": "1000", f"R_{o}": "0"})
    row.update({f"R_+{o}": "0" for o in range(1, 61)})
    return row


class WeatherSFTTest(unittest.TestCase):
    def test_sudden_onset_and_no_future_in_input(self):
        dry, _ = extract(row_at())
        row = row_at()
        row["R_+10"] = "1.016"
        wet, reason = extract(row)
        self.assertIsNone(reason)
        self.assertEqual(wet["category"], "sudden_onset")
        self.assertEqual(wet["outcome"]["first_rain_min"], 10)
        self.assertEqual(chat_record(dry)["messages"][:2], chat_record(wet)["messages"][:2])
        answer = json.loads(chat_record(wet)["messages"][-1]["content"])
        self.assertEqual(set(answer), {"weather_risk", "reason"})
        self.assertEqual(answer["weather_risk"], 10)
        self.assertEqual(answer["reason"], explain(dry)["reason"])
        observations = json.loads(chat_record(wet)["messages"][1]["content"])
        self.assertEqual(len(observations["min"]), 12)
        self.assertIsNone(observations["dP"][0])

    def test_missing_and_year_boundary(self):
        row = row_at()
        row["R_+60"] = ""
        self.assertEqual(extract(row)[1], "missing")
        self.assertEqual(extract(row_at("2019-12-31T23:30Z"))[1], "year_boundary")
        row = row_at()
        row["P_-60"] = "nan"
        self.assertEqual(extract(row)[1], "missing")

    def test_pressure_false_alarm(self):
        row = row_at()
        row["P_-5"] = "998.5"
        record, _ = extract(row)
        self.assertEqual(record["category"], "sensor_alert_no_rain")
        self.assertEqual(record["features"]["dP_5min"][-1], -1.5)
        reason = explain(record)["reason"]
        self.assertIn("変化のばらつき", reason)
        self.assertIn("正味変化", reason)
        self.assertEqual(explain(record)["weather_risk"], 0)

    def test_pressure_volatility_leads_reason(self):
        # 静穏（平坦）な場は荒天リスクを下げる向きの文言、変動大は主因の文言。
        calm, _ = extract(row_at())
        self.assertEqual(calm["features"]["P_volatility_hPa"], 0.0)
        calm_reason = explain(calm)["reason"]
        self.assertTrue(calm_reason.startswith("現地気圧はほぼ平坦"))
        self.assertIn("静穏な場", calm_reason)
        # 気圧を分単位で大きく上下させる
        row = row_at()
        for i, o in enumerate(range(-60, 0, 5)):
            row[f"P_{o}"] = str(1000 + (3 if i % 2 else -3))
        rough, _ = extract(row)
        self.assertGreaterEqual(rough["features"]["P_volatility_hPa"], 0.14)
        rough_reason = explain(rough)["reason"]
        self.assertTrue(rough_reason.startswith("現地気圧は5分ごとに大きく上下"))
        self.assertIn("主因", rough_reason)
        # 湿度・気温は補助として気圧の後に登場する
        self.assertLess(rough_reason.index("気圧"), rough_reason.index("相対湿度"))

    def test_pressure_volatility_formula(self):
        self.assertEqual(pressure_volatility([1000.0] * 12), 0.0)
        self.assertAlmostEqual(pressure_volatility([1000, 1001] * 6 + [1000]), 1.0, places=6)

    def test_intensification_and_light_rain(self):
        row = row_at()
        row["R_-20"] = "0.508"
        row["R_+15"] = "1.016"
        self.assertEqual(extract(row)[0]["category"], "rapid_intensification")
        row["R_+15"] = "0.254"
        self.assertEqual(extract(row)[0]["category"], "light_rain")

    def test_offline_reason_without_boilerplate_or_future(self):
        row = row_at()
        row["R_+10"] = "0.508"
        record, _ = extract(row)
        reason = explain(record)["reason"]
        for phrase in ("レーダー", "確認してください", "正解", "0.508", "断定でき", "注意してください", "雷", "雪"):
            self.assertNotIn(phrase, reason)
        self.assertIn("相対湿度", reason)
        self.assertIn("現地気圧", reason)

    def test_weather_risk_formula_integer_and_cap(self):
        for amount, expected in ((0, 0), (0.254, 3), (0.508, 5),
                                 (1.016, 10), (9.999, 100), (10, 100), (25.4, 100)):
            with self.subTest(amount=amount):
                self.assertEqual(weather_risk(amount), expected)
                row = row_at()
                row["R_+1"] = str(amount)
                record, _ = extract(row)
                self.assertEqual(record["outcome"]["weather_risk"], expected)
                self.assertEqual(explain(record)["weather_risk"], expected)
        for invalid in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                weather_risk(invalid)

    def test_reservoir_reproducibility_and_spacing(self):
        records = [extract(row_at(f"2019-06-01T{hour:02}:{minute:02}Z"))[0]
                   for hour in range(10, 14) for minute in (0, 5, 10)]
        a, b = [], []
        for record in records:
            remember(a, record, 8, 42)
        for record in reversed(records):
            remember(b, record, 8, 42)
        self.assertEqual(sorted(x[1] for x in a), sorted(x[1] for x in b))
        chosen = select({"dry": a}, 8, 60)
        self.assertLessEqual(len(chosen), 4)
        self.assertEqual(select({"dry": a}, 0, 60), [])

    def test_file_generation_and_overwrite_guard(self):
        import csv
        with tempfile.TemporaryDirectory() as temp:
            source, out = Path(temp) / "source", Path(temp) / "out"
            source.mkdir()
            row = row_at()
            with (source / "dataset_DSM.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
            generate(source, out, {"train": 1})
            self.assertEqual(len((out / "train.jsonl").read_text().splitlines()), 1)
            with self.assertRaises(FileExistsError):
                generate(source, out, {"train": 1})
            generate(source, out, {"train": 1}, overwrite=True)
            self.assertEqual(len((out / "train.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()