# vendor: gomi-kuzu/bme280_ros_with_uno_q（無改変の写し）
- 出典: https://github.com/gomi-kuzu/bme280_ros_with_uno_q （Apache-2.0、LICENSE 同梱）
- コミット: COMMIT ファイル参照（2026-10-15 取得）
- 用途: `prompt_maker_for_bme280.py` の `build_prompt()` が作る文面を評価側で寸分違わず再現するための参照。
  rclpy 等は入れていないので、`eval/scripts/make_baseline_prompts.py` がスタブを差して関数だけを呼ぶ。
