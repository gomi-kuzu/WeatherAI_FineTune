#!/usr/bin/env python3
"""ROS2 ノード: prompt_maker_from_sensor_values

/uno_q/bme280 (bme280_sensor_msgs/Bme280) を購読して以下を行う:
  * サンプルを重ならない一定時間 (デフォルト 5 分) のバケットに蓄積する
  * 各バケットの温度・湿度・気圧を平均化する
  * 直近 history_buckets 個 (デフォルト 12 = 1 時間分) のバケット平均を
    固定長キューに保持する
  * バケットが確定するたびに、値を埋め込んだ自然言語プロンプトを
    std_msgs/String として出力トピック (デフォルト /chat_input) に publish する

LLM の呼び出しは行わない。別途 chat_topic_demo_node のようなノードが
/chat_input を購読して llama_ros のアクションを叩く構成を想定している。

自然言語のプロンプト文面は後から差し替えられるように build_prompt()
にまとめてある。ROS まわりのコードには触らず、この関数の中身だけを
編集すれば文面を変更できる。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import rclpy
from rclpy.node import Node

from bme280_sensor_msgs.msg import Bme280
from std_msgs.msg import String


# 1 バケットあたりの秒数のデフォルト値 (= 5 分)。ROS パラメタで上書き可能。
DEFAULT_BUCKET_SECONDS = 5 * 60
# キューに保持するバケット数のデフォルト値 (= 12 個 = 1 時間)。ROS パラメタで上書き可能。
DEFAULT_HISTORY_BUCKETS = 12
# 購読するトピック名 (先頭スラッシュなし、ノード名前空間を尊重)
DEFAULT_INPUT_TOPIC = 'uno_q/bme280'
# プロンプトを吐き出すトピック名 (chat_topic_demo_node の input_topic に合わせる)
DEFAULT_OUTPUT_TOPIC = 'chat_input'


@dataclass
class BucketAccumulator:
    """1 バケット分の積算値を保持するデータクラス。"""

    # このバケットの開始時刻 [秒]
    start_sec: float = 0.0
    # このバケットに積まれたサンプル数
    n: int = 0
    # 各センサ値の合計
    sum_temp: float = 0.0
    sum_hum: float = 0.0
    sum_pres: float = 0.0

    def reset(self, start_sec: float) -> None:
        """新しいバケットとして初期化する。"""
        self.start_sec = start_sec
        self.n = 0
        self.sum_temp = 0.0
        self.sum_hum = 0.0
        self.sum_pres = 0.0

    def add(self, temp: float, hum: float, pres: float) -> None:
        """1 サンプルを加算する。"""
        self.n += 1
        self.sum_temp += temp
        self.sum_hum += hum
        self.sum_pres += pres

    def average(self) -> Optional[tuple[float, float, float]]:
        """(温度, 湿度, 気圧) の平均を返す。サンプル 0 個なら None。"""
        if self.n == 0:
            return None
        return (
            self.sum_temp / self.n,
            self.sum_hum / self.n,
            self.sum_pres / self.n,
        )


class PromptMakerForBme280(Node):
    """メインとなる ROS2 ノード。"""

    def __init__(self) -> None:
        super().__init__('prompt_maker_from_sensor_values')

        # ROS パラメタ宣言
        #   bucket_seconds  : 1 バケットあたりの秒数 (デバッグ時に短くできる)
        #   history_buckets : キューに保持する最大バケット数 (= 上限履歴長)
        #   input_topic     : 購読する BME280 トピック名
        #   output_topic    : プロンプトを publish する String トピック名
        self.declare_parameter('bucket_seconds', DEFAULT_BUCKET_SECONDS)
        self.declare_parameter('history_buckets', DEFAULT_HISTORY_BUCKETS)
        self.declare_parameter('input_topic', DEFAULT_INPUT_TOPIC)
        self.declare_parameter('output_topic', DEFAULT_OUTPUT_TOPIC)

        self._bucket_seconds: float = float(
            self.get_parameter('bucket_seconds').value
        )
        self._history_buckets: int = int(
            self.get_parameter('history_buckets').value
        )
        input_topic: str = str(self.get_parameter('input_topic').value)
        output_topic: str = str(self.get_parameter('output_topic').value)

        if self._bucket_seconds <= 0.0:
            raise ValueError('bucket_seconds は正の値である必要があります')
        if self._history_buckets <= 0:
            raise ValueError('history_buckets は正の整数である必要があります')

        # 直近 history_buckets 個分の平均を保持する固定長キュー
        # 各要素は (温度, 湿度, 気圧) のタプル
        self._history: Deque[tuple[float, float, float]] = deque(
            maxlen=self._history_buckets
        )
        # 現在集計中のバケット
        self._bucket = BucketAccumulator()
        # 最初のサンプルを受け取るまで start_sec を確定できないためのフラグ
        self._bucket_initialized = False

        # BME280 購読
        self._subscription = self.create_subscription(
            Bme280,
            input_topic,
            self._on_bme280,
            10,
        )

        # プロンプト出力用パブリッシャ
        # 生成が間に合わない場合の詰まりを避けるため深さは小さめ (=1)
        self._prompt_pub = self.create_publisher(String, output_topic, 1)

        self.get_logger().info(
            f"'{input_topic}' を購読し、'{output_topic}' にプロンプトを publish します。"
            f' {self._bucket_seconds:g} 秒毎に平均化し、直近 '
            f'{self._history_buckets} バケットまで保持します '
            f'(最小 1 バケットでも即座に publish します)。'
        )

    # ------------------------------------------------------------------ #
    # 購読コールバック / バケット処理
    # ------------------------------------------------------------------ #
    def _on_bme280(self, msg: Bme280) -> None:
        """BME280 メッセージ 1 件を受信するたびに呼ばれる。"""
        # メッセージのヘッダタイムスタンプを秒に変換
        stamp = msg.header.stamp
        t_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9

        # 初回サンプル: バケットの開始時刻をここで確定する
        if not self._bucket_initialized:
            self._bucket.reset(t_sec)
            self._bucket_initialized = True

        # 現在のバケットの範囲を越えたら、バケットを閉じて次の窓を開ける。
        # サンプルが飛んだ場合に備えて while ループにしてある。
        while t_sec - self._bucket.start_sec >= self._bucket_seconds:
            self._close_bucket()
            self._bucket.reset(self._bucket.start_sec + self._bucket_seconds)

        # 今回のサンプルを現在バケットに加算
        self._bucket.add(
            float(msg.temperature),
            float(msg.humidity),
            float(msg.pressure),
        )

    def _close_bucket(self) -> None:
        """現在のバケットを平均化してキューに追加し、プロンプトを publish する。"""
        avg = self._bucket.average()
        if avg is None:
            # サンプル 0 件のウィンドウはスキップ
            return

        self._history.append(avg)
        self.get_logger().info(
            f'バケット確定: 温度={avg[0]:.2f} C, '
            f'湿度={avg[1]:.2f} %, 気圧={avg[2]:.2f} hPa '
            f'(履歴サイズ={len(self._history)}/{self._history_buckets})'
        )

        # 最小 1 バケットでもプロンプトを組み立てて publish する。
        # deque の maxlen により古いバケットは自動的に押し出される。
        if len(self._history) >= 1:
            self._publish_prompt()

    # ------------------------------------------------------------------ #
    # プロンプト生成 / publish
    # ------------------------------------------------------------------ #
    def build_prompt(
        self,
        temperatures: List[float],
        humidities: List[float],
        pressures: List[float],
    ) -> str:
        """LLM に渡す自然言語のプロンプトを組み立てる。

        文面は暫定のもの。ROS 側のコードを触らずにこの関数の中だけ
        書き換えれば運用できる。
        """
        def fmt(vs: List[float]) -> str:
            return ', '.join(f'{v:.2f}' for v in vs)

        n = len(temperatures)
        # バケット幅と個数から実測期間を算出して埋め込む
        span_minutes = self._bucket_seconds * n / 60.0
        return (
            f'以下は、温湿度・気圧センサの直近 {span_minutes:g} 分間の測定値を、'
            f'{self._bucket_seconds:g} 秒ごとに取得したものです '
            f'(古い順、各 {n} 個)。\n'
            f'温度 [C]: {fmt(temperatures)}\n'
            f'湿度 [%]: {fmt(humidities)}\n'
            f'気圧 [hPa]: {fmt(pressures)}\n'
            'この情報を元にこの先1時間以内に天候が悪化する可能性を0から100の範囲で評価してください。\n'
            'あくまで推測でかまいません。回答は数値だけを端的に返したのち、改行文字を挟んで、その後短く理由を教えてください\n'
        )

    def _publish_prompt(self) -> None:
        """履歴からプロンプトを組み立てて output_topic に publish する。"""
        # 履歴を種別ごとのリストに分解
        temps = [b[0] for b in self._history]
        hums = [b[1] for b in self._history]
        press = [b[2] for b in self._history]

        prompt = self.build_prompt(temps, hums, press)

        msg = String()
        msg.data = prompt
        self._prompt_pub.publish(msg)

        self.get_logger().info(
            f"プロンプトを '{self._prompt_pub.topic_name}' に publish しました "
            f'({len(prompt)} 文字)。'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PromptMakerForBme280()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
