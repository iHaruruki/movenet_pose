#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
import numpy as np
import pygame
from pygame import sndarray

class BeepPanNode(Node):
    def __init__(self):
        super().__init__('beep_pan_node')

        # パラメータ定義と読み出し
        self.declare_parameter('joint_id',       0)      # 参照する関節ID
        self.declare_parameter('max_dist',       3.0)    # 距離の最大値[m]
        self.declare_parameter('min_freq',     150.0)    # 最低周波数[Hz]
        self.declare_parameter('max_freq',     880.0)    # 最高周波数[Hz]
        self.declare_parameter('beep_duration', 1.0)      # バッファ長[秒]
        self.declare_parameter('sample_rate',   44100)    # サンプリングレート[Hz]
        self.declare_parameter('gamma',         0.5)      # ガンマ補正値

        self.joint_id    = self.get_parameter('joint_id').get_parameter_value().integer_value
        self.max_dist    = abs(self.get_parameter('max_dist').get_parameter_value().double_value)
        self.min_freq    = self.get_parameter('min_freq').get_parameter_value().double_value
        self.max_freq    = self.get_parameter('max_freq').get_parameter_value().double_value
        self.duration    = self.get_parameter('beep_duration').get_parameter_value().double_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.gamma       = self.get_parameter('gamma').get_parameter_value().double_value

        # pygame ミキサー初期化（ステレオ）
        pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2)
        self.channel = pygame.mixer.Channel(0)

        # 初期気隙：中央、無音
        self.current_freq = None
        self.update_beep((self.min_freq + self.max_freq) / 2.0, 0.0, 0.0)

        # トピック購読
        self.sub = self.create_subscription(
            MarkerArray,
            'skeleton_markers',
            self.on_markers,
            10)

        self.get_logger().info('BeepPanNode started.')

    def make_sound(self, freq: float):
        """指定周波数でビープ音バッファ生成（ステレオ）"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        wave = 0.5 * np.sin(2.0 * np.pi * freq * t)  # 振幅 0.5
        audio = np.int16(wave * 32767)               # int16 にスケーリング
        stereo = np.column_stack((audio, audio))     # ステレオ化
        return sndarray.make_sound(stereo)

    def update_beep(self, freq: float, left_vol: float, right_vol: float):
        """周波数／左右音量を反映"""
        if self.current_freq == freq:
            # 周波数変化なしなら音量だけ更新
            self.channel.set_volume(left_vol, right_vol)
        else:
            # 新しいバッファを生成してループ再生
            beep = self.make_sound(freq)
            self.channel.stop()
            self.channel.play(beep, loops=-1)
            self.channel.set_volume(left_vol, right_vol)
            self.current_freq = freq

    def on_markers(self, msg: MarkerArray):
        # 指定 joint_id のマーカーを探す
        x = z = None
        for m in msg.markers:
            if m.id == self.joint_id:
                x = m.pose.position.x
                z = m.pose.position.z
                break
        if x is None or z is None:
            return

        # ステレオパン計算（ガンマ補正付き）
        pan = (x + self.max_dist) / (2.0 * self.max_dist)
        pan = max(0.0, min(1.0, pan))
        pan_adj = pan ** self.gamma
        left_vol  = 1.0 - pan_adj
        right_vol = pan_adj

        # 距離→周波数マッピング（近いほど高音）
        frac = max(0.0, min(1.0, z / self.max_dist))
        freq = self.max_freq - (self.max_freq - self.min_freq) * frac

        # ASCII バー表示
        bar_len = 20
        pos = int(pan_adj * bar_len)
        bar = '[' + '=' * pos + ' ' * (bar_len - pos) + ']'
        self.get_logger().info(
            f'Balance {bar} L={left_vol:.2f} R={right_vol:.2f}  freq={freq:.1f}Hz'
        )

        # 音更新
        self.update_beep(freq, left_vol, right_vol)

    def destroy_node(self):
        pygame.mixer.quit()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BeepPanNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
