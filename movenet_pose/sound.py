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

        # パラメータ
        self.declare_parameter('joint_id', 0)            # 参照する関節ID（MoveNet 出力に合わせる）
        self.declare_parameter('max_dist', 5.0)          # X軸の最大距離（m）
        self.declare_parameter('beep_freq', 440.0)       # ビープ音の周波数（Hz）
        self.declare_parameter('beep_duration', 1.0)     # バッファ長（秒）
        self.declare_parameter('sample_rate', 44100)     # サンプリングレート

        self.joint_id     = self.get_parameter('joint_id').get_parameter_value().integer_value
        self.max_dist     = abs(self.get_parameter('max_dist').get_parameter_value().double_value)
        freq              = self.get_parameter('beep_freq').get_parameter_value().double_value
        duration          = self.get_parameter('beep_duration').get_parameter_value().double_value
        sample_rate       = self.get_parameter('sample_rate').get_parameter_value().integer_value

        # pygame ミキサー初期化（ステレオ）
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2)
        # ビープ音バッファ生成
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        sine_wave = 0.5 * np.sin(2.0 * np.pi * freq * t)   # 振幅 0.5
        # int16 にスケーリング
        audio = np.int16(sine_wave * 32767)
        # ステレオ用にチャンネル複製
        stereo = np.column_stack((audio, audio))
        # Sound オブジェクト作成
        self.beep = sndarray.make_sound(stereo)
        # ループ再生
        self.channel = pygame.mixer.Channel(0)
        self.channel.play(self.beep, loops=-1)

        # skeleton_markers トピック購読
        self.sub = self.create_subscription(
            MarkerArray,
            'skeleton_markers',
            self.on_markers,
            10)

        self.get_logger().info('BeepPanNode started.')

    def on_markers(self, msg: MarkerArray):
        # joint_id の Marker を探す
        x = None
        for m in msg.markers:
            if m.id == self.joint_id:
                x = m.pose.position.x
                break
        if x is None:
            return

        # pan を計算：[-max_dist, +max_dist] → [0,1]
        pan = (x + self.max_dist) / (2.0 * self.max_dist)
        pan = max(0.0, min(1.0, pan))

        # 左右音量
        left_vol  = 1.0 - pan
        right_vol = pan

        # ボリューム更新
        self.channel.set_volume(left_vol, right_vol)

        self.get_logger().debug(
            f'x={x:.2f} m, pan={pan:.2f}, vol(L,R)=({left_vol:.2f},{right_vol:.2f})'
        )

def main(args=None):
    rclpy.init(args=args)
    node = BeepPanNode()
    try:
        rclpy.spin(node)
    finally:
        pygame.mixer.quit()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
