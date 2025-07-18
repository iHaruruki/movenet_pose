import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class MoveNetNode(Node):
    def __init__(self):
        super().__init__('movenet_node')
        self.br = CvBridge()

        # Declare and read parameters
        self.declare_parameter('model_url', 'https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('score_threshold', 0.3)

        model_url = self.get_parameter('model_url').get_parameter_value().string_value
        self.get_logger().info(f"Loading MoveNet model from: {model_url}")
        loaded = hub.load(model_url)
        self.infer = loaded.signatures['serving_default']

        # Publisher for skeleton markers
        self.marker_pub = self.create_publisher(MarkerArray, 'skeleton_markers', 10)

        # Subscribers for synchronized RGB and Depth
        sub_color = Subscriber(self, Image, '/camera/color/image_raw')
        sub_cinfo = Subscriber(self, CameraInfo, '/camera/color/camera_info')
        sub_depth = Subscriber(self, Image, '/camera/depth/image_raw')
        sub_dinfo = Subscriber(self, CameraInfo, '/camera/depth/camera_info')

        ats = ApproximateTimeSynchronizer(
            [sub_color, sub_cinfo, sub_depth, sub_dinfo],
            queue_size=10,
            slop=self.get_parameter('sync_slop').get_parameter_value().double_value
        )
        ats.registerCallback(self.callback)

        # Define skeleton connections (COCO/MoveNet ordering)
        self.connections = [
            (0,1),  (1,3),  # nose→left eye→left ear
            (0,2),  (2,4),  # nose→right eye→right ear
            (5,6),            # left shoulder→right shoulder
            (5,7),  (7,9),  # left shoulder→elbow→wrist
            (6,8),  (8,10), # right shoulder→elbow→wrist
            (5,11), (6,12), # shoulders→hips
            (11,12),         # left hip→right hip
            (11,13), (13,15), # left hip→knee→ankle
            (12,14), (14,16)  # right hip→knee→ankle
        ]

    def preprocess(self, img: np.ndarray) -> tf.Tensor:
        img = tf.expand_dims(img, axis=0)
        img = tf.cast(tf.image.resize_with_pad(img, 192, 192), dtype=tf.int32)
        return img

    def callback(self, img_msg, cinfo_msg, depth_msg, dinfo_msg):
        # Convert images
        color = self.br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        depth = self.br.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # Run MoveNet
        input_tensor = self.preprocess(color)
        outputs = self.infer(input=input_tensor)
        kpts = outputs['output_0'].numpy()[0, 0, :, :]  # (17,3)

        # Intrinsics & thresholds
        k = cinfo_msg.k
        fx, fy = k[0], k[4]
        cx, cy = k[2], k[5]
        depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        score_th = self.get_parameter('score_threshold').get_parameter_value().double_value

        markers = MarkerArray()
        positions = {}  # store 3D positions of visible keypoints

        # --- Sphere markers for each keypoint ---
        for i, (y_n, x_n, score) in enumerate(kpts):
            if score < score_th:
                continue

            u = int(x_n * img_msg.width)
            v = int(y_n * img_msg.height)
            if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
                continue

            z = float(depth[v, u]) * depth_scale
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            positions[i] = (x, y, z)

            m = Marker()
            m.header = img_msg.header
            m.ns = 'skeleton'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.a = 1.0
            m.color.r = 1.0
            markers.markers.append(m)

        # --- Line marker connecting the keypoints ---
        line_marker = Marker()
        line_marker.header = img_msg.header
        line_marker.ns = 'skeleton_lines'
        line_marker.id = 100  # unique ID
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.02  # line width
        line_marker.color.a = 1.0
        line_marker.color.g = 1.0  # green lines

        for a, b in self.connections:
            if a in positions and b in positions:
                # geometry_msgs Point must be constructed without positional args
                p1 = Point()
                p1.x, p1.y, p1.z = positions[a]
                p2 = Point()
                p2.x, p2.y, p2.z = positions[b]
                line_marker.points.extend([p1, p2])
        

        markers.markers.append(line_marker)

        # Publish both spheres and lines
        self.marker_pub.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = MoveNetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
