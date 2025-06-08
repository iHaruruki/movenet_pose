from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='movenet_pose',
            executable='movenet_node',
            name='movenet_node',
            output='screen',
            parameters=[
                {'model_path': '/path/to/movenet_saved_model'},
                {'depth_scale': 0.001},
                {'sync_slop': 0.1},
                {'score_threshold': 0.3},
            ]
        ),
    ])
