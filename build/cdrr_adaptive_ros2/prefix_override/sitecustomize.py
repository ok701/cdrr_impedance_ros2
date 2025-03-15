import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/awear/ros2_ws/src/cdrr_adaptive_ros2/install/cdrr_adaptive_ros2'
