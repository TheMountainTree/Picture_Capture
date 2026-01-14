import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/themountaintree/workspace/ros2_ws/install/get_picture'
