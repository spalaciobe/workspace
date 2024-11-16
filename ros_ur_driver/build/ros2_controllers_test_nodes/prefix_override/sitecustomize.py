import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sebas/workspace/ros_ur_driver/install/ros2_controllers_test_nodes'
