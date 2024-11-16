#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import time

class URController(Node):
    def __init__(self):
        super().__init__('ur_test_controller')
        
        # Publisher for sending commands
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # # Subscriber for receiving target positions
        # self.target_subscription = self.create_subscription(
        #     JointState,
        #     '/joint_states',  # This topic publishes current joint states
        #     self.target_callback,
        #     10
        # )

        # Create another subscriber for target positions
        self.target_subscription = self.create_subscription(
            JointTrajectory,
            '/joint_trajectory_target',  # Custom topic for receiving targets
            self.trajectory_callback,
            10
        )

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

    def trajectory_callback(self, msg):
        """Callback for receiving new trajectory targets"""
        print(f"Received new trajectory target")
        self.trajectory_publisher.publish(msg)

    # def target_callback(self, msg):
    #     """Callback for receiving current joint states"""
    #     # You can use this to monitor the current position
    #     # print(f"Current joint positions: {msg.position}")

    # def move_to_position(self, positions, time_from_start=1.0):
    #     trajectory_msg = JointTrajectory()
    #     trajectory_msg.joint_names = self.joint_names
        
    #     point = JointTrajectoryPoint()
    #     point.positions = positions
    #     point.velocities = [0.0] * 6
    #     point.accelerations = [0.0] * 6
    #     point.time_from_start.sec = int(time_from_start)
    #     point.time_from_start.nanosec = int((time_from_start % 1) * 1e9)
        
    #     trajectory_msg.points = [point]
        
    #     print(f"Publishing to: /scaled_joint_trajectory_controller/joint_trajectory")
    #     print(f"Moving to positions: {positions}")
    #     self.trajectory_publisher.publish(trajectory_msg)

def main():
    rclpy.init()
    controller = URController()
    print("UR Controller initialized")
    print("Waiting for trajectory commands on /joint_trajectory_target")
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Shutting down...")
    
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


"""
EXAMPLE MESSAGES
ros2 topic pub /joint_trajectory_target trajectory_msgs/msg/JointTrajectory "{joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], points: [{positions: [0.0, -1.57, 1.57, 0.0, 1.57, 0.0], time_from_start: {sec: 1, nanosec: 0}}]}"

ros2 topic pub /joint_trajectory_target trajectory_msgs/msg/JointTrajectory "{joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], points: [{positions: [0.0, -1.57, 1.2, 0.0, 1.57, 0.0], time_from_start: {sec: 1, nanosec: 0}}]}"

ros2 topic pub /joint_trajectory_target trajectory_msgs/msg/JointTrajectory "{joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], points: [{positions: [0.0, -1.57, 1.2, 0.5, 1.57, 0.0], time_from_start: {sec: 1, nanosec: 0}}]}"

ros2 topic pub /joint_trajectory_target trajectory_msgs/msg/JointTrajectory "{joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], points: [{positions: [0.0, -1.1, 1.2, 0.5, 1.57, 0.0], time_from_start: {sec: 1, nanosec: 0}}]}"

ros2 topic pub /joint_trajectory_target trajectory_msgs/msg/JointTrajectory "{joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], points: [{positions: [0.0, -1.1, 1.2, 0.5, 1.57, 1.0], time_from_start: {sec: 1, nanosec: 0}}]}"
 """