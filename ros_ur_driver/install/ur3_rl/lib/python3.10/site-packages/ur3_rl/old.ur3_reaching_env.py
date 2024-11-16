import time
import numpy as np
import gymnasium as gym

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
import sensor_msgs.msg
import geometry_msgs.msg

from std_srvs.srv import SetBool
from ur_msgs.srv import SetIO

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

class ReachingUR3(gym.Env):
    # def __init__(self, control_space="joint"):
        
    #     self.control_space = control_space  # joint or cartesian
    #     print(f"Initializing ReachingUR3 with control_space: {control_space}")

    #     # spaces
    #     self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(18,), dtype=np.float32)
    #     if self.control_space == "joint":
    #         self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
    #     elif self.control_space == "cartesian":
    #         self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    #     else:
    #         raise ValueError("Invalid control space:", self.control_space)

    #     # initialize the ROS node
    #     rclpy.init()
    #     self.node = Node(self.__class__.__name__)

    #     import threading
    #     threading.Thread(target=self._spin).start()

    #     # create publishers
    #     # self.pub_command_joint = self.node.create_publisher(sensor_msgs.msg.JointState, '/joint_states', QoSPresetProfiles.SYSTEM_DEFAULT.value)
    #     self.pub_command_joint = self.node.create_publisher(
    #         JointTrajectory, 
    #         '/scaled_joint_trajectory_controller/joint_trajectory',  
    #         QoSPresetProfiles.SYSTEM_DEFAULT.value
    #     )
        
    #     self.pub_command_cartesian = self.node.create_publisher(geometry_msgs.msg.Pose, '/pose', QoSPresetProfiles.SYSTEM_DEFAULT.value)

    #     # keep compatibility with UR3 robot state
    #     self.robot_state = {"joint_position": np.zeros((6,)),
    #                         "joint_velocity": np.zeros((6,)),
    #                         "cartesian_position": np.zeros((3,))}

    #     # create subscribers
    #     self.node.create_subscription(sensor_msgs.msg.JointState, '/joint_states', self._callback_joint_states, QoSPresetProfiles.SYSTEM_DEFAULT.value)
    #     self.node.create_subscription(geometry_msgs.msg.Pose, '/pose', self._callback_end_effector_pose, QoSPresetProfiles.SYSTEM_DEFAULT.value)

    #     # service clients
    #     self.client_set_io = self.node.create_client(SetIO, '/ur_hardware_interface/set_io')
    #     self.client_set_io.wait_for_service()

    #     print("Robot connected")

    #     self.dt = 1 / 120.0
    #     self.action_scale = 2.5
    #     self.dof_vel_scale = 0.1
    #     self.max_episode_length = 100
    #     self.robot_dof_speed_scales = 1
    #     self.target_pos = np.array([0.65, 0.2, 0.2])
    #     self.robot_default_dof_pos = np.radians([0, -90, 0, -90, 0, 0])
    #     self.robot_dof_lower_limits = np.array([-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944])
    #     self.robot_dof_upper_limits = np.array([ 2.9671,  2.0944,  2.9671,  2.0944,  2.9671,  2.0944])

    #     self.progress_buf = 1
    #     self.obs_buf = np.zeros((18,), dtype=np.float32)

    #     # Update publisher
    #     self.pub_command_joint = self.node.create_publisher(
    #         JointTrajectory, 
    #         '/scaled_joint_trajectory_controller/joint_trajectory',
    #         10  # QoS profile depth
    #     )

    #     print("Publisher created:", self.pub_command_joint)
    
    def __init__(self, control_space="joint"):
        print(f"1. Starting initialization with control_space: {control_space}")
        self.control_space = control_space  # joint or cartesian

        print("2. Setting up spaces")
        # spaces
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(25,), dtype=np.float32)
        if self.control_space == "joint":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        elif self.control_space == "cartesian":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        else:
            raise ValueError("Invalid control space:", self.control_space)

        print("3. Initializing ROS node")
        # initialize the ROS node
        rclpy.init()
        self.node = Node(self.__class__.__name__)

        print("4. Starting spin thread")
        import threading
        threading.Thread(target=self._spin).start()

        print("5. Creating publishers")
        # create publishers
        self.pub_command_joint = self.node.create_publisher(
            JointTrajectory, 
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        print("6. Setting up robot state")
        # keep compatibility with UR3 robot state
        self.robot_state = {"joint_position": np.zeros((6,)),
                            "joint_velocity": np.zeros((6,)),
                            "cartesian_position": np.zeros((3,))}

        print("7. Creating subscribers")
        # create subscribers
        self.node.create_subscription(sensor_msgs.msg.JointState, '/joint_states', self._callback_joint_states, QoSPresetProfiles.SYSTEM_DEFAULT.value)
        self.node.create_subscription(geometry_msgs.msg.Pose, '/pose', self._callback_end_effector_pose, QoSPresetProfiles.SYSTEM_DEFAULT.value)

        # print("8. Setting up service clients")
        # # service clients
        # self.client_set_io = self.node.create_client(SetIO, '/ur_hardware_interface/set_io')
        # self.client_set_io.wait_for_service()
        print("8. Setting up service clients")
        # service clients
        self.client_set_io = self.node.create_client(SetIO, '/ur_hardware_interface/set_io')
        try:
            print("Waiting for set_io service (timeout: 5 seconds)...")
            if not self.client_set_io.wait_for_service(timeout_sec=5.0):
                print("Warning: set_io service not available, continuing without it")
                self.client_set_io = None
            else:
                print("set_io service found")
        except Exception as e:
            print(f"Warning: Error waiting for set_io service: {e}")
            print("Continuing without set_io service")
            self.client_set_io = None


        print("9. Setting up remaining variables")
        self.dt = 1 / 120.0
        self.action_scale = 2.5
        self.dof_vel_scale = 0.1
        self.max_episode_length = 100
        self.robot_dof_speed_scales = 1
        self.target_pos = np.array([0.65, 0.2, 0.2])
        self.robot_default_dof_pos = np.radians([0, -90, 0, -90, 0, 0])
        self.robot_dof_lower_limits = np.array([-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944])
        self.robot_dof_upper_limits = np.array([ 2.9671,  2.0944,  2.9671,  2.0944,  2.9671,  2.0944])

        self.progress_buf = 1
        self.obs_buf = np.zeros((18,), dtype=np.float32)

        print("10. Initialization complete")    

    def _spin(self):
        rclpy.spin(self.node)

    def _callback_joint_states(self, msg):
        self.robot_state["joint_position"] = np.array(msg.position)
        self.robot_state["joint_velocity"] = np.array(msg.velocity)

    def _callback_end_effector_pose(self, msg):
        position = msg.position
        self.robot_state["cartesian_position"] = np.array([position.x, position.y, position.z])

    def _get_observation_reward_done(self):
        # observation
        robot_dof_pos = self.robot_state["joint_position"]
        robot_dof_vel = self.robot_state["joint_velocity"]
        end_effector_pos = self.robot_state["cartesian_position"]

        dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        dof_vel_scaled = robot_dof_vel * self.dof_vel_scale

        # Extended observation space to match the trained model (25 dimensions)
        self.obs_buf = np.zeros(25, dtype=np.float32)
        
        # Fill the observation buffer (25 dimensions total)
        self.obs_buf[0] = self.progress_buf / float(self.max_episode_length)  # Progress (1)
        self.obs_buf[1:7] = dof_pos_scaled                                    # Joint positions (6)
        self.obs_buf[7:13] = dof_vel_scaled                                  # Joint velocities (6)
        self.obs_buf[13:16] = end_effector_pos                               # End effector pos (3)
        self.obs_buf[16:19] = self.target_pos                                # Target pos (3)
        self.obs_buf[19:22] = end_effector_pos - self.target_pos            # Distance to target (3)
        self.obs_buf[22:25] = end_effector_pos                              # Duplicate end effector pos (3)

        # reward
        distance = np.linalg.norm(end_effector_pos - self.target_pos)
        reward = -distance

        # done
        done = self.progress_buf >= self.max_episode_length - 1
        done = done or distance <= 0.075

        print("Distance:", distance)
        if done:
            print("Target or Maximum episode length reached")
            time.sleep(1)

        return self.obs_buf, reward, done

    # def reset(self):
    #     print("Reseting...")

    #     # go to 1) safe position, 2) random position
    #     msg = sensor_msgs.msg.JointState()
    #     msg.position = self.robot_default_dof_pos.tolist()
    #     self.pub_command_joint.publish(msg)
    #     time.sleep(3)
    #     msg.position = (self.robot_default_dof_pos + 0.25 * (np.random.rand(6) - 0.5)).tolist()
    #     self.pub_command_joint.publish(msg)
    #     time.sleep(1)

    #     # get target position from prompt
    #     while True:
    #         try:
    #             print("Enter target position (X, Y, Z) in meters")
    #             raw = input("or press [Enter] key for a random target position: ")
    #             if raw:
    #                 self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
    #             else:
    #                 noise = (2 * np.random.rand(3) - 1) * np.array([0.1, 0.2, 0.2])
    #                 self.target_pos = np.array([0.6, 0.0, 0.4]) + noise
    #             print("Target position:", self.target_pos)
    #             break
    #         except ValueError:
    #             print("Invalid input. Try something like: 0.65, 0.0, 0.4")

    #     input("Press [Enter] to continue")

    #     self.progress_buf = 0
    #     observation, reward, done = self._get_observation_reward_done()

    #     return observation, {}

    def reset(self):
        print("Reseting...")

        # go to 1) safe position, 2) random position
        msg = JointTrajectory()
        msg.joint_names = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]
        
        # Safe position
        point = JointTrajectoryPoint()
        point.positions = self.robot_default_dof_pos.tolist()
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 3
        point.time_from_start.nanosec = 0
        
        msg.points = [point]
        print("Publishing safe position:", point.positions)
        self.pub_command_joint.publish(msg)
        time.sleep(3)

        # Random position
        point.positions = (self.robot_default_dof_pos + 0.25 * (np.random.rand(6) - 0.5)).tolist()
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0
        msg.points = [point]
        print("Publishing random position:", point.positions)
        self.pub_command_joint.publish(msg)
        time.sleep(1)

        # get target position from prompt
        while True:
            try:
                print("Enter target position (X, Y, Z) in meters")
                raw = input("or press [Enter] key for a random target position: ")
                if raw:
                    self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
                else:
                    noise = (2 * np.random.rand(3) - 1) * np.array([0.1, 0.2, 0.2])
                    self.target_pos = np.array([0.6, 0.0, 0.4]) + noise
                print("Target position:", self.target_pos)
                break
            except ValueError:
                print("Invalid input. Try something like: 0.65, 0.0, 0.4")

        input("Press [Enter] to continue")

        self.progress_buf = 0
        observation, reward, done = self._get_observation_reward_done()

        return observation, {}
    
    # def step(self, action):
    #     self.progress_buf += 1

    #     # control space
    #     # joint
    #     if self.control_space == "joint":
    #         joint_positions = self.robot_state["joint_position"] + (self.robot_dof_speed_scales * self.dt * action * self.action_scale)
    #         msg = sensor_msgs.msg.JointState()
    #         msg.position = joint_positions.tolist()
    #         self.pub_command_joint.publish(msg)
    #     # cartesian
    #     elif self.control_space == "cartesian":
    #         end_effector_pos = self.robot_state["cartesian_position"] + action / 100.0
    #         msg = geometry_msgs.msg.Pose()
    #         msg.position.x = end_effector_pos[0]
    #         msg.position.y = end_effector_pos[1]
    #         msg.position.z = end_effector_pos[2]
    #         msg.orientation.x = np.nan
    #         msg.orientation.y = np.nan
    #         msg.orientation.z = np.nan
    #         msg.orientation.w = np.nan
    #         self.pub_command_cartesian.publish(msg)

    #     # the use of time.sleep is for simplicity. It does not guarantee control at a specific frequency
    #     time.sleep(1 / 30.0)

    #     observation, reward, terminated = self._get_observation_reward_done()

    #     return observation, reward, terminated, False, {}

    def step(self, action):
        print("\nStep called with action:", action)
    
        self.progress_buf += 1

        if self.control_space == "joint":
            # Calculate new joint positions
            joint_positions = self.robot_state["joint_position"] + (self.robot_dof_speed_scales * self.dt * action * self.action_scale)
            print("Publishing joint positions:", joint_positions)
        
            # Create and publish trajectory message
            msg = JointTrajectory()
            msg.joint_names = [
                'shoulder_pan_joint', 
                'shoulder_lift_joint', 
                'elbow_joint',
                'wrist_1_joint', 
                'wrist_2_joint', 
                'wrist_3_joint'
            ]
            
            point = JointTrajectoryPoint()
            point.positions = joint_positions.tolist()
            point.velocities = [0.0] * 6
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(self.dt * 1e9)
            
            msg.points = [point]
            print("Publishing message:", msg)
        
            self.pub_command_joint.publish(msg)
            print("Message published")  # Debug print

        # cartesian
        elif self.control_space == "cartesian":
            end_effector_pos = self.robot_state["cartesian_position"] + action / 100.0
            msg = geometry_msgs.msg.Pose()
            msg.position.x = end_effector_pos[0]
            msg.position.y = end_effector_pos[1]
            msg.position.z = end_effector_pos[2]
            msg.orientation.x = np.nan
            msg.orientation.y = np.nan
            msg.orientation.z = np.nan
            msg.orientation.w = np.nan
            self.pub_command_cartesian.publish(msg)

        # the use of time.sleep is for simplicity. It does not guarantee control at a specific frequency
        time.sleep(1 / 30.0)

        observation, reward, terminated = self._get_observation_reward_done()

        return observation, reward, terminated, False, {}
    
    def render(self, *args, **kwargs):
        pass

    def close(self):
        # shutdown the node
        self.node.destroy_node()
        rclpy.shutdown()
