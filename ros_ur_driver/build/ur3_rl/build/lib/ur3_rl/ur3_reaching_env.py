# import time
# import numpy as np
# import gymnasium as gym

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSPresetProfiles
# import sensor_msgs.msg
# import geometry_msgs.msg

# from std_srvs.srv import SetBool
# from ur_msgs.srv import SetIO

# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# from control_msgs.action import FollowJointTrajectory
# from rclpy.action import ActionClient

# class ReachingUR3(gym.Env):
    
#     def __init__(self, control_space="joint"):
#         print(f"1. Starting initialization with control_space: {control_space}")
#         self.control_space = control_space  # joint or cartesian

#         print("2. Setting up spaces")
#         # spaces
#         self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(25,), dtype=np.float32)
#         if self.control_space == "joint":
#             self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
#         elif self.control_space == "cartesian":
#             self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
#         else:
#             raise ValueError("Invalid control space:", self.control_space)

#         print("3. Initializing ROS node")
#         # initialize the ROS node
#         rclpy.init()
#         self.node = Node(self.__class__.__name__)

#         print("4. Starting spin thread")
#         import threading
#         threading.Thread(target=self._spin).start()

#         print("5. Creating publishers")
#         # create publishers
#         self.pub_command_joint = self.node.create_publisher(
#             JointTrajectory, 
#             '/scaled_joint_trajectory_controller/joint_trajectory',
#             10
#         )

#         print("6. Setting up robot state")
#         # keep compatibility with UR3 robot state
#         self.robot_state = {"joint_position": np.zeros((6,)),
#                             "joint_velocity": np.zeros((6,)),
#                             "cartesian_position": np.zeros((3,))}

#         print("7. Creating subscribers")
#         # create subscribers
#         self.node.create_subscription(sensor_msgs.msg.JointState, '/joint_states', self._callback_joint_states, QoSPresetProfiles.SYSTEM_DEFAULT.value)
#         self.node.create_subscription(geometry_msgs.msg.Pose, '/pose', self._callback_end_effector_pose, QoSPresetProfiles.SYSTEM_DEFAULT.value)

#         # print("8. Setting up service clients")
#         # # service clients
#         # self.client_set_io = self.node.create_client(SetIO, '/ur_hardware_interface/set_io')
#         # self.client_set_io.wait_for_service()
#         print("8. Setting up service clients")
#         # service clients
#         self.client_set_io = self.node.create_client(SetIO, '/ur_hardware_interface/set_io')
#         try:
#             print("Waiting for set_io service (timeout: 5 seconds)...")
#             if not self.client_set_io.wait_for_service(timeout_sec=5.0):
#                 print("Warning: set_io service not available, continuing without it")
#                 self.client_set_io = None
#             else:
#                 print("set_io service found")
#         except Exception as e:
#             print(f"Warning: Error waiting for set_io service: {e}")
#             print("Continuing without set_io service")
#             self.client_set_io = None


#         print("9. Setting up remaining variables")
#         self.dt = 1 / 120.0
#         self.action_scale = 2.5
#         self.dof_vel_scale = 0.1
#         self.max_episode_length = 100
#         self.robot_dof_speed_scales = 0.5
#         self.target_pos = np.array([0.65, 0.2, 0.2])
#         self.robot_default_dof_pos = np.radians([0, -90, 0, -90, 0, 0])
#         self.robot_dof_lower_limits = np.array([0, 0, 0, 0, 0, 0])
#         self.robot_dof_upper_limits = np.array([1, 1, 1, 1, 1, 1])

#         self.progress_buf = 1
#         self.obs_buf = np.zeros((18,), dtype=np.float32)

#         print("10. Initialization complete")    

#     def _spin(self):
#         rclpy.spin(self.node)

#     def _callback_joint_states(self, msg):
#         self.robot_state["joint_position"] = np.array(msg.position)
#         self.robot_state["joint_velocity"] = np.array(msg.velocity)

#     def _callback_end_effector_pose(self, msg):
#         position = msg.position
#         self.robot_state["cartesian_position"] = np.array([position.x, position.y, position.z])

#     def _get_observation_reward_done(self):
#         # observation
#         robot_dof_pos = self.robot_state["joint_position"]
#         robot_dof_vel = self.robot_state["joint_velocity"]
#         end_effector_pos = self.robot_state["cartesian_position"]

#         dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
#         dof_vel_scaled = robot_dof_vel * self.dof_vel_scale

#         # Extended observation space to match the trained model (25 dimensions)
#         self.obs_buf = np.zeros(25, dtype=np.float32)
        
#         # Fill the observation buffer (25 dimensions total)
#         self.obs_buf[0] = self.progress_buf / float(self.max_episode_length)  # Progress (1)
#         self.obs_buf[1:7] = dof_pos_scaled                                    # Joint positions (6)
#         self.obs_buf[7:13] = dof_vel_scaled                                  # Joint velocities (6)
#         self.obs_buf[13:16] = end_effector_pos                               # End effector pos (3)
#         self.obs_buf[16:19] = self.target_pos                                # Target pos (3)
#         self.obs_buf[19:22] = end_effector_pos - self.target_pos            # Distance to target (3)
#         self.obs_buf[22:25] = end_effector_pos                              # Duplicate end effector pos (3)

#         # reward
#         distance = np.linalg.norm(end_effector_pos - self.target_pos)
#         reward = -distance

#         # done
#         done = self.progress_buf >= self.max_episode_length - 1
#         done = done or distance <= 0.075

#         print("Distance:", distance)
#         if done:
#             print("Target or Maximum episode length reached")
#             time.sleep(1)

#         return self.obs_buf, reward, done


#     def reset(self):
#         print("Reseting...")

#         # go to 1) safe position, 2) random position
#         msg = JointTrajectory()
#         msg.joint_names = [
#             'shoulder_pan_joint', 
#             'shoulder_lift_joint', 
#             'elbow_joint',
#             'wrist_1_joint', 
#             'wrist_2_joint', 
#             'wrist_3_joint'
#         ]
        
#         # Safe position
#         point = JointTrajectoryPoint()
#         point.positions = self.robot_default_dof_pos.tolist()
#         point.velocities = [0.0] * 6
#         point.time_from_start.sec = 3
#         point.time_from_start.nanosec = 0
        
#         msg.points = [point]
#         print("Publishing safe position:", point.positions)
#         self.pub_command_joint.publish(msg)
#         time.sleep(3)

#         # Random position
#         point.positions = (self.robot_default_dof_pos + 0.25 * (np.random.rand(6) - 0.5)).tolist()
#         point.time_from_start.sec = 1
#         point.time_from_start.nanosec = 0
#         msg.points = [point]
#         print("Publishing random position:", point.positions)
#         self.pub_command_joint.publish(msg)
#         time.sleep(1)

#         # get target position from prompt
#         while True:
#             try:
#                 print("Enter target position (X, Y, Z) in meters")
#                 raw = input("or press [Enter] key for a random target position: ")
#                 if raw:
#                     self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
#                 else:
#                     noise = (2 * np.random.rand(3) - 1) * np.array([0.1, 0.2, 0.2])
#                     self.target_pos = np.array([0.6, 0.0, 0.4]) + noise
#                 print("Target position:", self.target_pos)
#                 break
#             except ValueError:
#                 print("Invalid input. Try something like: 0.65, 0.0, 0.4")

#         input("Press [Enter] to continue")

#         self.progress_buf = 0
#         observation, reward, done = self._get_observation_reward_done()

#         return observation, {}
    
#     def step(self, action):
#         print("\nStep called with action:", action)
    
#         self.progress_buf += 1

#         if self.control_space == "joint":
#             # Calculate new joint positions
#             joint_positions = self.robot_state["joint_position"] + (self.robot_dof_speed_scales * self.dt * action * self.action_scale * 0.25)
#             joint_positions = np.clip(joint_positions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
#             print("Publishing joint positions:", joint_positions)
        
#             # Create and publish trajectory message
#             msg = JointTrajectory()
#             msg.joint_names = [
#                 'shoulder_pan_joint', 
#                 'shoulder_lift_joint', 
#                 'elbow_joint',
#                 'wrist_1_joint', 
#                 'wrist_2_joint', 
#                 'wrist_3_joint'
#             ]
            
#             point = JointTrajectoryPoint()
#             point.positions = joint_positions.tolist()
#             point.velocities = [0.0] * 6
#             point.time_from_start.sec = 0
#             point.time_from_start.nanosec = int(self.dt * 1e9)
            
#             msg.points = [point]
#             print("Publishing message:", msg)
        
#             self.pub_command_joint.publish(msg)
#             print("Message published")  # Debug print

#         # cartesian
#         elif self.control_space == "cartesian":
#             end_effector_pos = self.robot_state["cartesian_position"] + action / 100.0
#             msg = geometry_msgs.msg.Pose()
#             msg.position.x = end_effector_pos[0]
#             msg.position.y = end_effector_pos[1]
#             msg.position.z = end_effector_pos[2]
#             msg.orientation.x = np.nan
#             msg.orientation.y = np.nan
#             msg.orientation.z = np.nan
#             msg.orientation.w = np.nan
#             self.pub_command_cartesian.publish(msg)

#         # the use of time.sleep is for simplicity. It does not guarantee control at a specific frequency
#         time.sleep(1 / 10.0)

#         observation, reward, terminated = self._get_observation_reward_done()

#         return observation, reward, terminated, False, {}
    
#     def render(self, *args, **kwargs):
#         pass

#     def close(self):
#         # shutdown the node
#         self.node.destroy_node()
#         rclpy.shutdown()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import argparse
import torch
import os
from skrl.utils.runner.torch import Runner
from skrl.models.torch import Model, GaussianMixin
import torch.nn as nn
from ur3_rl.ur3_reaching_env import ReachingUR3
from skrl.envs.wrappers.torch import wrap_env

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.policy_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.net_container(inputs["states"])
        return self.policy_layer(x), self.log_std_parameter, {}

class RosSkrlPlayer(Node):
    def __init__(self, checkpoint_path, control_space="joint"):
        # Initialize ROS node first
        super().__init__('ros_skrl_player')
        
        self.get_logger().info("Initializing ROS SkRL player...")
        
        # Create environment
        # Note: ReachingUR3 will initialize its own ROS node, but that's okay
        # since it will run in a separate thread
        self.env = ReachingUR3(control_space=control_space)
        self.env = wrap_env(self.env)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        # Setup agent and load checkpoint
        self.setup_agent(checkpoint_path)
        
        # Create a timer for periodic execution
        self.timer = self.create_timer(0.1, self.step_callback)  # 10Hz
        
        self.get_logger().info("ROS SkRL player initialized")

    def setup_agent(self, checkpoint_path):
        # Configure experiment
        experiment_cfg = {
            "agent": {
                "experiment": {
                    "write_interval": 0,
                    "checkpoint_interval": 0
                }
            },
            "trainer": {
                "close_environment_at_exit": False
            }
        }

        # Create models dictionary
        models = {}
        models["policy"] = Policy(
            self.env.observation_space,
            self.env.action_space,
            self.device,
            clip_actions=True
        )

        # Load checkpoint
        self.get_logger().info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "policy" in checkpoint:
            models["policy"].load_state_dict(checkpoint["policy"])
        else:
            models["policy"].load_state_dict(checkpoint)

        # Setup runner
        self.runner = Runner(self.env, experiment_cfg)
        self.runner.set_agent_models(models)
        
        # Set evaluation mode
        self.runner.agent.set_running_mode("eval")
        
        # Get initial observation
        self.current_obs, _ = self.env.reset()

    def step_callback(self):
        with torch.inference_mode():
            # Get action from agent
            actions = self.runner.agent.act(self.current_obs, timestep=0, timesteps=0)[0]
            
            # Execute action and get new observation
            self.current_obs, reward, terminated, truncated, info = self.env.step(actions)
            
            if terminated or truncated:
                self.current_obs, _ = self.env.reset()

    def destroy_node(self):
        # Make sure to close the environment properly
        if hasattr(self, 'env'):
            self.env.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description="Play a SkRL agent with ROS")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--control_space", type=str, default="joint", choices=["joint", "cartesian"],
                        help="Control space to use (joint or cartesian)")
    
    args_cli = parser.parse_args()
    
    try:
        player = RosSkrlPlayer(args_cli.checkpoint, args_cli.control_space)
        rclpy.spin(player)
    except KeyboardInterrupt:
        pass
    finally:
        if 'player' in locals():
            player.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()