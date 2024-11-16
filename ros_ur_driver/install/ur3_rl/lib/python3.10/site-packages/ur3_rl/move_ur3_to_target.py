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
        super().__init__('ros_skrl_player')
        
        self.get_logger().info("Initializing ROS SkRL player...")
        
        # Create environment with the existing node
        self.env = ReachingUR3(control_space=control_space, node=self)
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

def main(args=None):
    # Initialize ROS once
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# ros2 run ur3_rl move_ur3_to_target --checkpoint /home/sebas/workspace/ros_ur_driver/src/ur3_rl/ur3_rl/best_agent.pt --control_space joint
