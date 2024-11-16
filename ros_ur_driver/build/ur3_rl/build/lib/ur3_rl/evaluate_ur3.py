import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
# from skrl.envs.torch import wrap_env
from skrl.envs.wrappers.torch import wrap_env

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net_container = nn.Sequential(
            nn.Linear(self.num_observations, 64),  # 25 -> 64
            nn.ReLU(),
            nn.Linear(64, 64),                    # 64 -> 64
            nn.ReLU()
        )

        self.policy_layer = nn.Linear(64, self.num_actions)  # 64 -> num_actions
        self.value_layer = nn.Linear(64, 1)                  # 64 -> 1

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.net_container(inputs["states"])
        if role == "policy":
            return self.policy_layer(x), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(x), {}

# Load the environment according to the ROS version
def get_active_ros_version():
    import os
    if os.environ.get("ROS_DISTRO"):
        return "ROS2" if os.environ.get("AMENT_PREFIX_PATH") else "ROS"
    return ""

active_ros_version = get_active_ros_version()

if active_ros_version == "ROS":
    from ur3_rl.ur3_reaching_env import ReachingUR3
elif active_ros_version == "ROS2":
    from ur3_rl.ur3_reaching_env import ReachingUR3
else:
    print("No active ROS version found")
    exit()

print("Starting evaluation script...")

control_space = "joint"   # joint or cartesian
print(f"Using control space: {control_space}")

env = ReachingUR3(control_space=control_space)
print("Environment created")

# wrap the environment
env = wrap_env(env)
print("Environment wrapped")

device = env.device
print(f"Using device: {device}")

# Instantiate the agent's policy.
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["experiment"]["write_interval"] = 32
cfg_ppo["experiment"]["checkpoint_interval"] = 0

# Create models dictionary
models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)

# Create agent
agent = PPO(models=models_ppo,
           memory=None,
           cfg=cfg_ppo,
           observation_space=env.observation_space,
           action_space=env.action_space,
           device=device)

# load checkpoints
if control_space == "joint":
    print("Loading checkpoint for joint control space")
    agent.load("/home/sebas/workspace/ros_ur_driver/src/ur3_rl/ur3_rl/best_agent.pt")
elif control_space == "cartesian":
    agent.load("./best_cartesian.pt")

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

print("Starting evaluation...")
trainer.eval()