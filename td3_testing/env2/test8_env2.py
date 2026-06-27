import time
import warnings
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_test8_env2 import GazeboEnv

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    # Function to get the action from the actor
    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    # Load actor weights
    def load(self, file_name, directory): 
        actor_weights = torch.load(
            f"{directory}/{file_name}_actor.pth",
            map_location=torch.device('cpu'),
            weights_only=True
        )
        self.actor.load_state_dict(actor_weights)

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu

num_robots = 8  # Define the number of robots

file_names = ["TD3_velodyne_0", "TD3_velodyne_0", "TD3_velodyne_0", "TD3_velodyne_0", "TD3_velodyne_0", "TD3_velodyne_0", "TD3_velodyne_0", "TD3_velodyne_0"]
file_names2 = ["TD3_velodyne_1", "TD3_velodyne_1", "TD3_velodyne_1", "TD3_velodyne_1", "TD3_velodyne_1", "TD3_velodyne_1", "TD3_velodyne_1", "TD3_velodyne_1"]
timestep = [0] * num_robots
episode_timesteps = [0] * num_robots
done = [True] * num_robots
flag_done = [False] * num_robots

# Create the training environment
seed = 1234  # Random seed number
max_ep = 1000  # maximum number of steps per episode
environment_dim = 20
robot_dim = 6
env = GazeboEnv("env2_8robots.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1
lr_act = 1e-4
lr_cr = 1e-4

network = [TD3(state_dim, action_dim, max_action) for _ in range(num_robots)]
network2 = [TD3(state_dim, action_dim, max_action) for _ in range(num_robots)]

try:
    for i in range(num_robots):
        network[i].load(file_names[i], "./pytorch_models/no-recl")
        network2[i].load(file_names2[i], "./pytorch_models/recl")
except:
    raise ValueError("Could not load the stored model parameters")

states = env.reset()
next_states = [None] * num_robots
done = [False] * num_robots
target = [False] * num_robots

# Begin the testing loop
while True:
    a_in = []  # Initialize list to store actions for all robots

    for i in range(num_robots):
        if not done[i] and not flag_done[i]:
            if np.any(env.moving_objects[i]) and min(env.laserscan_data[i]) < 2.5:
                action = network[i].get_action(np.array(states[i]))
                a_in.append([((action[0] + max_action) / 2) * 0.5, action[1]])
            else:
                action = network2[i].get_action(np.array(states[i]))
                a_in.append([((action[0] + max_action) / 2) * 0.5, action[1] * 0.5])

        else:
            a_in.append([0, 0])  # Stop robot when done
            flag_done[i] = True  # Set flag when robot is done
            
    # Step the environment with the current actions dynamically
    for i in range(num_robots):
        next_states[i], done[i], target[i] = env.step(a_in[i], i)

    # Reset the environment if all robots are done
    if all(flag_done):
        states = env.reset()
        
        # Reset all flags and timesteps
        done = [False] * num_robots
        flag_done = [False] * num_robots
        episode_timesteps = [0] * num_robots

    else:
        # Update states and timesteps only for active robots
        for i in range(num_robots):
            if not done[i]:
                states[i] = next_states[i]
                episode_timesteps[i] += 1

    # End the loop if any robot reaches max timesteps
    if any(t >= max_ep for t in episode_timesteps):
        states = env.reset()
        
        # Reset all flags and timesteps
        done = [False] * num_robots
        flag_done = [False] * num_robots
        episode_timesteps = [0] * num_robots