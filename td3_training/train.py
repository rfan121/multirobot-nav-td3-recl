import os
import time
from datetime import datetime
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import PrioritizedReplayBuffer
from env_train import GazeboEnv

def evaluate(network_0, network_1, epoch, eval_episodes=10):
    avg_reward_0 = 0.0
    col_0 = 0
    avg_reward_1 = 0.0
    col_1 = 0
    for _ in range(eval_episodes):
        count = 0
        state_0, state_1 = env.reset()
        done_0 = False
        done_1 = False
        flag_done_0 = False  # Flag for robot 0
        flag_done_1 = False  # Flag for robot 1

        while count < eval_step + 1:
            if not flag_done_0:  # Check the flag
                if not done_0:
                    action_0 = network_0.get_action_0(np.array(state_0))
                    a_in_0 = [(action_0[0] + max_action) / 2, action_0[1]]
                    state_0, reward_0, done_0, _ = env.step_0(a_in_0)
                    avg_reward_0 += reward_0
                    if min(env.laserscan_data_0) < 0.2:
                        col_0 += 1
                else:
                    flag_done_0 = True  # Trigger the flag when done
                    a_in_0 = [0, 0]  # Stop robot 0
                    state_0, _, done_0, _ = env.step_0(a_in_0)

            if not flag_done_1:  # Check the flag
                if not done_1:
                    action_1 = network_1.get_action_1(np.array(state_1))
                    a_in_1 = [(action_1[0] + max_action) / 2, action_1[1]]
                    state_1, reward_1, done_1, _ = env.step_1(a_in_1)
                    avg_reward_1 += reward_1
                    if min(env.laserscan_data_1) < 0.2:
                        col_1 += 1
                else:
                    flag_done_1 = True  # Trigger the flag when done
                    a_in_1 = [0, 0]  # Stop robot 1
                    state_1, _, done_1, _ = env.step_1(a_in_1)

            if flag_done_0 and flag_done_1:  # Both robots are done
                break

            count += 1

    avg_reward_0 /= eval_episodes
    avg_col_0 = col_0 / eval_episodes
    avg_reward_1 /= eval_episodes
    avg_col_1 = col_1 / eval_episodes

    print("-----------------------------------------------------")
    print(f"Epochs {epoch}, Episodes {network_0.iter_count_0}")
    print(
        f"Avg Reward R0 over {eval_episodes} Eval Eps {avg_reward_0:.2f}, Collision Rate {avg_col_0:.2f}"
    )
    print(
        f"Avg Reward R1 over {eval_episodes} Eval Eps {avg_reward_1:.2f}, Collision Rate {avg_col_1:.2f}"
    )
    print("-----------------------------------------------------")

    return avg_reward_0, avg_reward_1

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, lr_act):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()
        
        self.optimizer = optim.Adam(self.parameters(), lr_act)

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lr_cr):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr_cr)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, lr_act, lr_cr):
        # Initialize the Actor network
        self.actor_0 = Actor(state_dim, action_dim, lr_act).to(device)
        self.actor_target_0 = Actor(state_dim, action_dim, lr_act).to(device)
        self.actor_target_0.load_state_dict(self.actor_0.state_dict())
        self.actor_optimizer_0 = torch.optim.Adam(self.actor_0.parameters(), lr_act)

        # Initialize the Critic networks
        self.critic_0 = Critic(state_dim, action_dim, lr_cr).to(device)
        self.critic_target_0 = Critic(state_dim, action_dim, lr_cr).to(device)
        self.critic_target_0.load_state_dict(self.critic_0.state_dict())
        self.critic_optimizer_0 = torch.optim.Adam(self.critic_0.parameters(), lr_cr)

        # Initialize the Actor network
        self.actor_1 = Actor(state_dim, action_dim, lr_act).to(device)
        self.actor_target_1 = Actor(state_dim, action_dim, lr_act).to(device)
        self.actor_target_1.load_state_dict(self.actor_1.state_dict())
        self.actor_optimizer_1 = torch.optim.Adam(self.actor_1.parameters(), lr_act)

        # Initialize the Critic networks
        self.critic_1 = Critic(state_dim, action_dim, lr_cr).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim, lr_cr).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr_cr)

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count_0 = 0
        self.iter_count_1 = 0 

    def get_action_0(self, state_0):
        # Function to get the action from the actor
        state_0 = torch.Tensor(state_0.reshape(1, -1)).to(device)
        return self.actor_0(state_0).cpu().data.numpy().flatten()

    def get_action_1(self, state_1):
        # Function to get the action from the actor
        state_1 = torch.Tensor(state_1.reshape(1, -1)).to(device)
        return self.actor_1(state_1).cpu().data.numpy().flatten()

    # training cycle
    def train_0(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q_0 = 0
        max_Q_0 = -inf
        av_loss_0 = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (s0_batch, a0_batch, r0_batch, t0_batch, s2_0_batch,
            s1_batch, a1_batch, r1_batch, t1_batch, s2_1_batch), importance_weights, indices = replay_buffer.sample_batch(batch_size)

            importance_weights = torch.Tensor(importance_weights).to(device)  # Convert to tensor

            state_0 = torch.Tensor(s0_batch).to(device)
            next_state_0 = torch.Tensor(s2_0_batch).to(device)
            action_0 = torch.Tensor(a0_batch).to(device)
            reward_0 = torch.Tensor(r0_batch).to(device)
            done_0 = torch.Tensor(t0_batch).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action_0 = self.actor_target_0(next_state_0)

            # Add noise to the action
            noise_0 = torch.Tensor(a0_batch).data.normal_(0, policy_noise).to(device)
            noise_0 = noise_0.clamp(-noise_clip, noise_clip)
            next_action_0 = (next_action_0 + noise_0).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target_0(next_state_0, next_action_0)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q_0 += torch.mean(target_Q)
            max_Q_0 = max(max_Q_0, torch.max(target_Q))

            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward_0 + ((1 - done_0) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic_0(state_0, action_0)

            target_Q = target_Q.squeeze().view(batch_size, -1)  # Ensure correct shape

            loss_0 = (importance_weights.view(-1, 1) * (F.mse_loss(current_Q1, target_Q, reduction='none') +
                                                        F.mse_loss(current_Q2, target_Q, reduction='none'))).mean()

            # Perform the gradient descent
            self.critic_optimizer_0.zero_grad()
            loss_0.backward()
            self.critic_optimizer_0.step()

            td_error = torch.abs(current_Q1 - target_Q).cpu().data.numpy() + torch.abs(current_Q2 - target_Q).cpu().data.numpy()
            replay_buffer.update_priorities(indices, td_error)

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values (essentiallfy perform gradient ascent)
                actor_grad, _ = self.critic_0(state_0, self.actor_0(state_0))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer_0.zero_grad()
                actor_grad.backward()
                self.actor_optimizer_0.step()

                # Use soft update to update the actor-target network parameters by infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor_0.parameters(), self.actor_target_0.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing small amount of current parameters
                for param, target_param in zip(
                    self.critic_0.parameters(), self.critic_target_0.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss_0 += loss_0.item()

        self.iter_count_0 += 1

        # Write new values for tensorboard
        self.writer.add_scalar("Loss/loss0", av_loss_0 / iterations, self.iter_count_0)
        self.writer.add_scalar("Avg/Av. Q0", av_Q_0 / iterations, self.iter_count_0)
        self.writer.add_scalar("Max/Max. Q0", max_Q_0, self.iter_count_0)
    
    def train_1(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q_1 = 0
        max_Q_1 = -inf
        av_loss_1 = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (s0_batch, a0_batch, r0_batch, t0_batch, s2_0_batch,
            s1_batch, a1_batch, r1_batch, t1_batch, s2_1_batch), importance_weights, indices = replay_buffer.sample_batch(batch_size)

            importance_weights = torch.Tensor(importance_weights).to(device)  # Convert to tensor

            state_1 = torch.Tensor(s1_batch).to(device)
            next_state_1 = torch.Tensor(s2_1_batch).to(device)
            action_1 = torch.Tensor(a1_batch).to(device)
            reward_1 = torch.Tensor(r1_batch).to(device)
            done_1 = torch.Tensor(t1_batch).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action_1 = self.actor_target_1(next_state_1)

            # Add noise to the action
            noise_1 = torch.Tensor(a1_batch).data.normal_(0, policy_noise).to(device)
            noise_1 = noise_1.clamp(-noise_clip, noise_clip)
            next_action_1 = (next_action_1 + noise_1).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target_1(next_state_1, next_action_1)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q_1 += torch.mean(target_Q)
            max_Q_1 = max(max_Q_1, torch.max(target_Q))

            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward_1 + ((1 - done_1) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic_1(state_1, action_1)

            # Calculate the loss between the current Q value and the target Q value
            target_Q = target_Q.squeeze().view(batch_size, -1)  # Ensure correct shape

            loss_1 = (importance_weights.view(-1, 1) * (F.mse_loss(current_Q1, target_Q, reduction='none') +
                                                        F.mse_loss(current_Q2, target_Q, reduction='none'))).mean()

            # Perform the gradient descent
            self.critic_optimizer_1.zero_grad()
            loss_1.backward()
            self.critic_optimizer_1.step()

            td_error = torch.abs(current_Q1 - target_Q).cpu().data.numpy() + torch.abs(current_Q2 - target_Q).cpu().data.numpy()
            replay_buffer.update_priorities(indices, td_error)

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values (essentiallfy perform gradient ascent)
                actor_grad, _ = self.critic_1(state_1, self.actor_1(state_1))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer_1.zero_grad()
                actor_grad.backward()
                self.actor_optimizer_1.step()

                # Use soft update to update the actor-target network parameters by infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor_1.parameters(), self.actor_target_1.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing small amount of current parameters
                for param, target_param in zip(
                    self.critic_1.parameters(), self.critic_target_1.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss_1 += loss_1.item()

        self.iter_count_1 += 1

        # Write new values for tensorboard
        self.writer.add_scalar("Loss/loss1", av_loss_1 / iterations, self.iter_count_1)
        self.writer.add_scalar("Avg/Av. Q1", av_Q_1 / iterations, self.iter_count_1)
        self.writer.add_scalar("Max/Max. Q1", max_Q_1, self.iter_count_1)

    def save_0 (self, file_name_0, directory):
        torch.save(self.actor_0.state_dict(), "%s/%s_actor.pth" % (directory, file_name_0))
        torch.save(self.critic_0.state_dict(), "%s/%s_critic.pth" % (directory, file_name_0))

    def load_0(self, file_name_0, directory):
        # Load actor weights
        actor_weights = torch.load(
            f"{directory}/{file_name_0}_actor.pth", weights_only=True
        )
        self.actor_0.load_state_dict(actor_weights)

        # Load critic weights
        critic_weights = torch.load(
            f"{directory}/{file_name_0}_critic.pth", weights_only=True
        )
        self.critic_0.load_state_dict(critic_weights)

    def save_1 (self, file_name_1, directory):
        torch.save(self.actor_1.state_dict(), "%s/%s_actor.pth" % (directory, file_name_1))
        torch.save(self.critic_1.state_dict(), "%s/%s_critic.pth" % (directory, file_name_1))

    def load_1(self, file_name_1, directory):
        # Load actor weights
        actor_weights = torch.load(
            f"{directory}/{file_name_1}_actor.pth", weights_only=True
        )
        self.actor_1.load_state_dict(actor_weights)

        # Load critic weights
        critic_weights = torch.load(
            f"{directory}/{file_name_1}_critic.pth", weights_only=True
        )
        self.critic_1.load_state_dict(critic_weights)

# Set the parameters for the implementation
if torch.cuda.is_available():
    device = torch.device("cpu")
    warnings.warn("Training with CUDA")
else:
    warnings.warn("CUDA is not available. Running on CPU.")
    device = torch.device("cpu")

seed = 1234  # Random seed number
eval_freq = 5e3  # After how many steps to perform the evaluation
eval_step = 500 # maximum number of steps per episode in the evaluation
max_ep = 500 # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e6  # Maximum number of steps to perform
expl_noise = 0.5 #1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = 500000  # Number of steps over which the initial exploration noise will decay over
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1e6  # Maximum size of the buffer
file_name_0 = "TD3_velodyne_0"  # name of the file to store the policy
file_name_1 = "TD3_velodyne_1"  # name of the file to store the policy
folder_load = "recl_train"
model_path = os.path.join("./pytorch_models", folder_load)
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model
random_near_obstacle = True  # To take random actions near obstacles or not

# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Create the training environment
environment_dim = 20
robot_dim = 6
env = GazeboEnv("train_2robots.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1
lr_act = 1e-4
lr_cr = 1e-4

# Create the network
network_0 = TD3(state_dim, action_dim, max_action, lr_act, lr_cr)
network_1 = TD3(state_dim, action_dim, max_action, lr_act, lr_cr)

max_eps = 6000

# Set initial values for PER parameters
alpha = 0.6  # Controls prioritization strength (0 = uniform, 1 = fully prioritized)
beta = 0.4   # Importance sampling exponent (starts small and increases)
beta_increment_per_sampling = 0.0001
eps = 1e-6

# Initialize the Prioritized Experience Replay buffer
replay_buffer = PrioritizedReplayBuffer(
    buffer_size=int(buffer_size), 
    alpha=alpha, 
    beta=beta, 
    beta_increment_per_sampling=beta_increment_per_sampling, 
    eps=eps
)

if load_model:
    try:
        print(
            "Successfully Load The Previous Model"
        )
        network_0.load_0(file_name_0, model_path)
        network_1.load_1(file_name_1, model_path)
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# Create evaluation data store
evaluations = []

timestep_0 = 0
timesteps_0_since_eval = 0
done_0 = True

timestep_1 = 0
timesteps_1_since_eval = 0
done_1 = True
epoch = 1

count_rand_actions = 0
random_action = []

# Create a folder structure with date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join("./results", current_time)
models_dir = os.path.join("./pytorch_models", current_time)

# Create the directories if they do not exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Begin the training loop
while timestep_0 < max_timesteps and timestep_1 < max_timesteps:

    # On termination of episode
    if done_0 or done_1:
        if timestep_0 != 0:
            network_0.train_0(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )
        
        if timestep_1 != 0:
            network_1.train_1(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        if timesteps_0_since_eval >= eval_freq and timesteps_1_since_eval >= eval_freq:
            print("Validating After 5000 Steps")
            timesteps_0_since_eval %= eval_freq
            timesteps_1_since_eval %= eval_freq

            evaluations.append(
                evaluate(network_0 = network_0, network_1 = network_1, epoch=epoch, eval_episodes=eval_ep)
            )
            
            network_0.save_0(file_name_0, directory=models_dir)
            network_1.save_1(file_name_1, directory=models_dir)
            np.save(os.path.join(results_dir, f"{file_name_0}_evaluations.npy"), evaluations)
            np.save(os.path.join(results_dir, f"{file_name_1}_evaluations.npy"), evaluations)
            epoch += 1

        state_0, state_1= env.reset()
        done_0 = False
        done_1 = False
        episode_timesteps = 0

    # Add exploration noise
    expl_noise = max(expl_min, expl_noise - ((1 - expl_min) / expl_decay_steps))

    # Get actions
    action_0 = network_0.get_action_0(np.array(state_0))
    action_0 = (action_0 + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )
    action_1 = network_1.get_action_1(np.array(state_1))
    action_1 = (action_1 + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state_0[4:-8]) < 0.6
            and min(state_1[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-max_action, max_action, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action_0 = random_action
            action_0[0] = -max_action
            action_1 = random_action
            action_1[0] = -max_action

    # Perform actions and collect next states
    timeout = episode_timesteps + 1 == max_ep

    next_state_0, reward_0, done_0, _ = env.step_0(
        [(action_0[0] + max_action) / 2, action_0[1]], timeout=timeout
    )
    next_state_1, reward_1, done_1, _ = env.step_1(
        [(action_1[0] + max_action) / 2, action_1[1]], timeout=timeout
    )

    done_bool_0 = 0 if episode_timesteps + 1 == max_ep else int(done_0)
    done_bool_1 = 0 if episode_timesteps + 1 == max_ep else int(done_1)

    done_0 = 1 if episode_timesteps + 1 == max_ep else int(done_0)
    done_1 = 1 if episode_timesteps + 1 == max_ep else int(done_1)

    # Store experience in replay buffer
    replay_buffer.add(state_0, action_0, reward_0, done_bool_0, next_state_0,
                      state_1, action_1, reward_1, done_bool_1, next_state_1)

    # Update the counters
    state_0, state_1 = next_state_0, next_state_1
    episode_timesteps += 1
    timestep_0 += 1
    timestep_1 += 1
    timesteps_0_since_eval += 1
    timesteps_1_since_eval += 1

    # Curriculum stage switching every 5000 steps
    if (timestep_0 % 5000) < 3500:
        env.recl_stage = 0  # 0 for randomized
    else:
        env.recl_stage = 1  # 1 for constrained
    
# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network_0=network_0, network_1 = network_1, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network_0.save("%s" % file_name_0, directory="./models")
    network_1.save("%s" % file_name_1, directory="./models")
np.save("./results/%s" % file_name_0, evaluations)
np.save("./results/%s" % file_name_1, evaluations)
