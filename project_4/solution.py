import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode


class MLP(nn.Module):
    '''
    A simple ReLU MLP constructed from a list of layer widths.
    '''
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Critic(nn.Module):
    '''
    Simple MLP Q-function.
        '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        self.net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

        #####################################################################

    def forward(self, x, a):
        #####################################################################
        # DONE: code the forward pass
        # the critic receives a batch of observations and a batch of actions
        # of shape (batch_size x obs_size) and batch_size x action_size) respectively
        # and output a batch of values of shape (batch_size x 1)
        xa = torch.cat([x, a], 1)

        value = self.net(xa)
        #####################################################################
        return value

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    '''
    Gaussian stochastic actor.
    Outputs a (mean, std) of a normal dist. 
    '''
    def __init__(self, action_low, action_high,  obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # DONE: define the network layers
        # The network should output TWO values for each action dimension:
        # The mean (mu) and the log standard deviation (log_std)
        # So, the final layer should have 2 * action_size outputs

        self.net = MLP([obs_size] + ([num_units] * num_layers) + ([action_size*2]))

        # store action scale and bias: the actor's output can be squashed to [-1, 1]
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x):
        #####################################################################
        # DONE: code the forward pass
        # 1. Pass the observation x through the network
        # 2. Split the output into mean (mu) and log_std
        #    (Hint: use torch.chunk(..., 2, dim=-1))
        mu, log_std = torch.chunk(self.net(x), 2, dim=-1)

        # 3. Constrain log_std to be within [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)

        # 4. Calculate std = exp(log_std)
        std = torch.exp(log_std)

        # 5. Return a Normal distribution: torch.distributions.Normal(mu, std)
        # Return a distribution
        return torch.distributions.Normal(mu, std)


class Agent:

    # automatically select compute device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # DONE: store and tune hyperparameters here

    # sizes of the Neural networks
    num_units_pi = 256  
    num_layers_pi = 3   

    num_units_q = 256   
    num_layers_q = 3    

    batch_size: int = 256
    gamma: float = 0.99  # MDP discount factor

    tau: float = 0.005  # Polyak averaging coefficient
    learning_rate_q: float = 5e-4  # learning rate for critics
    learning_rate_pi: float = 5e-4  # learning rate for actor
    
    #########################################################################

    def __init__(self, env):

        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float()
        self.action_high = torch.tensor(env.action_space.high).float()
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2

        #####################################################################
        # DONE: initialize actor, critic and attributes
        # 1. Initialize Actor (pi) and Target Actor (pi_target)
        # (Use the Actor class)
        self.pi = Actor(self.action_low, self.action_high,
                        self.obs_size, self.action_size,
                        self.num_layers_pi, self.num_units_pi)
        self.pi_target = Actor(self.action_low, self.action_high, 
                               self.obs_size, self.action_size, 
                               self.num_layers_pi, self.num_units_pi)
        
        # 2. Initialize TWO Critics (q1, q2) and their Targets (q1_target, q2_target)
        self.q1 = Critic(self.obs_size, self.action_size, 
                         self.num_layers_q, self.num_units_q)
        self.q1_target = Critic(self.obs_size, self.action_size, 
                                self.num_layers_q, self.num_units_q)

        self.q2 = Critic(self.obs_size, self.action_size,
                         self.num_layers_q, self.num_units_q)
        self.q2_target = Critic(self.obs_size, self.action_size, 
                                self.num_layers_q, self.num_units_q)

        # Move all networks to device
        self.pi.to(self.device)
        self.pi_target.to(self.device)
        self.q1.to(self.device)
        self.q1_target.to(self.device)
        self.q2.to(self.device)
        self.q2_target.to(self.device)
        self.action_scale = self.action_scale.to(self.device)
        self.action_bias = self.action_bias.to(self.device)

        # 3. Initialize the optimizers for q1, q2, and pi
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=self.learning_rate_pi)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.learning_rate_q)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.learning_rate_q)

        # 4. noise parameters for policy smoothing
        self.std_policy_noise = 0.2  
        self.noise_clip = 0.5

        # 5. initialize iterations count and target critic update rate
        self.it = 0
        self.policy_freq = 2  
 
        # 6. Initialize target networks with same weights as online networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.pi_target.load_state_dict(self.pi.state_dict())

        #####################################################################
        # create buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.train_step = 0
    
    def train(self):
        '''
        Updates actor and critic with one batch from the replay buffer.
        '''
        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)
        
        # Ensure correct shapes for reward and done
        reward = reward.unsqueeze(-1) 
        done = done.unsqueeze(-1)    
        #####################################################################
        # DONE: code TD3 training logic

        with torch.no_grad():
            # 0. advance iterations count for delayed target update
            self.it += 1

            # 1. Get target policy distribution at next_obs
            dist_target = self.pi_target(next_obs)

            # 2. Sample next actions from the target policy (stochastic version)
            # next_action = dist_target.rsample()
            # noise = (torch.randn_like(next_action) * self.std_policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # next_action = (next_action + noise).clamp(self.action_low, self.action_high)
            
            # 2. Sample next actions from the target policy (deterministic version)
            next_action = dist_target.mean
            
            # 3. Add clipped noise to next actions for target policy smoothing
            noise = (torch.randn_like(next_action) * self.std_policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = next_action + noise
            
            # 4. Apply tanh squashing
            next_action = torch.tanh(next_action)
            
            # 5. Rescale to environment bounds and clamp
            next_action = next_action * self.action_scale + self.action_bias
            next_action = next_action.clamp(self.action_low, self.action_high)

            # 6. Compute target Q-values using both target critics
            q1_target_val = self.q1_target(next_obs, next_action)
            q2_target_val = self.q2_target(next_obs, next_action)
            min_Q = torch.min(q1_target_val, q2_target_val)
            
            # 7. Compute the critic target value 
            target_q = reward + self.gamma * (1 - done)*min_Q
            
        # 8. Compute current Q-value estimates from both critics
        q1_values = self.q1(obs, action)
        q2_values = self.q2(obs, action)
        q1_loss = nn.functional.mse_loss(q1_values, target_q)
        q2_loss = nn.functional.mse_loss(q2_values, target_q)
        q_loss = q1_loss + q2_loss
        
        # 9. Update both critics by minimizing MSE loss against target
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # --- Policy Update: Delayed Actor Updates ---
        # Update the actor less frequently than the critic (every d steps)
        if self.it % self.policy_freq == 0:
            # 10. Get current policy distribution at obs
            dist_action = self.pi(obs)

            # 11. Get next action (stochastic version)
            # sampled_actions = dist_action.rsample()
            # sampled_actions = torch.tanh(sampled_actions)
            # policy_actions = sampled_actions * self.action_scale + self.action_bias
            
            # 11. Get next action (deterministic version)
            action_mean = dist_action.mean

            # 12. Apply tanh squashing to actions
            action_tanh = torch.tanh(action_mean)
            
            # 13. Rescale actions to environment bounds
            policy_actions = action_tanh * self.action_scale + self.action_bias

            # 14. Compute Q-value of policy actions using first critic only
            # (Deterministic policy gradient: maximize Q1(s, pi(s)))
            Q1 = self.q1(obs, policy_actions)
                
            # 15. Compute the policy loss (maximize Q-value)
            # (Hint: pi_loss = -Q1(s, pi(s)).mean())
            pi_loss = -Q1.mean()
            
            # 16. Update the actor policy (only every d training steps)
            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            self.pi_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.pi.parameters(), self.pi_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        #####################################################################

    def get_action(self, obs, train):
        '''
        Returns the agent's action for a given observation.
        The train parameter can be used to control stochastic behavior.
        '''
        #####################################################################
        # DONE: return the agent's action
        
        # 1. Convert obs to a tensor and send to device
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 2. Get the action distribution from the actor (self.pi)
        dist = self.pi(obs_tensor)

        with torch.no_grad():
            # 3. Get action
            if train:
                # During training: sample from the distribution
                action_gaussian = dist.sample()
            else:
                # During testing: use the deterministic mean
                action_gaussian = dist.mean
            
            # 4. Squash the action through a Tanh function
            action_tanh = torch.tanh(action_gaussian)
            
            # 5. Rescale and shift the action to the environment's bounds
            # (Use self.action_scale and self.action_bias)
            action_scaled = action_tanh * self.action_scale + self.action_bias
            
            # 6. Convert to numpy array and return
            action = action_scaled.squeeze(0).cpu().numpy()
        
        #####################################################################
        return action

    def store(self, transition):
        '''
        Stores the observed transition in a replay buffer containing all past memories.
        '''
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 50  # interactive episodes
    TEST_EPISODES = 300  # evaluation episodes
    save_video = False
    verbose = True
    seeds = np.arange(10)  # seeds for public evaluation

    start = time.time()
    print(f'Running public evaluation.') 
    test_returns = {k: [] for k in seeds}

    for seed in seeds:

        # seeding to ensure determinism
        seed = int(seed)
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)
        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        env.np_random, _ = seeding.np_random(seed)

        agent = Agent(env)

        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)

        for _ in range(TRAIN_EPISODES):
            run_episode(env, agent, mode='train', verbose=verbose, rec=False)

        for n_ep in range(TEST_EPISODES):
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)  # only record last episode
            with torch.no_grad():
                episode_return = run_episode(env, agent, mode='test', verbose=verbose, rec=video_rec)
            test_returns[seed].append(episode_return)

    avg_test_return = np.mean([np.mean(v) for v in test_returns.values()])
    within_seeds_deviation = np.mean([np.std(v) for v in test_returns.values()])
    across_seeds_deviation = np.std([np.mean(v) for v in test_returns.values()])
    print(f'Score for public evaluation: {avg_test_return}')
    print(f'Deviation within seeds: {within_seeds_deviation}')
    print(f'Deviation across seeds: {across_seeds_deviation}')

    print("Time :", (time.time() - start)/60, "min")
