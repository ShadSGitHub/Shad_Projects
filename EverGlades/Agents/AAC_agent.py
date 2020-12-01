import os
import time
import random
from collections import namedtuple, deque
#test
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import gym_everglades

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-5               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

NODE_CONNECTIONS = {
    1: [2, 4],
    2: [1, 3, 5],
    3: [2, 4, 5, 6, 7],
    4: [1, 3, 7],
    5: [2, 3, 8, 9],
    6: [3, 9],
    7: [3, 4, 9, 10],
    8: [5, 9, 11],
    9: [5, 6, 7, 8, 10],
    10: [7, 9, 11],
    11: [8, 10]
}

NUM_GROUPS = 12

ENV_MAP = {
    'everglades': 'Everglades-v0',
    'everglades-vision': 'EvergladesVision-v0',
    'everglades-stoch': 'EvergladesStochastic-v0',
    'everglades-vision-stoch': 'EvergladesVisionStochastic-v0',
}

device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, log_prob, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, log_prob, reward, next_state, done)

    def act(self, state, eps=0.):

        # Epsilon-greedy action selection
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            dist, value = self.qnetwork_local(state)
        self.qnetwork_local.train()
        actionTaken = dist.sample()
        log_prob = dist.log_prob(actionTaken)

        actions = np.zeros((7, 2))
        prioritized_actions = np.flip(actionTaken.cpu().data.numpy().argsort())
        selected_groups = []
        for action in prioritized_actions[0]:
            group = np.floor(action / 11.).astype(int)
            node = int(action % 11) + 1
            if group not in selected_groups:
                actions[len(selected_groups), 0] = group
                actions[len(selected_groups), 1] = node
                selected_groups.append(group)
            if len(selected_groups) >= 7:
                break
        return actions, log_prob
       
    def learn(self):
        if len(self.memory) > BATCH_SIZE:
            for i in range(20):
                experiences = self.memory.sample()

                states, actions, log_prob, rewards, next_states, dones = experiences
                
                dist, values = self.qnetwork_local(states)
                next_dist, next_values = self.qnetwork_local(next_states)
                
                delta = rewards + (GAMMA * next_values * (1 - dones))
                advantage = delta - values
                allActions = dist.sample()
                new_log_prob =  dist.log_prob(allActions)

                ratio = torch.exp(new_log_prob - log_prob)
                
                actor_loss = -(ratio*advantage).mean()
                critic_loss = F.mse_loss(delta, values)
                entropy = 0.5*(dist.entropy().mean())

                loss = entropy + critic_loss + actor_loss
                
                # Minimize the loss
                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm(self.qnetwork_local.parameters(), 0.1)
                self.optimizer.step()
            self.memory = ReplayBuffer(132, BUFFER_SIZE, BATCH_SIZE, 0)

        # ------------------- update target network ------------------- #
        #self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "log_prob", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.counter = 0
    
    def add(self, state, action, log_probs, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action,log_probs, reward, next_state, done)
        self.memory.append(e)
        self.counter += 1
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        log_probs = torch.from_numpy(np.vstack([e.log_prob for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, log_probs, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=202, fc2_units=202):
        super(QNetwork, self).__init__()
        #self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.l1 = nn.Linear(state_size,fc1_units)
        self.l2 = nn.Linear(fc1_units,fc2_units)
        self.l3 = nn.Linear(fc2_units,1)

        self.std = nn.Parameter(torch.ones(action_size))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(x)
        x = self.fc3(x)

        v = F.relu(self.l1(state))
        v = F.relu(self.l2(v))
        v = self.l3(v)

        std = self.std

        mean = x
        std  = std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, F.softplus(std))
        return dist, v
