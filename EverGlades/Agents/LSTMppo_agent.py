import os
import time
import random
from collections import namedtuple, deque

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
    
    def __init__(self, state_size, action_size):
       
        self.state_size = state_size
        self.action_size = action_size

        self.qnetwork_policy = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_policy.parameters(), lr=LR)

        self.reward_memory = []
        self.log_probs_memory = []
        self.entropy_memory = []
        self.gamma = 0.99
        self.BETA = 0.001
        self.GAE_LAMBDA = 0.95
        self.PPO_EPSILON = 0.1
        self.CRITIC_DISCOUNT = 0.5
        self.MINI_BATCH_SIZE = 114
        self.PPO_EPOCHS = 20
        self.tau = 0.1

        self.h0 = torch.zeros(1, 202)
        self.c0 = torch.zeros(1, 202)
        self.h1 = torch.zeros(1, 202)
        self.c1 = torch.zeros(1, 202)

    def act(self, state, done):
        if done:
            self.h0 = torch.zeros(1, 202)
            self.c0 = torch.zeros(1, 202)
            self.h1 = torch.zeros(1, 202)
            self.c1 = torch.zeros(1, 202)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        dist, value , self.h0, self.c0, self.h1, self.c1 = self.qnetwork_policy(state, self.h0, self.c0, self.h1, self.c1)

        allActions = dist.sample()

        entropy = dist.entropy().sum(-1).unsqueeze(-1)

        log_prob = dist.log_prob(allActions)

        prioritized_all_actions = np.flip(allActions.cpu().data.numpy().argsort())
       
        actions = np.zeros((7, 2))
        selected_groups = []
        for action in prioritized_all_actions[0]:
            group = np.floor(action / 11.).astype(int)
            node = int(action % 11) + 1
            if group not in selected_groups:
                actions[len(selected_groups), 0] = group
                actions[len(selected_groups), 1] = node
                selected_groups.append(group)
            if len(selected_groups) >= 7:
                break
        return actions, value, log_prob, allActions , (self.h0, self.c0, self.h1, self.c1)

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, lam=0.95):
        values = values + [next_value]
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * lam * masks[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + values[step])
        return returns
    
    def normalize(self,x):
        x -= x.mean()
        x /= x.std() if x.std() > 0 else 1
        return x

    def ppo_iter(self, states, actions, log_probs, returns, advantage, h):
        h0, c0, h1, c1 = h
        batch_size = states.size(0)
        # generates random mini-batches until we have covered the full batch
        for i in range(batch_size // self.MINI_BATCH_SIZE):
            rand_ids = np.random.randint(0, batch_size, self.MINI_BATCH_SIZE)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :], \
            (h0[rand_ids, :], c0[rand_ids, :], h1[rand_ids, :], c1[rand_ids, :])
                 
    def ppo_update(self, states, actions, log_probs, returns, advantages, h, clip_param=0.1):

        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        for i in range(self.PPO_EPOCHS):
            # grabs random mini-batches several times until we have covered all data
            
            for state, action, old_log_probs, return_, advantage, hidden in self.ppo_iter(states, actions, log_probs, returns, advantages, h):
                #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                h0, c0, h1, c1 = hidden 
                #print(h0.size())
                dist, value, i,g,n,ore = self.qnetwork_policy(state, h0, c0, h1, c1)
                
                entropy = dist.entropy().mean()
                
                new_log_probs = dist.log_prob(action)

                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * advantage
                
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = -torch.min(surr1, surr2).mean() 
                critic_loss = F.mse_loss(return_, value) 

                loss = self.CRITIC_DISCOUNT * critic_loss + actor_loss - self.BETA * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=202, fc2_units=202):
        super(QNetwork, self).__init__()

        self.lstm_policy = nn.LSTMCell(state_size, 202)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, 404)
        self.lstm_policy = nn.LSTMCell(404, 202)
        self.fc3 = nn.Linear(202, action_size)

       


        self.l1 = nn.Linear(state_size,fc1_units)
        self.l2 = nn.Linear(fc1_units,404)
        self.lstm_value = nn.LSTMCell(404, 202)
        self.l3 = nn.Linear(fc2_units,1)

        self.std = nn.Parameter(torch.ones(action_size))

    def forward(self, state, h0, c0, h1, c1):  
        x = self.fc1(state)   
        
        x = self.fc2(x)
        h0, c0 = self.lstm_policy(x,(h0, c0))
        x = self.fc3(h0)

        v = self.l1(state)
        v = self.l2(h1)
        h1 , c1 = self.lstm_value(v,(h1, c1))
        v = self.l3(h1)

        std = self.std

        mean = torch.tanh(x)
        std  = std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, F.softplus(std))
        return dist, v , h0, c0, h1, c1