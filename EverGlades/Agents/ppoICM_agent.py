import os
import time
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import torch.optim as optim
from itertools import chain
from torch.autograd import Variable


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
#device = torch.device("cuda:0")

class Agent():
    
    def __init__(self, state_size, action_size):
       
        self.state_size = state_size
        self.action_size = action_size

        self.qnetwork_policy = QNetwork(state_size, action_size).to(device)
        self.ICM_policy = ICM(state_size, action_size).to(device)

        self.optimizer = optim.Adam(chain(self.qnetwork_policy.parameters(),self.ICM_policy.parameters()), lr=LR)

        self.reward_memory = []
        self.log_probs_memory = []
        self.entropy_memory = []
        self.gamma = 0.99
        self.BETA = 0.001
        self.GAE_LAMBDA = 0.95
        self.PPO_EPSILON = 0.4
        self.CRITIC_DISCOUNT = 0.5
        self.MINI_BATCH_SIZE = 40
        self.PPO_EPOCHS = 10
        self.curiosity_beta = 0.2
        self.curiosity_lambda = 0.1

    def test_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        dist, _ = self.qnetwork_policy(state)
        action = dist.sample()
        prioritized_actions = np.flip(action.cpu().data.numpy().argsort())
    
        actions = np.zeros((7, 2))
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
        return actions

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        dist, value = self.qnetwork_policy(state)

        allActions = dist.sample()

        entropy = dist.entropy().sum(-1).unsqueeze(-1)

        ret_log_prob = dist.log_prob(allActions)

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
        return actions, value, ret_log_prob, allActions
    
    # remember to experiment with different coefficients
    def Intrinsic(self, rewards, states, next_states, actions, reward_scale=0.01, eta=0.01):
        new_rewards = []

        for step in range(len(states)):
            dist, pred_next_state, encoded_next_state = self.ICM_policy(states[step], next_states[step], actions[step])
            intrinsic_reward = reward_scale / 2 *F.mse_loss(pred_next_state, encoded_next_state)
            new_rewards.insert(len(new_rewards),(1. - eta) * rewards[step] + eta * intrinsic_reward)
        return new_rewards

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

    def normalize_rewards(self,x):
        t = torch.cat(x).detach()
        mean = t.mean()
        std = t.std() if t.std() > 0 else 1
        for step in range(len(x)):
            x[step] = x[step] - mean
            x[step] = x[step] / std
        return x

    def ppo_iter(self, states, next_states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        # generates random mini-batches until we have covered the full batch
        for _ in range(batch_size // self.MINI_BATCH_SIZE):
            rand_ids = np.random.randint(0, batch_size, self.MINI_BATCH_SIZE)
            yield states[rand_ids, :], next_states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

                 
    def ppo_update(self, states, next_states, actions, log_probs, returns, advantages, clip_param=0.1):
        for _ in range(self.PPO_EPOCHS):
            for state, next_state, action, old_log_probs, return_, advantage in self.ppo_iter(states, next_states, actions, log_probs, returns, advantages):

                dist, value = self.qnetwork_policy(state)

                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = torch.exp((new_log_probs - old_log_probs))
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(return_ , value)

                
                cur_loss = self.curiosityloss(state, next_state, action)

                self.optimizer.zero_grad()
                
                loss = self.CRITIC_DISCOUNT * critic_loss + actor_loss - self.BETA * entropy
                loss = loss * self.curiosity_lambda + cur_loss
                loss.backward()
                self.optimizer.step()
                
               

    def curiosityloss(self, states, next_states, actions):
        dist, pred_next_state, encoded_next_state = self.ICM_policy(states, next_states, actions)
        #forward_loss = 0.5*F.mse_loss(pred_next_state, encoded_next_state)
        forward_loss = F.mse_loss(pred_next_state, encoded_next_state)
        action_hat = dist.sample()
        #loss = MSELoss()
        #inverse_loss = F.mse_loss(action_hat, actions)
        
        inverse_loss = F.cross_entropy(action_hat, torch.max(actions, 1)[1])

        curiosity_loss = self.curiosity_beta * forward_loss + (1 - self.curiosity_beta) * inverse_loss
        #curiosity_loss = self.curiosity_beta * forward_loss + self.curiosity_beta * inverse_loss
        return curiosity_loss

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=202, fc2_units=202):
        super(QNetwork, self).__init__()
        '''
            input *2/3 + output = number of node in hidden layer
            input <- state_size <- 105
            output <- acton_size <- 132
            105 *2/3 + 132 = 202

            this calcualtion was decided based on research and testing from other researchers
            input *2/3 + output = number of node in hidden layer is a general rule of thumb 
            but not universal.
        '''
        # Actor or policy netowork
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        # Critic or Value network
        self.l1 = nn.Linear(state_size,fc1_units)
        self.l2 = nn.Linear(fc1_units,fc2_units)
        self.l3 = nn.Linear(fc2_units,1)

        self.std = nn.Parameter(torch.ones(action_size))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        v = self.l1(state)
        v = self.l2(v)
        v = self.l3(v)

        std = self.std
        mean = torch.tanh(x)
        std  = std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, F.softplus(std))
        return dist, v

class ICM(nn.Module):
    def __init__(self, state_size, action_size):
        super(ICM, self).__init__()
        
        self.encoderI = nn.Linear(state_size,202)
        self.encoderH = nn.Linear(202, 202)
        self.encoderO = nn.Linear(202, 202)
        
        # ICM forward network
        self.F1 = nn.Linear(202 + action_size, 327)
        self.F2 = nn.Linear(327, 327)
        self.F3 = nn.Linear(327, 202)

        # ICM Inverse network
        self.I1 = nn.Linear(202 * 2, 400)
        self.I2 = nn.Linear(400,400)
        self.I3 = nn.Linear(400, action_size)

        self.std = nn.Parameter(torch.ones(action_size))


    def forward(self, state, next_state, action):
        
        state = self.encoderI(state)
        state = self.encoderH(state)
        state = self.encoderO(state)

        next_state = self.encoderI(next_state)
        next_state = self.encoderH(next_state)
        next_state = self.encoderO(next_state)

        stateAction = torch.cat([action,state],1)

        phi = F.relu(self.F1(stateAction))
        phi = F.relu(self.F2(phi))
        pred_next_state = F.relu(self.F3(phi))

        stateNextState = torch.cat([state,next_state],1)

        stateNextState = F.relu(self.I1(stateNextState))
        stateNextState = F.relu(self.I2(stateNextState))
        pred_action = F.relu(self.I3(stateNextState))
        std = self.std
        mean = torch.tanh(pred_action)
        std  = std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, F.softplus(std))

        return dist, pred_next_state, next_state