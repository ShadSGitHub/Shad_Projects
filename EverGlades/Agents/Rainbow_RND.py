# Rainbow DQN(Double, Dueling, Multi-Step, PER, Noisy Net)
import os
import time
import math, random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
import gym_everglades

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-5               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
HIDDEN_NODES = 202      # number of nodes in hidden layer


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

        self.FloatTensor =  torch.FloatTensor
        self.LongTensor = torch.LongTensor

        # Q-Network
        self.qnetwork_local = NoisyQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = NoisyQNetwork(state_size, action_size, seed).to(device)
        self.rnd = RND(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # The new replay memory for PER structure
        self.memory = PrioritizedExperienceReplay(BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def completeBellSteps(self, done):
        self.memory.finishBellSteps(done)
    
    def step(self, state, action, reward, next_state, done):
        rState = torch.from_numpy(state).float().unsqueeze(0).to(device)
        i_reward = self.rnd.get_reward(rState).detach().clamp(-1.0,1.0).item()

        reward += i_reward
        # Save experience in SumTree
        experience = state, action, reward, next_state, done
        self.memory.store(experience)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if (self.memory.tree.sizeTree) > BATCH_SIZE:
                self.learn(self.memory, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            # return np.argmax(action_values.cpu().data.numpy())
            actions = np.zeros((7, 2))
            prioritized_actions = np.flip(action_values.cpu().data.numpy().argsort())
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
        else:
            # return random.choice(np.arange(self.action_size))
            actions = np.zeros((7, 2))
            actions[:, 0] = np.random.choice(12, 7, replace=False)
            actions[:, 1] = np.random.choice(11, 7, replace=False) + 1
            return actions

    # modified for experience replay
    def learn(self, memory, gamma):
        
        
        # Sampling the Replay buffer
        miniBatch, batchIndex = memory.sample(BATCH_SIZE)

        # Taking the values out of the miniBatch
        [states, actions, rewards, next_states, dones] = zip(*miniBatch)
        
        # Applying values to Tensors
        state = Variable(self.FloatTensor(states))
        action = Variable(self.LongTensor(actions))
        reward = Variable(self.FloatTensor(rewards))
        nextState = Variable(self.FloatTensor(next_states))


        Ri = self.rnd.get_reward(state)
        self.rnd.update(Ri)

        Q_expected = self.qnetwork_local(state).gather(1, action.view(-1,1)).view(-1)

        next_local_value = self.qnetwork_local(nextState).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(nextState).detach().gather(dim = 1, index = next_local_value).squeeze(1)
        for i in range(BATCH_SIZE):
            if dones[i]:
                Q_targets_next.data[i] = 0
        
        Q_targets = (Q_targets_next * gamma) + reward
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)        
        
        td_error = Q_targets - Q_expected
        for i in range(BATCH_SIZE):
            val = abs(td_error[i].item()) 
            memory.updatePriorities(batchIndex[i],val)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

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

class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    More memory effecient than NoisyLinear, gets the same results
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)

class NoisyQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=HIDDEN_NODES, fc2_units=HIDDEN_NODES):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(NoisyQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = NoisyFactorizedLinear(fc1_units, fc2_units)

        self.value = nn.Linear(fc2_units, 1)
        self.Advantage = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        Value = self.value(x)
        Advantage = self.Advantage(x)

        advAveragSumTreee  = torch.mean(Advantage, dim=1, keepdim=True)
        Q_value = Value + Advantage - advAveragSumTreee

        return Q_value

class PrioritizedExperienceReplay:
    epsilon = 0.01
    alpha = 0.6


    # N-Step modification
    
    
    # Creating the tree
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.Bell = []
        self.bellSteps = 4*7 
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    # Stores the experiences in the tree with their priority
    # Modified for N-Step
    
    def store(self, experience):
        
        # N-Step modification
        self.Bell.append(experience)

        _, _, _, next_state, _ = experience


        if(len(self.Bell) < self.bellSteps):
            return

        R = sum([self.Bell[i][2]*(GAMMA**i) for i in range(self.bellSteps)])
        popped_state, popped_action, _, _, popped_done = self.Bell.pop(0)

        e = self.experience(popped_state, popped_action, R, next_state, popped_done)


        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.

        self.tree.add(max_priority, e)
    '''
    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.

        self.tree.add(max_priority, experience)
    '''

    # Determines which batch of experiences to sample from
    def sample(self, z):

        miniBatch = []
        batchIndex = []
        prioritySegment = self.tree.totalTree / z

        for i in range(z):
            x = prioritySegment * i
            y = prioritySegment * (i + 1)
            value = np.random.uniform(x, y)
            index, priority, data = self.tree.get(value)
            miniBatch.append(data)
            batchIndex.append(index)

        return miniBatch, batchIndex

    # updating the priorities of the values
    def updatePriorities(self, treeIndex, error):
        priority = self.getPriority(error)
        self.tree.update(treeIndex, priority)

    def getPriority(self, error):
        return (error + self.epsilon) ** self.alpha

    def finishBellSteps(self, done):
        while len(self.Bell) > 0:
            R = sum([self.Bell[i][2]*(GAMMA**i) for i in range(len(self.Bell))])
            popped_state, popped_action, _, popped_next_state, _ = self.Bell.pop(0)

            e = self.experience(popped_state, popped_action, R, popped_next_state, done)

            max_priority = np.max(self.tree.tree[-self.tree.capacity:])
            if max_priority == 0:
                max_priority = 1.

            self.tree.add(max_priority, e)
        return


# Implementation of sum tree data structure. Every node is the sum of its children
# In this case the priorities will be the leaf nodes
"""
EXAMPLE:
                20
               /  \
              /    \
             7      13
"""
# The reason for using this is because storing in a sorted array would have horrible 
# efficiency. You would have O(n log n) for adding/updating, then O(n) for sampling.
# With this sumtree, adding/updating/sampling all take O(log n) time

class SumTree:
    
    constant = 0


    # This initializes the tree, and makes all nodes = 0
    def __init__(self, capacity):

        # Determinies the size of the tree
        self.capacity = capacity
        self.sizeTree = 0

        # Forms the treecapacity
        self.tree = np.zeros(2 * capacity - 1)

        # Holds the experiences
        self.data = np.zeros(capacity, dtype=object)

    # Adds the priority to the leaf and also adds the experience inside of data
    def add(self, x, data):

        #determines the index for the experience, puts the expereience in, then updates the priority of the leaf
        treeIndex = self.constant + self.capacity - 1
        self.data[self.constant] = data
        self.update(treeIndex, x)

        self.constant += 1
        self.sizeTree += 1

        # this is so that if we reach capacity, we go back to the first and start to overwrite
        if self.constant >= self.capacity:
            self.constant = 0

    # updating the tree
    def update(self, treeIndex, x):

        # updates the initial priority
        temporalChange = x - self.tree[treeIndex]
        self.tree[treeIndex] = x

        # updates the tree for the new priority being added
        while treeIndex !=0:
            treeIndex = (treeIndex - 1) // 2
            self.tree[treeIndex] += temporalChange


    # get the priority/sample the experience
    def get(self, y):
        parent = 0
        while True:

            leftChild = 2 * parent + 1
            rightChild = leftChild + 1

            # this will search for the priority node we want
            if leftChild >= len(self.tree):
                leaf = parent
                break
            else:
                if y <= self.tree[leftChild]:
                    parent = leftChild
                else:
                    y -= self.tree[leftChild]
                    parent = rightChild

        dataReturn = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[dataReturn]

    # This will return the whole tree
    @property
    def totalTree(self):
        return self.tree[0]

class QNetworkNoSeed(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=202, fc2_units=202):
        super(QNetworkNoSeed, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class RND:
    def __init__(self,state_size, action_size, seed):
        self.target = QNetworkNoSeed(state_size,action_size,seed).to(device)
        self.predict = QNetworkNoSeed(state_size,action_size,seed).to(device)
        self.optimizer = optim.Adam(self.predict.parameters(),lr=LR)
        
    def get_reward(self,state):
        y_true = self.target(state).detach()
        y_pred = self.predict(state)
        reward = torch.pow(y_pred - y_true,2).sum()
        return reward
    
    def update(self,Ri):
        self.optimizer.zero_grad()
        Ri.sum().backward()
        self.optimizer.step()