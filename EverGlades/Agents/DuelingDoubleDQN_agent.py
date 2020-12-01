# Proportional Prioritized Experience Replay DQN Implementation

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
from gym_everglades.envs.everglades_env import UnitState, GroupState



BUFFER_SIZE = int(1e7)	# replay buffer size
BATCH_SIZE = 128		# minibatch size
GAMMA = 0.99			# discount factor
TAU = 1e-3				# for soft update of target parameters
LR = 1e-4				# learning rate
UPDATE_EVERY = 5		# how often the network gets updated



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

ENV_MAP = {
    'everglades': 'Everglades-v0',
    'everglades-vision': 'EvergladesVision-v0',
    'everglades-stoch': 'EvergladesStochastic-v0',
    'everglades-vision-stoch': 'EvergladesVisionStochastic-v0',
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





class Agent():


	def __init__(self, state_size, action_size, random_seed):

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




        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)



    # Borth act and learn need to be rewritten from our original dqn for the prioritization
    def learn(self, experiences, gamma):
    	return 0
    def act(self,state, eps=0):
    	return 0	

# Implementation of sum tree data structure. Every node is the sum of its children
# In this case the priorities will be the leaf nodes
"""
EXAMPLE:
				20
			   /  \
			  /    \
			 7	    13
"""
class SumTree():
	def __init__(left, right, self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )
        self.left = left
        self.right = right

        # checks if node is a leaf
        self.is_leaf = is_leaf
        # if it is not a leaf, then add the node to its left and the node to its right
        if not self.is_leaf:
        	self.value = self.left.value + self.right.value
        # if it is then do nothing

        if left is not None:
        	left.parent = self
        if right is not None:
        	right.parent = self

    def complete(self):
    	return self.tree[0]

    def additions(self, addition, input):
    	x = self.write + self.capacity - 1
    	self.data[self.write] = input
    	self.update(x, addition)

    	self.write += 1















#####################################
"""
	ReplayBuffer needs to be changed for prioritization
"""
###################################
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def train_dqn(env, agent, n_episodes=2000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.999):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores_deque = deque(maxlen=100)   # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            actions = agent.act(state, eps)
            # print(actions)

            next_state, reward, done, _ = env.step(actions)

            # DQN step() can only train one action at a time, so step 7 times
            for index in range(actions.shape[0]):
                top_action = int(actions[index, 0] * 11 + actions[index, 1] - 1)
                agent.step(state, top_action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break

        scores_deque.append(score)       # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('Episode {}\tAverage Score: {:.4f}\tEpisode Score: {:.4f}'.format(i_episode, np.mean(scores_deque), score))

        if i_episode>100 and np.mean(scores_deque)>=0.8:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return


def main():

	agent = Agent(state_size=105, action_size = 12*11, seed = 0)


	# uncomment the ones that you want
	#watch_untrained_agent(env,agent)
	#train_dqn(env,agent)