## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb
import time
import random

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

from everglades_server import server

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
# Plotting Function
def plot(aveScore, aveScore_deque1000, aveScoreTotal):
    #plt.figure(1)
    plt.clf()        
    plt.title('Rainbow DQN')
    plt.xlabel('Episode')
    plt.ylabel('Winning Rate')
    plt.axhline(y=50, color='r', linestyle='-')
    plt.axhline(y=75, color='g', linestyle='-')
    plt.axhline(y=95, color='b', linestyle='-')
    #plt.axhline(y=60, linestyle='dotted')
    #plt.axhline(y=70, linestyle='dotted')
    #plt.axhline(y=80, linestyle='dotted')
    #plt.axhline(y=90, linestyle='dotted')
    plt.plot(aveScore)
    plt.plot(aveScore_deque1000, color='m', linestyle='dashed')
    plt.plot(aveScoreTotal, color='y', linestyle='dotted')  
    #plt.pause(0.001)
    #if is_ipython: display.clear_output(wait=True)
    plt.savefig('graphRainbow.png', dpi=200)

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path
agent0_file = 'agents/Rainbow.py'
agent1_file = 'agents/random_actions.py'

config_dir = './config/'
map_file = config_dir + 'DemoMap.json'
setup_file = config_dir + 'GameSetup.json'
unit_file = config_dir + 'UnitDefinitions.json'
output_dir = './game_telemetry/'

debug = 0

## Specific Imports
agent0_name, agent0_extension = os.path.splitext(agent0_file)
agent0_mod = importlib.import_module(agent0_name.replace('/','.'))
agent0_class = getattr(agent0_mod, os.path.basename('Agent'))

agent1_name, agent1_extension = os.path.splitext(agent1_file)
agent1_mod = importlib.import_module(agent1_name.replace('/','.'))
agent1_class = getattr(agent1_mod, os.path.basename(agent1_name))

## Main Script
env = gym.make('everglades-v0')
players = {}
names = {}

players[0] = agent0_class(state_size=105, action_size=12*11, seed=0)
names[0] = agent0_class.__name__
players[1] = agent1_class(env.num_actions_per_turn, 1)
names[1] = agent1_class.__name__

actions = {}

n_episodes=200000
max_t=150
eps_start=1.0
eps_end=0.01
eps_decay=0.999

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
aveScore_deque = deque(maxlen=200000)

scores_deque1000 = deque(maxlen=1000) 
aveScore_deque1000 = deque(maxlen=200000) # 1000

scores_dequeTotal = deque(maxlen=200000)
aveScore_dequeTotal = deque(maxlen=200000)
eps = eps_start                    # initialize epsilon

# temporal storage
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

for i_episode in range(1, n_episodes+1):

    observations = env.reset(
        players=players,
        config_dir = config_dir,
        map_file = map_file,
        unit_file = unit_file,
        output_dir = output_dir,
        pnames = names
    )

    state = observations[0]
    score = 0
    memoryOfGame = deque(maxlen=1050)

    for t in range(max_t):
        actions0 = players[0].act( observations[0], eps)
        actions1 = players[1].get_action( observations[1] )

        actions[0] = actions0
        actions[1] = actions1
        # print(actions)

        observations, reward, done, info = env.step(actions)

        # DQN step() can only train one action at a time, so step 7 times
        for index in range(actions0.shape[0]):
            top_action = int(actions0[index, 0] * 11 + actions0[index, 1] - 1)
            e = experience(state, top_action, reward[0], observations[0], done)
            memoryOfGame.append(e)

        state = observations[0]
        score += reward[0]

        if done:
            for ex in memoryOfGame:
                players[0].step(ex.state, ex.action, ex.reward, ex.next_state, ex.done)
            players[0].completeBellSteps(done)
            break
    
    scores_deque.append(score if score == 1 else 0)       # save most recent score
    aveScore_deque.append(100*np.mean(scores_deque))

    scores_deque1000.append(score if score == 1 else 0)     # save most recent score
    aveScore_deque1000.append(100*np.mean(scores_deque1000))

    scores_dequeTotal.append(score if score == 1 else 0)
    aveScore_dequeTotal.append(100*np.mean(scores_dequeTotal))

    plot(aveScore_deque, aveScore_deque1000, aveScore_dequeTotal)
    eps = max(eps_end, eps_decay*eps) # decrease epsilon
    print('Episode {}\tWinning Rate: {:.4f}\tEpisode Reward: {:.4f}'.format(i_episode, np.mean(scores_deque), score))
    torch.save(players[0].qnetwork_local.state_dict(), 'checkpoint_rainbow.pth')
    