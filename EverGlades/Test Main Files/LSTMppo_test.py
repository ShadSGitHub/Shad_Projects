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

def plot(aveScore):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Winning Rate')
    plt.axhline(y=70, color='b', linestyle='-')
    plt.axhline(y=60, color='g', linestyle='-')
    plt.axhline(y=50, color='r', linestyle='-')
    
    plt.plot(aveScore)    
    plt.pause(0.001)
    plt.savefig('LSTM_PPO_Graph.png')
    #plt.close()
    if is_ipython: display.clear_output(wait=True)

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path
agent0_file = 'agents/LSTMppo_agent.py'
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

players[0] = agent0_class(state_size=105, action_size=12*11)
names[0] = agent0_class.__name__
players[1] = agent1_class(env.num_actions_per_turn, 1)
names[1] = agent1_class.__name__

actionsArray = {}

n_episodes=100000
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
aveScore_deque = deque(maxlen=20000)
eps = eps_start                    # initialize epsilon

PPO_steps = 456

device = torch.device("cpu")

log_probs  = []
values     = []
values_    = []
states     = []
actions    = []
rewards    = []
masks      = []
h0s         = []
c0s         = []
h1s         = []
c1s         = []



ppo_counter = 0
done = False
d = False

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

    for t in range(max_t):
        actions0, value, log_prob, allActions, hidden = players[0].act(state, d)
        actions1 = players[1].get_action( observations[1] )
        actionsArray[0] = actions0
        actionsArray[1] = actions1 

        observations, reward, done, info = env.step(actionsArray)
        h0, c0, h1, c1 = hidden
       
        h0s.append(h0.detach()) 
        c0s.append(c0.detach()) 
        h1s.append(h1.detach())
        c1s.append(c1.detach())

        log_probs.append(log_prob)
        values.append(value)
        
        rewards.append(torch.FloatTensor([reward[0]]).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
        
        state = np.reshape(state, (-1, len(state)))
        state = torch.FloatTensor(state).to(device)
        
        states.append(state)
        actions.append(allActions)

        #keep this line for now
        state = observations[0]
        score += reward[0]
        ppo_counter += 1

        if ppo_counter % PPO_steps == 0:
            d = True
            _, next_value, _, _ , _= players[0].act(observations[0], done)
            returns = players[0].compute_gae(next_value, rewards, masks, values)
            
            
            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values
            advantage = players[0].normalize(advantage)

            h0s         = torch.cat(h0s)
            c0s         = torch.cat(c0s)
            h1s         = torch.cat(h1s)
            c1s         = torch.cat(c1s)
            h = (h0s,c0s,h1s,c1s)
            players[0].ppo_update(states, actions, log_probs, returns, advantage, h)

            log_probs  = []
            values     = []
            values_    = []
            states     = []
            actions    = []
            rewards    = []
            masks      = []
            h0s         = []
            c0s         = []
            h1s         = []
            c1s         = []

            ppo_counter = 0
        else:
            d = False

        if done:
            break

    scores_deque.append(score if score == 1 else 0)       # save most recent score
    aveScore_deque.append(100*np.mean(scores_deque))

    plot(aveScore_deque)
    eps = max(eps_end, eps_decay*eps) # decrease epsilon
    print('Episode {}\tAverage Score: {:.4f}\tEpisode Score: {:.4f}'.format(i_episode, np.mean(scores_deque), score))
    torch.save(players[0].qnetwork_policy.state_dict(), 'checkpoint_PPO.pth')

    if i_episode>100 and np.mean(scores_deque)>=1.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
        torch.save(players[0].qnetwork_policy.state_dict(), 'checkpoint_PPO_1.0.pth')
        break