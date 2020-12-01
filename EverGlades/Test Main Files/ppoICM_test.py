## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb
import time
import random
import pickle

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

from everglades_server import server

from multiprocessing_env import SubprocVecEnv

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

def plot(aveScore, aveScore_deque1000, aveScoreTotal):
    #plt.figure(1)
    plt.clf()        
    plt.title('ICM PPO')
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
    plt.savefig('graph_PPO-ICM_final.png', dpi=200)

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path
agent0_file = 'agents/ppoICM_agent.py'
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
def make_env():
    # returns a function which creates a single environment
    def _thunk():
        env = gym.make('everglades-v0')
        return env
    return _thunk

#env = [make_env() for i in range(8)]
#env = SubprocVecEnv(env)
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
scores = deque(maxlen=n_episodes)
scores_deque = deque(maxlen=100)   # last 100 scores
aveScore_deque = deque(maxlen=20000)

scores_deque1000 = deque(maxlen=1000) 
aveScore_deque1000 = deque(maxlen=200000) # 1000

scores_dequeTotal = deque(maxlen=200000)
aveScore_dequeTotal = deque(maxlen=200000)
eps = eps_start                    # initialize epsilon

PPO_steps = 456

device = torch.device("cpu")
#device = torch.device("cuda:0")


log_probs   = []
values      = []
states      = []
next_states = []
actions     = []
rewards     = []
masks       = []

ppo_counter = 0
max = 0
players[0].qnetwork_policy.load_state_dict(torch.load('checkpoint_PPOICM_max90kplus.pth'))

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
        actions0, value, log_prob, allActions = players[0].act(state)
        actions1 = players[1].get_action( observations[1] )
        actionsArray[0] = actions0
        actionsArray[1] = actions1 

        observations, reward, done, info = env.step(actionsArray)
        
        log_probs.append(log_prob)
        values.append(value)
        
        rewards.append(torch.FloatTensor([reward[0]]).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
        
        state = np.reshape(state, (-1, len(state)))
        state = torch.FloatTensor(state).to(device)
        
        next_state = np.reshape(observations[0], (-1, len(observations[0])))
        next_state = torch.FloatTensor(next_state).to(device)
        
        next_states.append(next_state)
        states.append(state)
        actions.append(allActions)

        #keep this line for now
        state = observations[0]
        score += reward[0]
        ppo_counter += 1

        if ppo_counter % PPO_steps == 0:
            _, next_value, _, _ = players[0].act(observations[0])

            new_rewards = players[0].Intrinsic(rewards, states, next_states, actions)
            #new_rewards = players[0].normalize_rewards(new_rewards)
            returns = players[0].compute_gae(next_value, new_rewards, masks, values)
            
            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            
            states    = torch.cat(states)
            next_states = torch.cat(next_states)

            actions   = torch.cat(actions)
            advantage = returns - values
            advantage = players[0].normalize(advantage)
            
            players[0].ppo_update(states, next_states, actions, log_probs, returns, advantage)

            log_probs   = []
            values      = []
            states      = []
            next_states = []
            actions     = []
            rewards     = []
            masks       = []

            ppo_counter = 0

        if done:
            break

    scores_deque.append(score if score == 1 else 0)       # save most recent score
    aveScore_deque.append(100*np.mean(scores_deque))

    scores_deque1000.append(score if score == 1 else 0)     # save most recent score
    aveScore_deque1000.append(100*np.mean(scores_deque1000))

    scores_dequeTotal.append(score if score == 1 else 0)
    aveScore_dequeTotal.append(100*np.mean(scores_dequeTotal))

    scores.append(score if score == 1 else 0)

    pickle_out = open("PPO-ICMdata","wb")
    pickle.dump(scores, pickle_out)
    pickle_out.close()

    plot(aveScore_deque, aveScore_deque1000, aveScore_dequeTotal)
    
    print('Episode {}\tAverage Score: {:.4f}\tEpisode Score: {:.4f}'.format(i_episode, np.mean(scores_deque), score))
    #torch.save(players[0].qnetwork_policy.state_dict(), 'checkpoint_PPOICM_max.pth')
    torch.save(players[0].qnetwork_policy.state_dict(), 'checkpoint_PPOICM_final.pth')



