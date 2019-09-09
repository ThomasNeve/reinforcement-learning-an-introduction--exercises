import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import time

#github source: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

env = gym.make('CartPole-v0')

EPISODES = 100000
RENDER_INTERVAL = 5000
RENDER_REAL_TIME = False

#observation = [cart_position=[-4.8,4.8], cart_velocity=[inf, inf], pole angle=[-24,24]deg, pole velocity at tip=[-Inf,Inf]]
BUCKET_SIZE = [30, 30, 30, 30] 
LIMIT_VAR = [
    (env.observation_space.low[0], env.observation_space.high[0]),
    (-4, 4),
    (env.observation_space.low[2], env.observation_space.high[2]),
    (-5, 5)
]
WINDOW_SIZE = []

ACTION_SPACE = env.action_space.n
real_size = [(size+2) if size!=1 else 1 for size in BUCKET_SIZE]
Q_values = np.zeros(real_size + [ACTION_SPACE])

LR = 0.1
DISCOUNT = 0.95

#goal is specific for every environment
SUCCES_COND = [(-2.4, 2.4), (-np.inf, np.inf), (-12, 12), (-np.inf, np.inf)]

#save parameters
SAVE_MODEL = [1, 1000, 5000, 25000] #final model automatically saved
SAVE_DIR = './models/model_[{}, {}, {}, {}]/{}'.format(BUCKET_SIZE[0], BUCKET_SIZE[1], BUCKET_SIZE[2], BUCKET_SIZE[3], time.time())

#setup plot
PLOT_REAL_TIME = False
PLOT_INTERVAL = 1000
PLOT_AVG = 1000

SAVE_PLOT = True

episodes = np.array([])
rewards = np.array([])

if PLOT_REAL_TIME:
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(episodes, rewards, 'b-')


def plot(episode):
    
    global episodes
    global rewards

    ax.plot(episodes, rewards, 'b-')

    line1.set_xdata(episodes)
    line1.set_ydata(rewards)

    fig.canvas.draw()


def discretize():

    for i in range(len(BUCKET_SIZE)):

        if BUCKET_SIZE[i] != 1:
            WINDOW_SIZE.append( (LIMIT_VAR[i][1]-LIMIT_VAR[i][0]) / BUCKET_SIZE[i] )
        else:
            WINDOW_SIZE.append(1)



def checkGoal(state):
    
    goal = True

    for i in range(len(state)):

        if SUCCES_COND[i][0]>state[i] or SUCCES_COND[i][1]<state[i]:
            goal = False

    return goal


def stateIndex(state_observation):

    # index = (state_observation - env.observation_space.low)/WINDOW_SIZE
    # index[BUCKET_SIZE == 1] = 1 #environment variable is not discretized

    index = np.array([])

    for i in range(len(state_observation)):
        
        if BUCKET_SIZE[i] != 1:

            if  LIMIT_VAR[i][0] <= state_observation[i] <= LIMIT_VAR[i][1]:

                index = np.append(index, 1 + (state_observation[i] - LIMIT_VAR[i][0])/WINDOW_SIZE[i])

            elif state_observation[i] > LIMIT_VAR[i][1]:

                index = np.append(index, real_size[i]-1)
            
            else:

                index = np.append(index, 0)

        else:
            index = np.append(index, 0)

    return tuple(index.astype(int))


def chooseAction(discrete_state):

    epsilon = 0.05

    if np.random.random() > epsilon:
        # Get action from Q table
        action = np.argmax(Q_values[discrete_state])
    else:
        # Get random action
        action = np.random.randint(0, ACTION_SPACE)

    return action


#Start of algorithm


#setup save directory
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

episode_rewards = np.array([])

discretize()

for episode in tqdm(range(EPISODES), ascii=True, desc='episode'):

    observation = env.reset()
    discrete_state = stateIndex(observation)

    episode_reward = 0

    done = False

    while not done:

        action = chooseAction(discrete_state)
        observation, reward, done, _ = env.step(action)

        episode_reward += reward

        if RENDER_REAL_TIME and episode%RENDER_INTERVAL == 0:
            env.render()

        if not done:

            new_discrete_state = stateIndex(observation)
            Q_old = Q_values[discrete_state + (action,)]
            Q_max_future = Q_values[new_discrete_state].max()
            
            Q_new = (1-LR)*Q_old + LR*(reward + DISCOUNT*Q_max_future)

            Q_values[discrete_state + (action,)] = Q_new
            
        elif checkGoal(observation):

            new_discrete_state = stateIndex(observation)

            Q_old = Q_values[discrete_state + (action,)]
            Q_max_future = Q_values[new_discrete_state].max()

            Q_new = (1-LR)*Q_old + LR*(reward + DISCOUNT*Q_max_future)
            
            Q_values[discrete_state + (action,)] = Q_new


        discrete_state = new_discrete_state

    #log data

    if episode%PLOT_AVG == 0:
        
        avg_reward = episode_rewards.sum()/PLOT_INTERVAL        

        episodes = np.append(episodes, episode)
        rewards = np.append(rewards, avg_reward)

        episode_rewards = []

    if (episode%PLOT_INTERVAL == 0) and PLOT_REAL_TIME:
        plot(episode)

    if episode in SAVE_MODEL:
        np.save(SAVE_DIR+'/model_at_{}_episodes_trained'.format(episode), Q_values)

    episode_rewards = np.append(episode_rewards, episode_reward)

env.close()

if SAVE_PLOT:
    np.save(SAVE_DIR+'/rewards_episodes_trained', np.array([episodes, rewards]))

np.save(SAVE_DIR+'/model_at_{}_episodes_trained'.format(EPISODES), Q_values)

plt.figure()
plt.plot(episodes, rewards)
plt.show()
