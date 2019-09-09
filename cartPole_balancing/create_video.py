import gym
import numpy as np
from gym import wrappers

import os

env = gym.make('CartPole-v0')

model_dir = './models/model_[50, 50, 50, 50]/1567524154.1553748'

save_dir = model_dir+'/videos'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

env = wrappers.Monitor(env, save_dir, force = True)

BUCKET_SIZE = [50, 50, 50, 50] 
LIMIT_VAR = [
    (env.observation_space.low[0], env.observation_space.high[0]),
    (-4, 4),
    (env.observation_space.low[2], env.observation_space.high[2]),
    (-5, 5)
]
WINDOW_SIZE = []
real_size = [(size+2) if size!=1 else 1 for size in BUCKET_SIZE]

Q_values = np.load('./models/model_[50, 50, 50, 50]/1567524154.1553748/model_at_100000_episodes_trained.npy')

def chooseAction(discrete_state):

    action = np.argmax(Q_values[discrete_state])

    return action

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

def discretize():

    for i in range(len(BUCKET_SIZE)):

        if BUCKET_SIZE[i] != 1:
            WINDOW_SIZE.append( (LIMIT_VAR[i][1]-LIMIT_VAR[i][0]) / BUCKET_SIZE[i] )
        else:
            WINDOW_SIZE.append(1)

discretize()

done = False
total_reward = 0

observation = env.reset()
discrete_state = stateIndex(observation)

while not done:

    action = chooseAction(discrete_state)
    observation, reward, done, _ = env.step(action)

    discrete_state = stateIndex(observation)
    total_reward += reward

    env.render()

    if done:

        print('end condition met with {} reward'.format(total_reward))

        break

env.close()