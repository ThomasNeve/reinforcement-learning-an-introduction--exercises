import gym
import numpy as np

#github source: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

env = gym.make('CartPole-v0')

EPISODES = 1
RENDER_INTERVAL = 50
PLOT_INTERVAL = 10

#observation = [cart_position=[-4.8,4.8], cart_velocity=[inf, inf], pole angle=[-24,24]deg, pole velocity at tip=[-Inf,Inf]]
BUCKET_SIZE = [20, 1, 20, 1] #ignore velocity
space_high = env.observation_space.high[BUCKET_SIZE==1] = 1
space_low = env.observation_space.low[BUCKET_SIZE==1] = 1
WINDOW_SIZE = (space_high-space_low)/BUCKET_SIZE

ACTION_SPACE = env.action_space.n

Q_values = np.ones(BUCKET_SIZE + [ACTION_SPACE])

LR = 0.5
DISCOUNT = 0.9

#goal is specific for every environment
SUCCES_COND = [(-2.4, 2.4), (-np.inf, np.inf), (-12, 12), (-np.inf, np.inf)]


def checkGoal(state):
    
    goal = True

    for i in range(len(state)):

        if SUCCES_COND[i][0]>state[i] or SUCCES_COND[i][1]<state[i]:
            goal = False

    return goal


def stateIndex(state_observation):

    index = (state_observation - env.observation_space.low)/WINDOW_SIZE
    index[BUCKET_SIZE == 1] = 1 #environment variable is not discretized

    return tuple(index.astype(int))

episode_rewards = np.array([])

for episode in range(EPISODES):

    observation = env.reset()
    discrete_state = stateIndex(observation)

    episode_reward = 0

    done = False

    while not done:

        action = np.argmax(Q_values[discrete_state])
        observation, reward, done, _ = env.step(action)

        episode_reward += reward

        if episode%RENDER_INTERVAL == 0:
            env.render()

        if not done:

            new_discrete_state = stateIndex(observation)

            Q_old = Q_values[discrete_state + (action,)]
            Q_max_future = Q_values[new_discrete_state]

            Q_new = (1-LR)*Q_old + LR*(reward + DISCOUNT*Q_max_future)

            Q_values[discrete_state + (action,)] = Q_new
            
        elif checkGoal():
            pass

        discrete_state = new_discrete_state


env.close()


print(env.observation_space.high)
print(env.observation_space.low)

print(dir(env))
print(dir(env.reward_range))
print(dir(env.action_space))

print(env.action_space.n)