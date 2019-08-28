import gym
import numpy as np

env = gym.make('CartPole-v0')
env.reset()

#observation = [cart_position=[-4.8,4.8], cart_velocity=[inf, inf], pole angle=[-24,24]deg, pole velocity at tip=[-Inf,Inf]]

BUCKET_SIZE = [20, 1, 20, 1] #ignore velocity
ACTION_SPACE = env.action_space.n

Q_values = np.ones(BUCKET_SIZE + [ACTION_SPACE])

done = False

LR = 0.5
DISCOUNT = 0.9

def state_index(state_observation):

    index = (state_observation - env.observation_space.low)/BUCKET_SIZE
    return index.astype(int)

while not done:

    obesrvation, reward, done, _ = env.step(env.action_space.sample())
    env.render()

    print()
    print(obesrvation)
    print(reward)


env.close()


print(env.observation_space.high)
print(env.observation_space.low)

print(dir(env))
print(dir(env.reward_range))
print(dir(env.action_space))

print(env.action_space.n)