import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

import time

EXP_RENT_LOC1 = 3
EXP_RENT_LOC2 = 4

EXP_RETURN_LOC1 = 3
EXP_RETURN_LOC2 = 2

MOVE_COST = 2
RENT_REWARD = 10

MAX_MOVED = 5

GAMMA = 0.9

MAX_CARS = 20
state_vals = np.zeros((MAX_CARS+1, MAX_CARS+1)) #row is the cars at the first location, columns at the second location
policy = np.zeros((MAX_CARS+1, MAX_CARS+1), dtype=np.int) #move cars form the first to the second location

actions = np.arange(-MAX_MOVED, MAX_MOVED+1)

clamp = lambda value: max(min(value, MAX_CARS-1), 0)

#max expected number of cars rented and returned
N = 12

poisson_cache = {}

def poisson(n, lam):

    global poisson_cache

    if lam not in poisson_cache.keys():
        poisson_cache[lam] = {}
        poisson_cache[lam][n] = scipy.stats.poisson.pmf(n, lam)

    if n not in poisson_cache[lam].keys():
        poisson_cache[lam][n] = scipy.stats.poisson.pmf(n, lam)

    return poisson_cache[lam][n]

def value_estimate(state, action, simple_model=False):
    
    cars_loc1 = state[0]
    cars_loc2 = state[1]
    
    #take action, move cars
    cars_loc1 = clamp(cars_loc1-action)
    cars_loc2 = clamp(cars_loc2+action)
    
    value = 0
    value += abs(action)*(-1)*MOVE_COST

    #rent out cars
    for rented_loc1 in range(N):
        for rented_loc2 in range(N):
            
            prob_rent = poisson(rented_loc1, EXP_RENT_LOC1)*poisson(rented_loc2, EXP_RENT_LOC2)

            valid_rented_loc1 = min(cars_loc1, rented_loc1)
            valid_rented_loc2 = min(cars_loc2, rented_loc2)

            reward = RENT_REWARD*(valid_rented_loc1+valid_rented_loc2)

            remaining_cars_loc1 = cars_loc1 - valid_rented_loc1
            remaining_cars_loc2 = cars_loc2 - valid_rented_loc2
            
            if not simple_model:

                for return_loc1 in range(N):
                    for return_loc2 in range(N):

                        prob_return = poisson(return_loc1, EXP_RETURN_LOC1)*poisson(return_loc2, EXP_RENT_LOC2)

                        remaining_cars_loc1 = clamp(remaining_cars_loc1 + return_loc1)
                        remaining_cars_loc2 = clamp(remaining_cars_loc2 + return_loc2)


                        prob = prob_rent*prob_return

                        value += prob*(reward + GAMMA*state_vals[remaining_cars_loc1, remaining_cars_loc2])
            else:

                return_loc1 = EXP_RETURN_LOC1
                return_loc2 = EXP_RETURN_LOC2

                remaining_cars_loc1 = clamp(remaining_cars_loc1 + return_loc1)
                remaining_cars_loc2 = clamp(remaining_cars_loc2 + return_loc2)

                prob_return = 1
                prob = prob_rent*prob_return

                value += prob*(reward + GAMMA*state_vals[remaining_cars_loc1, remaining_cars_loc2])
            
    return value



def policy_eval(delta_stop = 1e-4):

    delta = delta_stop+1

    while delta > delta_stop:

        delta = 0
        old_val = state_vals.copy()

        for loc1 in range(len(state_vals)):
            for loc2 in range(len(state_vals[0])):
                
                #move cars from locaction one to location two
                action = policy[loc1, loc2]
                state = [loc1, loc2]
                
                state_vals[loc1, loc2] = value_estimate(state, action, simple_model=True)

        delta = abs(state_vals-old_val).max()

        print('max value change: {}'.format(delta))


def policy_improvement():

    policy_stable = True

    for loc1 in range(len(state_vals)):
            for loc2 in range(len(state_vals[0])):

                policy_old = policy[loc1, loc2]

                #generate actions
                V_opt = -np.inf
                policy_new = 0

                for action in actions:

                    if (0 <= action <= loc1) or (-loc2 <= action <= 0): 
                        
                        state = [loc1, loc2]

                        V = value_estimate(state, action, simple_model=True)

                        if V > V_opt:
                            V_opt = V
                            policy_new = action

                policy[loc1, loc2] = policy_new

                if policy_old != policy_new:
                    policy_stable = False

    print('policy stable {}'.format(policy_stable))

    return policy_stable



if __name__ == '__main__':

    policy_stable = False

    while not policy_stable:

        policy_eval()
        policy_stable = policy_improvement()

    print('final state and policy:')

    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    fig1 = sns.heatmap(np.flipud(policy), ax=ax1)
    fig1.set_xlabel('# of cars at the second location')
    fig1.set_ylabel('# of cars at the first location')
    fig1.set_yticks(list(reversed(range(21))))
    fig1.set_title('optimal policy')

    fig2 = sns.heatmap(np.flipud(state_vals), ax=ax2)
    fig2.set_xlabel('# of cars at the second location')
    fig2.set_ylabel('# of cars at the first location')
    fig2.set_yticks(list(reversed(range(21))))
    fig2.set_title('state values')

    plt.show()
    fig.savefig('figure_car_rental.png')










