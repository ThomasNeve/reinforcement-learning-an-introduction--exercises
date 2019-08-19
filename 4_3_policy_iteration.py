import numpy as np
from scipy.stats import poisson

EXP_RENT_LOC1 = 3
EXP_RENT_LOC2 = 4

EXP_RETURN_LOC1 = 3
EXP_RETURN_LOC2 = 2

MOVE_COST = 2
RENT_REWARD = 10

MAX_MOVED = 5

GAMMA = 0.9

MAX_CARS = 20
state_vals = np.zeros((MAX_CARS, MAX_CARS)) #row is the cars at the first location, columns at the second location
policy = np.zeros((MAX_CARS, MAX_CARS), dtype=np.int) #move cars form the first to the second location

actions = np.arange(-MAX_MOVED, MAX_MOVED+1)

clamp = lambda value: max(min(value, MAX_CARS-1), 0)

#max expected number of cars rented and returned
N = 12


def value_estimate(state, action):
    
    cars_loc1 = state[0]
    cars_loc2 = state[1]
    
    #take action, move cars
    cars_loc1 = clamp(cars_loc1-action)
    cars_loc2 = clamp(cars_loc2+action)
    
    reward = (-1)*abs(action*MOVE_COST)
    value = 0

    #rent out cars
    for rented_loc1 in range(N):
        for rented_loc2 in range(N):

            prob_rent = poisson.pmf(rented_loc1, EXP_RENT_LOC1)*poisson.pmf(rented_loc2, EXP_RENT_LOC2)

            valid_rented_loc1 = min(cars_loc1, rented_loc1)
            valid_rented_loc2 = min(cars_loc2, rented_loc2)

            reward += RENT_REWARD*(valid_rented_loc1+valid_rented_loc2)

            remaining_cars_loc1 = cars_loc1 - valid_rented_loc1
            remaining_cars_loc2 = cars_loc2 - valid_rented_loc2

            for return_loc1 in range(N):
                for return_loc2 in range(N):

                    prob_return = poisson.pmf(return_loc1, EXP_RETURN_LOC1)*poisson.pmf(return_loc2, EXP_RENT_LOC2)

                    remaining_cars_loc1 = clamp(remaining_cars_loc1 + return_loc1)
                    remaining_cars_loc2 = clamp(remaining_cars_loc2 + return_loc2)


                    prob = prob_rent*prob_return

                    value += prob*(reward + GAMMA*state_vals[remaining_cars_loc1, remaining_cars_loc2])

    return value



def policy_eval(delta_stop = 0.1):

    delta = delta_stop+1

    while delta > delta_stop:

        delta = 0

        for loc1 in range(len(state_vals)):
            for loc2 in range(len(state_vals[0])):

                old_val = state_vals[loc1, loc2]
                
                #move cars from locaction one to location two
                action = policy[loc1, loc2]
                state = [loc1, loc2]

                state_vals[loc1, loc2] = value_estimate(state, action)

                delta = max(delta, abs(old_val-state_vals[loc1, loc2]))

                print(delta)
                print('value', state_vals[0][0])


def policy_improvement():

    policy_stable = True

    for loc1 in range(len(state_vals)):
            for loc2 in range(len(state_vals[0])):

                #generate actions
                V_opt = 0
                policy = 0

                for action in actions:
                    
                    state = [loc1, loc2]
                    
                    V = value_estimate(state, action)

                    if V > V_opt:
                        V_opt = V
                        policy = action

                policy[loc1][loc2] = policy




                



policy_eval(delta_stop=1)
policy_improvement()











