# imports
import gymnasium as gym
import numpy as np
import time

env = gym.make('MountainCar-v0', render_mode="human")
env.reset()

# state where the car starts in terms of position and velocity
print("Initial state: " + str(env.state))

n_states = 10                               # number of states
min_position, max_position = -1.2, 0.6      # the default minimum and maximum positions for the car
min_velocity, max_velocity = -0.07, 0.07    # the default minimum and maximum velocity for the car
num_episodes = 250                          # the number of times you want the car to reach the flag


# Initialize Q-table
# There are n_states number tables of (n_states)x3. Each column represents the three actions and the number of
# tables and the rows represent the discrete positions and discrete velocity
q_table = np.zeros((n_states, n_states, 3))
print(q_table)  # initial table

# Define hyperparameters
gamma = 0.99    # used for discount future rewards - determinses importance given to future rewards (0-1)
alpha = 0.10    # learning rate - controls how much the agent updates the Q-values at each iteration (0-1)

'''
Convert a continuous state in the environment to a descrete state that can be used as an index in the 
        Q-table.

Parameters
----------
state :  continuous state vector that represents the position and velocity of the car

Returns - a tuple
-------
discrete_position : int

discrete_velocity : int
'''
def discretize_state(state):
    position = state[0]
    velocity = state[1]

    position_scale = n_states / (max_position - min_position)
    velocity_scale = n_states / (max_velocity - min_velocity)

    discrete_position = int((position - min_position) * position_scale)
    discrete_velocity = int((velocity - min_velocity) * velocity_scale)
    return discrete_position, discrete_velocity

# Run the Q-Learning algorithm
for episode in range(num_episodes):

    state = env.reset()
    initial = state[0]  # isolating only the position and velocity information
    discrete_state = discretize_state(initial)
    done = False
    start_time = time.time()    # meausring time taken for each episode - to check if each next episode gets faster for completion

    # while the env.step() does not return 'done' - which is when the car reaches the flag
    while not done:

        action = np.argmax(q_table[discrete_state])
        # Take action and observe next state and reward
        next_state, reward, done, info, infob = env.step(action) # variables updated when next action taken
        next_discrete_state = discretize_state(next_state)
        # Update Q-value for current state-action pair
        sample = reward + gamma * np.max(q_table[next_discrete_state]) - q_table[discrete_state + (action,)]    # action is a tuple with a single value, 
                                                                                                                # is used to index the Q-table with the current state and action
        #Update the q_table after each action
        q_table[discrete_state + (action,)] += alpha * sample
        discrete_state = next_discrete_state

    # recording the time taken to get to the flag when the gamma and n_states were varied
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken: " + str(time_taken))
    print(q_table)  # print the new q-values table

env.close()
print("End")