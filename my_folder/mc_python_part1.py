from environments.gridworld import GridWorld

import numpy as np

env = GridWorld(3, 4)

# Define terminal states and obstacles
state_matrix = np.zeros(shape=(3, 4))
state_matrix[0, 3] = 1
state_matrix[1, 3] = 1
state_matrix[1, 1] = -1

# Define the reward matrix
reward_matrix = np.full((3, 4), -0.04)
reward_matrix[0, 3] = 1
reward_matrix[1, 3] = -1

# Define transition matrix
transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                              [0.1, 0.8, 0.1, 0.0],
                              [0.0, 0.1, 0.8, 0.1],
                              [0.1, 0.0, 0.1, 0.8]])

# Define the policy matrix
# 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, NaN=Obstacle, -1=NoAction
# This is the optimal policy for world with reward=-0.04
policy_matrix = np.array([[1, 1, 1, -1],
                          [0, np.NaN, 0, -1],
                          [0, 3, 3, 3]])

# Set the matrices. Practically it just checks the shape of the matrix
env.setStateMatrix(state_matrix)
env.setRewardMatrix(reward_matrix)
env.setTransitionMatrix(transition_matrix)

# Reset the environment
observation = env.reset()
# Display the world printing on terminal
env.render()

# Run an episode
for _ in range(1000):
    action = policy_matrix[observation[0], observation[1]]
    observation, reward, done = env.step(action)
    print("")
    print("ACTION: " + str(action))
    print("REWARD: " + str(reward))
    print("DONE: " + str(done))
    env.render()
    if done:
        break
