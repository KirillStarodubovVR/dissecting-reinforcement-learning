import numpy as np
import matplotlib.pyplot as plt


def return_state_utility(v, T, u, reward, gamma):
    """Return state utility
    v - the state vector
    T - transition matrix
    u - utility vector
    reward - reward for the state
    gamma - discount factor"""

    action_array = np.zeros(4)
    for action in range(0, 4):
        # print(v, "\n")
        # print(T[:, :, action], "\n")
        # print(np.dot(v, T[:, :, action])[0][:4], "\n", np.dot(v, T[:, :, action])[0][4:8], "\n", np.dot(v, T[:, :, action])[0][8:12], "\n")
        # print(np.sum(np.multiply(u, np.dot(v, T[:, :, action]))), "\n")

        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:, :, action])))

    return reward + gamma * np.max(action_array)


def main_part1():
    #  Starting state vector
    #  The agent starts from (1,1)
    v = np.array([[0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0]])

    #  Transition matrix
    path_to_T = "C:\\Users\\SKG\\GitHub\\dissecting-reinforcement-learning\\src\\1\\T.npy"
    T = np.load(path_to_T)

    #  At this example it is magical utility vector
    u = np.array([[0.812, 0.868, 0.918, 1.0,
                   0.762, 0.0, 0.660, -1.0,
                   0.705, 0.655, 0.611, 0.388]])

    #  Define reward for state (1,1)
    reward = -0.04

    #  Assume that discount factor is 1.0
    gamma = 1.0

    # Use Bellman equation to find the utility of state (1,1)
    utility_11 = return_state_utility(v, T, u, reward, gamma)
    print("Utility of state (1,1): " + str(utility_11))


def val_Iter_Alg():
    """Value iteration algorithm"""
    #  Number of total states
    tot_states = 12
    gamma = 0.9999999  # Discount factor
    iteration = 0  # Iteration counter
    epsilon = 0.0001  # Stopping criteria

    # list with all iteration's data
    graph_list = list()

    #  Transition matrix
    path_to_T = "C:\\Users\\SKG\\GitHub\\dissecting-reinforcement-learning\\src\\1\\T.npy"
    T = np.load(path_to_T)

    r = np.array([-0.04, -0.04, -0.04, +1.0,
                  -0.04, 0.0, -0.04, -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    # Utility vector
    u = np.zeros(tot_states)
    u1 = np.zeros(tot_states)
    flag = True

    while flag:
        delta = 0
        u = u1.copy()
        iteration += 1
        graph_list.append(u)
        for s in range(tot_states):
            reward = r[s]
            # create empty vector for state
            v = np.zeros((1, tot_states))
            # put our robot in certain position
            v[0, s] = 1.0
            # Calculate utility value for current state
            u1[s] = return_state_utility(v, T, u, reward, gamma)
            # delta = max(delta, np.abs(u1[s]-u[s]))  # Stopping criteria
        delta = np.abs(u1 - u).max()
        if delta < epsilon * (1 - gamma) / gamma:
            print("=================== FINAL RESULT ==================")
            print("Iterations: " + str(iteration))
            print("Delta: " + str(delta))
            print("Gamma: " + str(gamma))
            print("Epsilon: " + str(epsilon))
            print("===================================================")
            print(np.round(u[0:4], decimals=3))
            print(np.round(u[4:8], decimals=3))
            print(np.round(u[8:12], decimals=3))
            print("===================================================")
            print(len(graph_list))
            flag = False

            plt.figure(figsize=(12, 8))
            plt.plot(np.arange(iteration), [u[0] for u in graph_list], 'b-', label="1")
            plt.plot(np.arange(iteration), [u[1] for u in graph_list], 'g-', label="2")
            plt.plot(np.arange(iteration), [u[2] for u in graph_list], 'r-', label="3")
            plt.plot(np.arange(iteration), [u[3] for u in graph_list], 'c-', label="4")
            plt.plot(np.arange(iteration), [u[4] for u in graph_list], 'm-', label="5")
            plt.plot(np.arange(iteration), [u[5] for u in graph_list], 'y-', label="6")
            plt.plot(np.arange(iteration), [u[6] for u in graph_list], 'k-', label="7")
            plt.plot(np.arange(iteration), [u[7] for u in graph_list], 'bo', label="8")
            plt.plot(np.arange(iteration), [u[8] for u in graph_list], 'go', label="9")
            plt.plot(np.arange(iteration), [u[9] for u in graph_list], 'ro', label="10")
            plt.plot(np.arange(iteration), [u[10] for u in graph_list], 'co', label="11")
            plt.plot(np.arange(iteration), [u[11] for u in graph_list], 'co', label="12")
            plt.xlabel("Iterations")
            plt.ylabel("Utility")
            plt.title("Utility(Iteration)")
            plt.legend()
            plt.show()


def return_policy_evaluation(p, u, r, T, gamma):
    """Return the policy utility
    p - policy vector
    u - utility vector
    r - reward vector
    T - transition matrix
    gamma - discount factor
    return the utility vector"""
    for s in range(12):
        if not np.isnan(p[s]):
            v = np.zeros((1, 12))  # state vector
            v[0, s] = 1.0
            action = int(p[s])
            u[s] = r[s] + np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
    return u


def return_expected_action(u, T, v):
    """Return the expected action.
    It returns an action based on the
    expected utility of doing a in state s,
    according to T and u. This action is
    the one that maximize the expected
    utility.
    u - utility vector
    T - transition matrix
    v - starting vector
    return expected action (int)
    """
    actions_array = np.zeros(4)  # up, down, left, right
    for action in range(4):
        # Expected utility of doing a in state s, according to T and u.
        actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
    return np.argmax(actions_array)


def print_policy(p, shape):
    """printing utility
    print the policy with symbols:
    ^, <, >, v - up, left, right, down
    * - terminal states
    # obstacles"""
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if (p[counter] == -1):
                policy_string += " *  "
            elif (p[counter] == 0):
                policy_string += " ^  "
            elif (p[counter] == 1):
                policy_string += " <  "
            elif (p[counter] == 2):
                policy_string += " v  "
            elif (p[counter] == 3):
                policy_string += " >  "
            elif (np.isnan(p[counter])):
                policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)


def pol_Iter_Alg():
    gamma = 0.999
    epsilon = 0.0001
    iteration = 0

    path_to_T = "C:\\Users\\SKG\\GitHub\\dissecting-reinforcement-learning\\src\\1\\T.npy"
    T = np.load(path_to_T)
    # Generate first policy randomly
    # NaN = Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    p[5] = np.NaN
    p[3] = p[7] = -1

    # Utility vectors
    u = np.array([0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0])
    # Reward vector
    r = np.array([-0.04, -0.04, -0.04, +1.0,
                  -0.04, 0.0, -0.04, -1.0,
                  -0.04, -0.04, -0.04, -0.04])
    flag = True
    while flag:
        iteration += 1
        # 1 - policy evaluation
        u_0 = u.copy()
        u = return_policy_evaluation(p, u, r, T, gamma)
        # Stopping criteria
        delta = np.absolute(u - u_0).max()
        if delta < epsilon * (1 - gamma) / gamma: flag = False

        for s in range(12):
            if not np.isnan(p[s]) and not p[s] == -1:
                v = np.zeros((1, 12))
                v[0, s] = 1.0
                a = return_expected_action(u, T, v)

                if a != p[s]: p[s] = a

        print_policy(p, shape=(3, 4))

    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(np.round(u[0:4], decimals=3))
    print(np.round(u[4:8], decimals=3))
    print(np.round(u[8:12], decimals=3))
    print("===================================================")
    print_policy(p, shape=(3, 4))
    print("===================================================")


if __name__ == "__main__":
    pol_Iter_Alg()
