import copy
import numpy as np
from matplotlib import pyplot as plt

def generate_cliffworld():
    # ======================================================================
    # Simplified CliffWorld
    # ----------------------------------------------------------------------
    # -------------------         4 is the goal state
    # | 4 | 9 | 14 | 19 |         20 is terminal state reached only via the state 4
    # -------------------         3, 2, 1 are chasms
    # | 3 | 8 | 13 | 18 |         uniform initial state distribution
    # -------------------
    # | 2 | 7 | 12 | 17 |         all transitions are deterministic
    # -------------------         Actions: 0=down, 1=up, 2=left, 3=right
    # | 1 | 6 | 11 | 16 |
    # ------------------------    rewards are all zeros except at chasms (-100)
    # | 0 | 5 | 10 | 15 | 20 |    reward for going into the goal state is +1
    # ------------------------
    # ----------------------------------------------------------------------
    P = np.zeros((21, 4, 21))
    for state_idx in range(21):
        for action_idx in range(4):
            if state_idx in [1, 2, 3]:  # chasms: reset to start state 0
                new_state_idx = 0
            elif state_idx == 4:  # goal state: agent always goes to 20
                new_state_idx = 20
            elif state_idx == 20:  # terminal state
                new_state_idx = 20
            else:  # move according to the deterministic dynamics
                x_new = x_old = state_idx // 5
                y_new = y_old = state_idx % 5
                if action_idx == 0:  # Down
                    y_new = np.clip(y_old - 1, 0, 4)
                elif action_idx == 1:  # Up
                    y_new = np.clip(y_old + 1, 0, 4)
                elif action_idx == 2:  # Left
                    x_new = np.clip(x_old - 1, 0, 3)
                elif action_idx == 3:  # Right
                    x_new = np.clip(x_old + 1, 0, 3)
                new_state_idx = 5 * x_new + y_new

            P[state_idx, action_idx, new_state_idx] = 1

    r = np.zeros((21, 4))
    r[1, :] = r[2, :] = r[3, :] = -100  # negative reward for falling into chasms
    r[4, :] = +1  # positive reward for finding the goal terminal state


    rho = np.ones(21) / 21

    P_CliffWorld = copy.deepcopy(P)
    r_CliffWorld = copy.deepcopy(r)
    rho_CliffWorld = copy.deepcopy(rho)

    state_space = np.arange(21)
    action_space = np.arange(4)

    return state_space, action_space, rho_CliffWorld, P_CliffWorld, r_CliffWorld

def v_star():
    return np.array([
        5.314410000000001633e-01,
        -9.952170309999999631e+01,
        -9.952170309999999631e+01,
        -9.952170309999999631e+01,
        1.000000000000000000e+00,
        5.904900000000001814e-01,
        6.561000000000001275e-01,
        7.290000000000000924e-01,
        8.100000000000000533e-01,
        9.000000000000000222e-01,
        5.314410000000001633e-01,
        5.904900000000001814e-01,
        6.561000000000001275e-01,
        7.290000000000000924e-01,
        8.100000000000000533e-01,
        4.782969000000001358e-01,
        5.314410000000001633e-01,
        5.904900000000001814e-01,
        6.561000000000001275e-01,
        7.290000000000000924e-01,
        0.000000000000000000e+00,
    ])

def value_iteration(P, r, gamma, K, v_star, theta=1e-4):
    # Initialize value function
    num_states, num_actions, _ = P.shape
    v = np.zeros(num_states)
    gaps = []
        
    for k in range(K):
        v_old = copy.deepcopy(v)

        for s in range(num_states):
            q_list = np.zeros(num_actions)
            for a in range(num_actions):
                p = P[s, a, :]
                q_list[a] = r[s, a] + gamma * np.sum(p * v_old)
            v[s] = np.max(q_list)
        

        # Calculate vpik
        policyk = np.zeros((num_states, num_actions)) 
        for s in range(num_states):
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                q_values[a] = r[s, a] + gamma * np.sum(P[s, a, :] * v)
            policyk[s, np.argmax(q_values)] = 1
        rpi = np.zeros(num_states)
        Ppi = np.zeros((num_states, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                rpi[s] += r[s, a] * policyk[s, a]
                Ppi[s, :] += P[s, a, :] * policyk[s, a]
        vpik = np.linalg.inv(np.eye(num_states) - gamma * Ppi) @ rpi

        # Check convergence
        gap = np.max(np.abs(vpik - v_star))
        # gap = np.max(np.abs(v - v_star))
        gaps.append(gap)
        if gap < theta:
            print(f"Converged at iteration {k}")
            break

    # Find the optimal policy
    policy = np.zeros(num_states)
    for s in range(num_states):
        q_values = np.zeros(num_actions)
        for a in range(num_actions):
            q_values[a] = r[s, a] + gamma * np.sum(P[s, a, :] * v)
        policy[s] = np.argmax(q_values)
         
    return v, policy, gaps


def policy_iteration(P, r, gamma, K, v_star, theta=1e-4):
    # Initialize value function
    num_states, num_actions, _ = P.shape
    v = np.zeros(num_states)
    policy = np.zeros((num_states, num_actions)) 
    initial_policy = (1.0/num_actions) * np.ones((num_states, num_actions))
    gaps = []
    # print(initial_policy)

    for k in range(K):
        # Policy evaluation
        v_old = copy.deepcopy(v)
        rpi = np.zeros(num_states)
        Ppi = np.zeros((num_states, num_states))

        if k == 0:
            for s in range(num_states):
                for a in range(num_actions):
                    rpi[s] += r[s, a] * initial_policy[s, a]
                    Ppi[s, :] += P[s, a, :] * initial_policy[s, a]
        else:
            for s in range(num_states):
                for a in range(num_actions):
                    rpi[s] += r[s, a] * policy[s, a]
                    Ppi[s, :] += P[s, a, :] * policy[s, a]
        
        v = np.linalg.inv(np.eye(num_states) - gamma * Ppi) @ rpi  # .reshape((21, 1))

        # Policy improvement  
        policy = np.zeros((num_states, num_actions)) 
        for s in range(num_states):
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                q_values[a] = r[s, a] + gamma * np.sum(P[s, a, :] * v)
            policy[s, np.argmax(q_values)] = 1

        # Check convergence
        gap = np.max(np.abs(v - v_star))
        gaps.append(gap)
        # print(f"At iteration {k} with gap {gap}")
        if gap < theta:
            print(f"Converged at iteration {k}")
            break

    policy = np.argmax(policy, axis=1)    

    return v, policy, gaps

def plot_gap(gaps, name):
    k = len(gaps)
    plt.figure()
    plt.plot(np.arange(k), gaps)
    plt.xlabel("Iteration")
    plt.ylabel("$\|v - v^*\|_\infty$")
    plt.title(f"Convergence of {name}")
    plt.show()


if __name__ == "__main__":
    state_space, action_space, rho, P, r = generate_cliffworld()
    gamma = 0.9
    K = 100
    v_star = v_star()
    # print(f"The optimal value function is: {v_star}. \n")

    # value iteration
    v_vi, pi_vi, gaps_vi = value_iteration(P, r, gamma, K, v_star, theta=1e-4)
    # print(f"The optimal value function from value iteration is: {v_vi}. \n")
    print(f"The optimal policy from value iteration is {pi_vi}. \n")

    # policy iteration
    v_pi, pi_pi, gaps_pi = policy_iteration(P, r, gamma, K, v_star, theta=1e-4)
    # print(f"The optimal value function from policy iteration is: {v_pi}. \n")
    print(f"The optimal policy from policy iteration is {pi_pi}. \n")
                                           
    # plotting
    plot_gap(gaps_vi, "value iteration")
    # plot_gap(gaps_pi, "policy iteration")