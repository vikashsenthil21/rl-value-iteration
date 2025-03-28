# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. The environment is closed with a fence, so the agent cannot leave the gridworld. The agent must determine the best actions to take from each state to maximize its reward.

## VALUE ITERATION ALGORITHM
```
1. Initialize the value function for all states to zero.
2. Iterate until the values converge, meaning changes become very small.
3. For each state, evaluate all possible actions.
4. Estimate expected rewards by considering next states and their probabilities.
5. Update the value function by selecting the best action that maximizes future rewards.
6. Repeat the process until the value function stops changing significantly.
7. Extract the optimal policy by choosing the action that leads to the highest value for each state.
8. Ensure the agent follows the best possible path to maximize rewards.
9. Used in Markov Decision Processes (MDPs) where the environment is uncertain or stochastic.
10. Guarantees finding the optimal policy, making it useful in reinforcement learning applications.
```

## VALUE ITERATION FUNCTION
### Name: VIKASH S
### Register Number: 212222240115
```
envdesc  = ['FSFF','FFHF','FHFF', 'FGFH']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 13
P = env.env.P

def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = V[s]
            action_values = []
            for a in range(len(P[s])):
                action_value = 0
                for prob, next_state, reward, done in P[s][a]:
                    action_value += prob * (reward + gamma * V[next_state] * (not done))
                action_values.append(action_value)
            V[s] = max(action_values)
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break

    pi = lambda s: np.argmax([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in P[s][a]]) for a in range(len(P[s]))])

    return V, pi
```

## OUTPUT:
## optimal policy
![image](https://github.com/user-attachments/assets/2f7b8814-f838-49f1-885a-21fa93fa3885)
## optimal value function
![image](https://github.com/user-attachments/assets/818ca053-1eef-439e-ada4-6d78d83d6eff)
## success rate for the optimal policy
![image](https://github.com/user-attachments/assets/4b8b62a1-6375-4abb-99a6-a821c3af1073)

## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
