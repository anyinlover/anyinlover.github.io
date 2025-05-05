import numpy as np
from collections import defaultdict
from gym_env.envs.stochastic_grid_world import StochasticGridWorldEnv


def value_iter(env, theta=0.001, discount_factor=1.0):
    def one_step_lookahead(state, V):
        A = np.zeros(len(env._action_to_direction))
        for a in env._action_to_direction:
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = defaultdict(int)
    while True:
        delta = 0
        for s in env.P:
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value

        if delta < theta:
            break

    policy = defaultdict(int)
    for s in env.P:
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s] = best_action

    return policy, V


if __name__ == "__main__":
    env = StochasticGridWorldEnv()
    policy, V = value_iter(env)
    print(policy)
    print(V)
