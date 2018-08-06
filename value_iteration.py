"""
Core idea of value-iterations is to compute all values of Q(s, a) and for each
state calculate the max action of Q(s, a) given the state. We then know that
V(s) = the action that maximized Q(s, a)
"""

import numpy as np
import gym


def compute_q(P, s, nA, gamma, prev_v):
    q = np.zeros(nA)
    for a in range(nA):
        q_a = []
        # Probability of next state, value of next state, reward, and is done.
        for p, s_, r, _ in P[s][a]:
            q_a.append(p * (r + gamma * prev_v[s_]))

        # Sum across all possible next states
        q[a] = sum(q_a)

    return q


def run_policy(env, policy, gamma, render):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    done = False
    while not done:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1

    return total_reward

def evaluate_policy(env, policy, gamma, n=100):
    scores = [run_policy(env, policy, gamma, render=False) for _ in range(n)]
    return np.mean(scores)



gamma = 1.0

eps = 1e-20

env = gym.make('FrozenLake8x8-v0')
nS = env.observation_space.n
nA = env.action_space.n

# Probability of reward given state and action.
P = env.env.P

v = np.zeros(nS)
for i in range(10000):
    prev_v = np.copy(v)

    for s in range(nS):
        q = compute_q(P, s, nA, gamma, prev_v)
        v[s] = max(q)

    if np.sum(np.fabs(prev_v - v)) <= eps:
        print('Converged at iteration %.2f' % i)
        break


policy = np.zeros(nS)
for s in range(nS):
    q = compute_q(P, s, nA, gamma, v)

    policy[s] = np.argmax(q)


total_reward = evaluate_policy(env, policy, gamma)

print('Average policy score %.2f' % (total_reward))
print(policy)
