import numpy as np
import gym

env = gym.make('FrozenLake8x8-v0')

Q = np.zeros((env.observation_space.n, env.action_space.n))
lr = 0.8
gamma = 0.95
all_rewards = []

for i in range(2000):
    s0 = env.reset()
    total_reward = 0

    for j in range(100):
        a = np.argmax(Q[s0, :] + np.random.randn(1, env.action_space.n) * (1.0 / (i+1)))

        s1, reward, done, _ = env.step(a)
        all_rewards.append(reward)

        Q[s0, a] += lr * (reward + gamma * np.max(Q[s1, :]) -
                Q[s0, a])

        if done:
            break

        s0 = s1

    if i % 100 == 0:
        # Display a running average of the rewards
        print('%i) Reward: %.5f' % (i, np.mean(all_rewards[:-1000])))


print('Learned Policy')
print(np.argmax(Q, axis=1))
