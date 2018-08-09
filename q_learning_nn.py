# https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb
import random
import math
from collections import deque
import numpy as np

import gym

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from collections import namedtuple


class DummyEnv(object):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        next_state, reward, done, info = self.env.step(action[0])
        return np.expand_dims(next_state, 0), reward, done, info

    def reset(self):
        return np.expand_dims(self.env.reset(), 0)

    @property
    def observation_shape(self):
        return [1, *self.env.observation_space.shape]

    @property
    def action_space(self):
        return self.env.action_space



USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size=128):
        super().__init__()
        print('there are # states', n_states)
        self.layers = nn.Sequential(
                nn.Linear(n_states, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions)
            )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data.numpy()
        else:
            action = np.random.randint(low = 0, high = env.action_space.n,
                    size=(1,))

        return action


env = gym.make('CartPole-v0')

env = DummyEnv(env)

model = DQN(env.observation_shape[1], env.action_space.n)
if USE_CUDA:
    model = model.cuda()

optimizer = optim.Adam(model.parameters())

replay_buffer = ReplayMemory(1000)

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    q_values_next = model(next_state)

    q_value = q_values.gather(1, action).squeeze(1)

    next_q_value =  q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


log_interval = 100
losses = []
all_rewards = []
episode_reward = 0

num_frames = int(1e5)
batch_size = 32
gamma = 0.99

eps_start = 1.0
eps_end = 0.01
eps_decay = 500

state = env.reset()

for frame_idx in range(num_frames):
    # Calculate epsilon
    eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * frame_idx /
            eps_decay)

    action = model.act(state, eps)

    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.data.numpy())

    if frame_idx % log_interval == 0:
        print('Iteration %i) Loss: %.5f, Reward: %.5f' %
                (frame_idx, np.mean(losses[-log_interval:]),
                    np.mean(all_rewards[-log_interval:])))

