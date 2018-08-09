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
            state = Variable(torch.FloatTensor(state), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data.numpy()
        else:
            action = np.random.randint(low = 0, high = env.action_space.n,
                    size=(1,))

        return action


env = gym.make('CartPole-v0')
env = DummyEnv(env)

cur_model = DQN(env.observation_shape[1], env.action_space.n)
target_model = DQN(env.observation_shape[1], env.action_space.n)


if USE_CUDA:
    cur_model = cur_model.cuda()
    target_model = target_model.cuda()

optimizer = optim.Adam(cur_model.parameters())

replay_buffer = ReplayMemory(1000)

def update_target_model(cur_model, target_model):
    # Copies all of the parameters of `cur_model` to `target_model`
    target_model.load_state_dict(cur_model.state_dict())

update_target_model(cur_model, target_model)

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = cur_model(state)
    q_values_next = cur_model(next_state)
    q_state_values_next = target_model(next_state)

    q_value = q_values.gather(1, action).squeeze(1)
    q_value_next =  q_state_values_next.gather(1, torch.max(q_values_next, 1)[1].unsqueeze(1)).squeeze(1)

    expected_q_value = reward + gamma * q_value_next * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


losses = []
all_rewards = []
episode_reward = 0

# Hyperparameters
log_interval = 100
num_frames = int(1e4)
batch_size = 32
gamma = 0.99

eps_start = 1.0
eps_end = 0.01
eps_decay = 500

model_sync_interval = 100

state = env.reset()

for frame_idx in range(num_frames):
    eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * frame_idx /
            eps_decay)

    action = cur_model.act(state, eps)

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
        update_target_model(cur_model, target_model)

    if frame_idx % log_interval == 0:
        print('Iteration %i) Loss: %.5f, Reward: %.5f' %
                (frame_idx, np.mean(losses[-log_interval:]),
                    np.mean(all_rewards[-log_interval:])))

