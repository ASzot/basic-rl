import random

import gym

import torch
import torch.nn as nn
import torch.optim as optim



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 32)
        self.fc2 = nn.Linear(32, n_actions)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.tanh(self.fc2(x))



batch_size = 128
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
target_update = 10

policy_net = DQN()
target_net = DQN()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select(state):
    global steps_done

    sample = rando.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            result = policy_net(state).max(1)
            import pdb; pdb.set_trace()
            return result
    else:
        return torch.tensor([[random.randrange(2)]], dtype=torch.long)


def train():
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)

    batch = Transition(*zip(*transitions))

