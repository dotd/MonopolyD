import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
from ReplayMemory import ReplayMemory
from ReplayMemory import Transition

class Agent:
    def __init__(self, state_dim, batch_size, action_dim, H, gamma, BATCH_SIZE):
        # The network
        self.action_dim = action_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, action_dim),
            torch.nn.ReLU(),
        )
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.gamma = gamma
        self.memory = ReplayMemory(capacity=2000)
        self.BATCH_SIZE = BATCH_SIZE

    def select_action(self, state):
        global steps_done
        sample = random.random()

        steps_done += 1
        if sample > 0.01:
            pt_state = Variable(state, volatile=True).type(torch.FloatTensor)
            return self.model(pt_state).data.max(1)[1].view(1, 1)
        else:
            return torch.FloatTensor([[random.randrange(self.action_dim)]])

    def optimize_model(self):
        global last_sync
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE).type(torch.Tensor))
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def get_Q(self, state):
        output = self.model(state)
        return output

    def step(self, state, nxt_state, reward, gamma):
        self.Qvec = self.get_Q(state)
        self.action = self.Qvec.data.max(1)[0]
        self.Q = self.Qvec[self.action]
        self.Qvec_nxt = self.get_Q(nxt_state)
        self.action_nxt = self.Qvec_nxt.data.max(1)[0]
        self.Q_next = self.Qvec[self.action_nxt]

        loss = self.loss_fn(reward + self.gamma *self.Q_next , self.Q)
