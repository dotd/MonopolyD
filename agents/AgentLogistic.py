import torch
from torch.autograd import Variable
import random
import math
from collections import namedtuple
import torch.nn.functional as F

dtype = torch.float
device = torch.device("cpu")
import agents.AgentBase

#use_cuda = torch.cuda.is_available()
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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

class SimpleDoubleAction(torch.nn.Module):
    '''
    This net class is only the network used in AgentLogistic
    '''
    def __init__(self, D_in, D_rep, D_out):
        super(SimpleDoubleAction, self).__init__()
        self.linear_chttgrkigeldielkcommon = torch.nn.Linear(D_in, D_rep)
        self.linear_discrete = torch.nn.Linear(D_rep, D_out)

    def forward(self, state):
        self.h_rep = self.linear_common(state).clamp(min=0)
        self.action_discrete = self.linear_discrete(self.h_rep)
        return self.action_discrete


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

class AgentLogistic(agents.AgentBase):
    def __init__(self, random):
        self.random = random
        self.steps_done = 0
        self.memory = ReplayMemory(10000)

    def init_agent(self, dim_state, rep_size, dim_action):
        self.dim_state = dim_state
        self.rep_size = rep_size
        self.dim_action = dim_action
        self.model = SimpleDoubleAction(self.dim_state, self.rep_size, self.dim_action)
        self.optimizer = torch.optim.RMSprop(self.model.parameters())

    def predict(self, state):
        '''
        0-Nothing
        1-Buy
        '''
        Q = self.model(Variable(state, volatile=True).type(FloatTensor)).data
        action = Q.max(1)[1].view(1, 1)
        return action

    def choose_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                  math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            action = self.predict(state)
        else:
            action = LongTensor([[random.randrange(2)]])
        return action

    def optimize_model(self):
        global last_sync
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
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
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


