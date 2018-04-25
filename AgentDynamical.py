import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
from ReplayMemory import ReplayMemory
from ReplayMemory import Transition
from collections import OrderedDict

dtype = torch.FloatTensor

class AgentDynamical():
    def __init__(self, sizes_middle_net, activations, input_size, input_activation, output_size, output_activation):
        self.input_size = input_size
        self.input_activation = input_activation
        self.output_size = output_size
        self.output_activation = output_activation
        self.sizes_middle_net = sizes_middle_net
        self.activations = activations
        self.generate_initial_model()

    def generate_initial_model(self):
        sizes_middle_net = [self.input_size, self.output_size]
        self.sizes = [self.input_size, *self.sizes_middle_net, self.output_size]
        self.activations = [self.input_activation, *self.activations, self.output_activation]
        self.num_layers = len(self.sizes)-1

        self.layers = []
        self.W_and_b = []
        for layer_idx in range(self.num_layers):
            W_and_b = torch.nn.Linear(self.sizes[layer_idx], self.sizes[layer_idx+1])
            self.layers.append( ("layer"+str(layer_idx) , W_and_b) )
            self.W_and_b.append(W_and_b)

            if self.activations[layer_idx] is not None:
                self.layers.append( ("activation" + str(layer_idx) ,  self.activations[layer_idx]) )
        self.layers_values = [x[1] for x in self.layers]
        self.model = torch.nn.Sequential(*self.layers_values)

    def predict(self, state):
        return self.model(state)

    def __str__(self):
        strs = []
        strs.append(str(self.model))
        #strs.append(str(list(self.model.parameters())))
        return "\n".join(strs)

    def get_weights_and_biases(self, index):
        W = self.W_and_b[index].weight
        b = self.W_and_b[index].bias
        return W,b

    def increase_input(self, new_size):
        W,b = self.get_weights_and_biases(0)
        addition_size = new_size - W.size()[1]
        Wnew = torch.cat(( W, Variable(torch.zeros([W.size()[0],addition_size])) ), 1)
        list(self.model.parameters())[0].data = Wnew.data
        return


