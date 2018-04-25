import AgentDynamical as ad
import torch
from torch.autograd import Variable

input_size = 2
input_activation = torch.nn.ReLU()
output_size = 2
output_activation = None
sizes_middle_net = [5,4]
activations = [torch.nn.ReLU()]

x = ad.AgentDynamical(sizes_middle_net = sizes_middle_net, activations = activations,
                      input_size=input_size, input_activation=input_activation,
                      output_size=output_size, output_activation=output_activation)
print(x)

for i in [0,-1]:
    W,b = x.get_weights_and_biases(i)
    print("W {}:\n{}".format(i,W))
    print("b {}:\n{}".format(i,b))

print("predict test 1:")
print(x.model(Variable(torch.FloatTensor([[1,2]]))))
print("increase network:")
x.increase_input(3)
print(x)
print("predict test 2:")
print(x.model(Variable(torch.FloatTensor([[1,2,0]]))))

