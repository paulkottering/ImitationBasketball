from torch import nn, as_tensor, tanh, square, sum, sqrt, prod
from math import log

class Player:
    def __init__(self, id, name, jersey):
        self.id = id
        self.name = name
        self.jersey = jersey
        self.positions = []
    def __str__(self):
        return f"{self.name} #{self.jersey}"

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        if input_size < output_size:
            raise Exception("Input layer cannot have lesser features than output layer for this model.")
        # for (i) cells in input layer and (o) cells in output layer create (i - o) fully-connected hidden layers
        # if input and output sizes are same, have atleast 1 hidden layer
        elif input_size == output_size:
            self.hidden_layers = nn.ModuleList([nn.Linear(input_size, input_size)])
        # each hidden layer has one less feature than the previous layer, till it converges to output size
        else:
            self.hidden_layers = nn.ModuleList([nn.Linear(i, i-1) for i in range(input_size, output_size, -1)])
    def forward(self, x):
        for hidden in self.hidden_layers:
            x = hidden(x)
        output = tanh(x)
        return output
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def __getitem__(self, key):
        return getattr(self, key)

class DLoss(nn.Module):
    def __init__(self):
        super(DLoss, self).__init__()
    def forward(self, size, scores, l_actions, e_actions):
        l, e = 0, 0
        # compute expected values of learner and expert actions
        for i in range(size):
            l += self.entropy(l_actions[i]).item()/size
            e += self.entropy(e_actions[i]).item()/size
        max = 0
        for score in scores:
            if score.item() < 0:
                continue
            s = l * -log(score.item()) + e * -log(1 - score.item())
            max = s if s > max else max
        return as_tensor(max).float().requires_grad_()
    def entropy(self, t):
        return sqrt(sum(square(t))/prod(t, 0))

class VLoss(nn.Module):
    def __init__(self):
        super(VLoss, self).__init__()
    def forward(self, actions):
        var = 0
        for action in actions:
            var += square(self.entropy(action)).item()/len(actions)
        return as_tensor(var).float().requires_grad_()
    def entropy(self, t):
        return sqrt(sum(square(t))/prod(t, 0))