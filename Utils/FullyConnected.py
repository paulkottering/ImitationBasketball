import torch
import torch.nn.functional as F
from torch import nn

class FullConnec(nn.Module):
    def __init__(self):
        # call constructor from superclass
        super().__init__()

        # define network layers
        self.fc1 = nn.Linear(12, 30)
        self.fc2 = nn.Linear(30, 70)
        self.fc3 = nn.Linear(70, 40)
        self.fc4 = nn.Linear(40, 10)

    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x