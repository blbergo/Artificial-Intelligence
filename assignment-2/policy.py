import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChessPolicy(nn.Module):
    def __init__(self, action_size, hidden_size=128):
        super(ChessPolicy, self).__init__()
        self.fc1 = nn.Linear(64, hidden_size)  # Changed to match flattened input size
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = x.view(-1, 64)  # Flatten the 8x8 input to 64
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)
    