from torch import nn

class ChessActor(nn.Module):
    def __init__(self, n_obs, n_act):
        super(ChessActor, self).__init__()
        self.fc1 = nn.Linear(n_obs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_act)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        
        return x
        