import torch.nn as nn
import torch.nn.functional as F
import torch

class aspectNet(nn.Module):
    def __init__(self, data_dim):
        super(aspectNet, self).__init__()
        self.fc1   = nn.Linear(data_dim, 4000)
        self.fc2   = nn.Linear(5000, 400)
        self.fc3   = nn.Linear(400, 40)
        self.fc4   = nn.Linear(40, 5)

    def forward(self, x, y):
        out = F.relu(self.fc1(x))
        out_cat = torch.cat((out, y), 1)
        assert out_cat.shape[1] == 5000
        out = F.relu(self.fc2(out_cat))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class aspectNet2(nn.Module):
    def __init__(self, data_dim):
        super(aspectNet2, self).__init__()
        self.fc1   = nn.Linear(data_dim, 4000)
        self.fc2   = nn.Linear(4000, 400)
        self.fc3   = nn.Linear(400, 40)
        self.fc4   = nn.Linear(45, 5)

    def forward(self, x, y):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out_cat = torch.cat((out, y), 1)
        assert out_cat.shape[1] == 45
        out = F.relu(self.fc4(out_cat))
        return out

class aspectNet3(nn.Module):
    def __init__(self, data_dim):
        super(aspectNet3, self).__init__()
        self.fc1   = nn.Linear(data_dim, 4000)
        self.fc2   = nn.Linear(4000, 400)
        self.fc3   = nn.Linear(405, 40)
        self.fc4   = nn.Linear(40, 5)

    def forward(self, x, y):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out_cat = torch.cat((out, y), 1)
        assert out_cat.shape[1] == 405
        out = F.relu(self.fc3(out_cat))
        out = self.fc4(out)
        return out

class aspectNet4(nn.Module):
    def __init__(self, data_dim):
        super(aspectNet4, self).__init__()
        self.fc1   = nn.Linear(data_dim, 30)
        self.fc2   = nn.Linear(35, 5)

    def forward(self, x, y):
        out = F.relu(self.fc1(x))
        out_cat = torch.cat((out, y), 1)
        assert out_cat.shape[1] == 35
        out = self.fc2(out_cat)
        return out

class aspectNet5(nn.Module):
    def __init__(self, data_dim):
        super(aspectNet5, self).__init__()
        self.fc1   = nn.Linear(data_dim, 4000)
        self.fc2   = nn.Linear(4000, 400)
        self.fc3   = nn.Linear(400, 40)
        self.fc4   = nn.Linear(45, 5)

    def forward(self, x, y):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out_cat = torch.cat((out, y), 1)
        assert out_cat.shape[1] == 45
        out = self.fc4(out_cat)
        return out