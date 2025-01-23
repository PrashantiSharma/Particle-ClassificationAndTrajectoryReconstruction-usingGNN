import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Classification Model
class ClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the Trajectory Model
class TrajectoryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrajectoryModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
