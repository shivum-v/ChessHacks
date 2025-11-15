# /src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self, input_channels=12, board_size=8, num_actions=4672):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

        self.fc_policy = nn.Linear(128 * board_size * board_size, num_actions)
        self.fc_value = nn.Linear(128 * board_size * board_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))

        return policy, value
