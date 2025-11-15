# /src/model_strong.py
# Enhanced model architecture for stronger play
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Enhanced residual block with squeeze-and-excitation"""
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Squeeze-and-Excitation block for better feature selection
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # Apply SE
        se_weight = self.se(x)
        x = x * se_weight
        
        x += residual
        x = F.relu(x)
        return x

class AttentionBlock(nn.Module):
    """Self-attention block for long-range dependencies"""
    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate query, key, value
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, H * W)
        value = self.value_conv(x).view(batch_size, -1, H * W)
        
        # Attention
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        out = self.gamma * out + x
        return out

class StrongAlphaZeroNet(nn.Module):
    """Much stronger architecture inspired by Leela Chess Zero"""
    def __init__(self, input_channels=32, board_size=8, num_actions=4672, 
                 num_res_blocks=20, num_attention=2, channels=256):
        super().__init__()
        
        # Initial convolutional block with more channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        # Residual tower with attention blocks interspersed
        self.res_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(channels))
            # Add attention every 5 blocks
            if i % 5 == 4 and len(self.attention_blocks) < num_attention:
                self.attention_blocks.append(AttentionBlock(channels))
        
        # Policy head - more sophisticated
        self.policy_conv = nn.Conv2d(channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc1 = nn.Linear(32 * board_size * board_size, 512)
        self.policy_fc2 = nn.Linear(512, num_actions)
        self.policy_dropout = nn.Dropout(0.1)
        
        # Value head - deeper
        self.value_conv = nn.Conv2d(channels, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 512)
        self.value_fc2 = nn.Linear(512, 256)
        self.value_fc3 = nn.Linear(256, 1)
        self.value_dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Initial conv
        x = self.conv_block(x)
        
        # Residual tower with attention
        att_idx = 0
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            # Apply attention if it's time
            if i % 5 == 4 and att_idx < len(self.attention_blocks):
                x = self.attention_blocks[att_idx](x)
                att_idx += 1
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.relu(self.policy_fc1(policy))
        policy = self.policy_dropout(policy)
        policy = self.policy_fc2(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_dropout(value)
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))
        
        return policy, value

