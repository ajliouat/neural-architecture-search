import torch
import torch.nn as nn
import torch.optim as optim

class RLPolicy(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(action_space, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, state):
        return self.policy_net(state)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()