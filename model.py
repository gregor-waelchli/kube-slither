# model.py
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=1024, output_size=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class QTrainer:
    def __init__(self, model, lr=0.001, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.model.fc[0].weight.device)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(self.model.fc[0].weight.device)
        action = torch.tensor([action], dtype=torch.long).to(self.model.fc[0].weight.device)
        reward = torch.tensor([reward], dtype=torch.float).to(self.model.fc[0].weight.device)
        done = torch.tensor([done], dtype=torch.bool).to(self.model.fc[0].weight.device)

        pred = self.model(state)
        target = pred.clone()

        Q_new = reward + (1 - done.float()) * self.gamma * torch.max(self.model(next_state))

        target[0][action] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), action, reward, np.array(next_state), done)

    def __len__(self):
        return len(self.buffer)