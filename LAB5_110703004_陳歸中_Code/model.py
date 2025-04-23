import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN_task1(nn.Module):
    """
    fully connected nn for task 1 (CartPole)
    - Input size is the same as the state dimension; the output size is the same as the number of actions
    """
    def __init__(self, num_actions, input_dim):
        super(DQN_task1, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.network(x)
    
class DQN_task2(nn.Module):
    """
    CNN-based DQN for Task 2
    Input: stacked frames
    """
    def __init__(self, num_actions):
        super(DQN_task2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)