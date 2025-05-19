import torch
import torch.nn as nn
import torch.nn.functional as F

class DeterministicNet(nn.Module):
    """
    Deterministic (non-Bayesian) CNN for MNIST with structure mirroring BayesianMnistNet.
    """
    def __init__(
            self, 
            in_channels= 3,
            input_size = (32, 32),
            p_mc_dropout = None,
            hidden_dim = 128,
            num_classes = 10,
        ):
        super().__init__()
        self.p_mc_dropout = p_mc_dropout
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        # Make a dummy pass to know the dimension of input to fc1
        with torch.no_grad():
            H, W = input_size
            dummy = torch.zeros(1, in_channels, H, W)
            x = F.max_pool2d(self.conv1(dummy), 2)
            x = F.relu(x)
            x = F.max_pool2d(self.conv2(x), 2)
            x = F.relu(x)
            n_flat = x.view(1, -1).size(1)

        # now we can create fc layers with the correct input size
        self.fc1 = nn.Linear(n_flat, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        if self.p_mc_dropout is not None:
            x = F.dropout2d(x, p=self.p_mc_dropout, training=self.training)
        x = F.relu(F.max_pool2d(x, 2))

        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.p_mc_dropout is not None:
            x = F.dropout(x, p=self.p_mc_dropout, training=self.training)

        x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x

class Ensemble(nn.Module):
    def __init__(self, models):
        self.n_models = len(models)
        self.models = models

    @torch.no_grad()
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return outputs.mean(0)