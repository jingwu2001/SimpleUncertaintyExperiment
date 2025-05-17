import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistNet(nn.Module):
    """
    Deterministic (non-Bayesian) CNN for MNIST with structure mirroring BayesianMnistNet.
    """
    def __init__(self, p_mc_dropout=0.5):
        super().__init__()
        self.p_mc_dropout = p_mc_dropout
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)

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


