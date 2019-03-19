import torch
import torch.nn as nn
from torch.nn import functional


class MNISTEncoder(nn.Module):
    def __init__(self):
        super(MNISTEncoder, self).__init__()

        self.fc = nn.Linear(784, 392)
        self.mean = nn.Linear(392, 24)
        self.log_var = nn.Linear(392, 24)

    def forward(self, x):
        h = functional.relu(self.fc(x))
        return self.mean(h), self.log_var(h)


class MNISTDecoder(nn.Module):
    def __init__(self):
        super(MNISTDecoder, self).__init__()

        self.fc_1 = nn.Linear(24, 392)
        self.fc_2 = nn.Linear(392, 784)

    def forward(self, z):
        h = functional.relu(self.fc_1(z))
        return torch.sigmoid(self.fc_2(h))


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        z = torch.randn_like(std)
        return z * std + mean

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), mean, log_var


