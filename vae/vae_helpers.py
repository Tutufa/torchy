import torch
from torch.nn import functional


def vae_loss(x_hat, x, mean, log_var):
    reconstruction_loss = functional.binary_cross_entropy(x_hat, x, reduction='sum')
    latent_reg = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + latent_reg


def vae_train_step():
    pass
