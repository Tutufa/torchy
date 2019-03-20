import torch
import torch.nn as nn
from torch.nn import functional
from torch import optim
from torch.autograd import Variable
from torchvision import utils as vutils
from tensorboardX import SummaryWriter
import typing


def vae_loss(x_hat: Variable, x: Variable, mean: Variable, log_var: Variable) -> Variable:
    reconstruction_loss = functional.binary_cross_entropy(x_hat, x, reduction='sum')
    latent_reg = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + latent_reg


def vae_train_step(model: nn.Module, batch: Variable, loss: typing.Callable, solver: optim.Optimizer) -> float:
    model.zero_grad()

    batch_hat, mean, log_var = model(batch)

    vae_batch_loss = loss(batch_hat, batch, mean, log_var)
    batch_loss = vae_batch_loss.detach().item()

    vae_batch_loss.backward()
    solver.step(closure=None)

    return batch_loss


def vae_eval_step(model: nn.Module, batch: Variable, loss: typing.Callable) -> float:
    model.eval()

    batch_hat, mean, log_var = model(batch)

    vae_batch_loss = loss(batch_hat, batch, mean, log_var)
    batch_loss = vae_batch_loss.detach().item()

    model.train()
    return batch_loss


def log_vae_step(step: int, logger: SummaryWriter, model: nn.Module, eval_batch: Variable=None,
                 eval_shape: typing.Tuple[int, int, int]=(1, 28, 28),
                 loss: float=None, train: bool=True, img: bool=False) -> None:

    if img:
        model.eval()
        batch_hat, _, _ = model(eval_batch)

        x = vutils.make_grid(batch_hat.view((-1, ) + eval_shape), nrow=10, normalize=True, scale_each=True)
        logger.add_image('Samples', x, step)

        model.train()
    else:
        if train:
            logger.add_scalars('vae', {'train_loss': loss}, step)
        else:
            logger.add_scalars('vae', {'eval_loss': loss}, step)
