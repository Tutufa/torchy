import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import typing
from torchvision import utils as vutils
from tensorboardX import SummaryWriter


def base_gen_step(gen: nn.Module, discr: nn.Module, z: Variable, gen_solver: optim.Optimizer,
                  device: str, loss: typing.Callable) -> float:

    gen.zero_grad()

    fake_examples = gen(z)
    discr_fake_probs = discr(fake_examples)
    gen_loss = loss(discr_fake_probs, Variable(torch.ones(z.shape[0], 1).to(device)))
    batch_loss = gen_loss.detach().item()

    gen_loss.backward()
    gen_solver.step(closure=None)

    return batch_loss


def base_discr_step(gen: nn.Module, discr: nn.Module, z: Variable, batch: Variable, discr_solver: optim.Optimizer,
                    device: str, loss: typing.Callable) -> float:
    discr.zero_grad()

    fake_examples = gen(z)
    discr_fake_probs = discr(fake_examples)

    discr_real_probs = discr(batch)
    discr_real_loss = loss(discr_real_probs, Variable(torch.ones(batch.shape[0], 1).to(device)))
    discr_fake_loss = loss(discr_fake_probs, Variable(torch.zeros(batch.shape[0], 1).to(device)))

    discr_loss = discr_real_loss + discr_fake_loss
    batch_loss = discr_loss.detach().item()

    discr_loss.backward()
    discr_solver.step(closure=None)

    return batch_loss


bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()


# Simple GAN (CE loss)
def bce_gen_step(gen: nn.Module, discr: nn.Module, z: Variable, gen_solver: optim.Optimizer, device: str,
                 loss: typing.Callable=bce_loss) -> float:
    batch_loss = base_gen_step(gen, discr, z, gen_solver, device, loss=loss)
    return batch_loss


def bce_discr_step(gen: nn.Module, discr: nn.Module, z: Variable, batch: Variable, discr_solver: optim.Optimizer,
                   device: str, loss: typing.Callable=bce_loss) -> float:
    batch_loss = base_discr_step(gen, discr, z, batch, discr_solver, device, loss=loss)
    return batch_loss


# LS-GAN (LS loss)
def ls_gen_step(gen: nn.Module, discr: nn.Module, z: Variable, gen_solver: optim.Optimizer, device: str,
                loss: typing.Callable=mse_loss) -> float:
    batch_loss = base_gen_step(gen, discr, z, gen_solver, device, loss=loss)
    return batch_loss


def ls_discr_step(gen: nn.Module, discr: nn.Module, z: Variable, batch: Variable, discr_solver: optim.Optimizer,
                  device: str, loss: typing.Callable=mse_loss) -> float:
    batch_loss = base_discr_step(gen, discr, z, batch, discr_solver, device, loss=loss)
    return batch_loss


# W-GAN-GP (Wasserstein loss)
def wass_gen_step():
    pass


def wass_discr_step():
    pass


def gan_train_step(gen_step: typing.Callable, discr_step: typing.Callable, gen: nn.Module, discr: nn.Module,
                   batch: Variable, z: Variable, gen_solver: optim.Optimizer, discr_solver: optim.Optimizer,
                   device: str, step: int, logger: SummaryWriter) -> None:
    # generator step
    gen_loss = gen_step(gen, discr, z, gen_solver, device)
    # discriminator update
    discr_loss = discr_step(gen, discr, z, batch, discr_solver, device)
    # logging
    log_gan_step(step, gen, gen_loss, discr_loss, z.shape[1], device, logger)


def log_gan_step(step: int, gen: nn.Module, gen_loss: float, discr_loss: float, z_dim: int, device: str,
                 logger: SummaryWriter, scalar_period: int=100, img_period: int=250) -> None:
    if step % scalar_period == 0:
        logger.add_scalars('gan', {'gen_loss': gen_loss,
                                   'discr_loss': discr_loss}, step)

    if step % img_period == 0:
        gen.eval()
        z = Variable(torch.rand((10 * 10, z_dim)).to(device))
        fake_examples = gen(z)
        x = vutils.make_grid(fake_examples, nrow=10, normalize=True, scale_each=True)
        logger.add_image('Samples', x, step)

        gen.train()
