import torch
import torch.nn as nn
from torch.autograd import Variable

from training_loop_mnist import z_dim
from utils import log_step


def base_gen_step(gen, discr, z, gen_solver, device, loss=None):
    gen.zero_grad()

    fake_examples = gen(z)
    discr_fake_probs = discr(fake_examples)
    gen_loss = loss(discr_fake_probs, Variable(torch.ones(z.shape[0], 1).to(device)))
    batch_loss = gen_loss.detach().item()

    gen_loss.backward()
    gen_solver.step()

    return batch_loss


def base_discr_step(gen, discr, z, batch, discr_solver, device, loss=None):
    discr.zero_grad()

    fake_examples = gen(z)
    discr_fake_probs = discr(fake_examples)

    discr_real_probs = discr(batch)
    discr_real_loss = loss(discr_real_probs, Variable(torch.ones(batch.shape[0], 1).to(device)))
    discr_fake_loss = loss(discr_fake_probs, Variable(torch.zeros(batch.shape[0], 1).to(device)))

    discr_loss = discr_real_loss + discr_fake_loss
    batch_loss = discr_loss.detach().item()

    discr_loss.backward()
    discr_solver.step()

    return batch_loss


bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()


# Simple GAN (CE loss)
def bce_gen_step(gen, discr, z, gen_solver, device, loss=bce_loss):
    batch_loss = base_gen_step(gen, discr, z, gen_solver, device, loss=loss)
    return batch_loss


def bce_discr_step(gen, discr, z, batch, discr_solver, device, loss=bce_loss):
    batch_loss = base_discr_step(gen, discr, z, batch, discr_solver, device, loss=loss)
    return batch_loss


# LS-GAN (LS loss)
def ls_gen_step(gen, discr, z, gen_solver, device, loss=mse_loss):
    batch_loss = base_gen_step(gen, discr, z, gen_solver, device, loss=loss)
    return batch_loss


def ls_discr_step(gen, discr, z, batch, discr_solver, device, loss=mse_loss):
    batch_loss = base_discr_step(gen, discr, z, batch, discr_solver, device, loss=loss)
    return batch_loss


# W-GAN-GP (Wasserstein loss)
def wass_gen_step():
    pass


def wass_discr_step():
    pass


def gan_train_step(gen_step, discr_step, gen, discr, batch, z, gen_solver, discr_solver, device, step, logger):
    # generator step
    gen_loss = gen_step(gen, discr, z, gen_solver, device)
    # discriminator update
    discr_loss = discr_step(gen, discr, z, batch, discr_solver, device)
    # logging
    log_step(step, gen, gen_loss, discr_loss, z_dim, device, logger)
