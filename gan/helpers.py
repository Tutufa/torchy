import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils

import logging
py_logger = logging.getLogger('gan_training_logging')


class MNISTDataset:
    def __init__(self, path_to_data: str, batch_size: int=128):
        self.path = path_to_data
        self.bs = batch_size

    def get_iterators(self):

        with open(self.path, 'rb') as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

        train_ds = TensorDataset(x_train.view((-1, 1, 28, 28)), y_train)
        valid_ds = TensorDataset(x_valid.view((-1, 1, 28, 28)), y_valid)

        train_dl = DataLoader(train_ds, batch_size=self.bs)
        valid_dl = DataLoader(valid_ds, batch_size=self.bs)

        return train_dl, valid_dl


def log_step(step, gen, gen_loss, discr_loss, z_dim, device, logger, scalar_period=100, img_period=250):
    if scalar_period % 100 == 0:
        logger.add_scalars('gan', {'gen_loss': gen_loss,
                                   'discr_loss': discr_loss}, step)
        py_logger.info('Logging scalar, step = {}'.format(step))

    if img_period % 250 == 0:
        gen.eval()
        z = Variable(torch.rand((10 * 10, z_dim)).to(device))
        fake_examples = gen(z)
        x = vutils.make_grid(fake_examples, nrow=10, normalize=True, scale_each=True)
        logger.add_image('Samples', x, step)

        gen.train()
        py_logger.info('Logging img, step = {}'.format(step))


bce_loss = nn.BCELoss()


# Simple GAN (CE loss) Generator step
def bce_gen_step(gen, discr, z, gen_solver, device, loss=bce_loss):
    gen.zero_grad()

    fake_examples = gen(z)
    discr_fake_probs = discr(fake_examples)
    gen_loss = loss(discr_fake_probs, Variable(torch.ones(z.shape[0], 1).to(device)))
    batch_loss = gen_loss.detach().item()

    gen_loss.backward()
    gen_solver.step()

    return batch_loss


def bce_discr_step(gen, discr, z, batch, discr_solver, device, loss=bce_loss):
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


mse_loss = nn.MSELoss()


# Better GAN (LS loss)
def ls_gen_step(gen, discr, z, gen_solver, device, loss=mse_loss):
    batch_loss = bce_gen_step(gen, discr, z, gen_solver, device, loss=loss)
    return batch_loss


def ls_discr_step(gen, discr, z, batch, discr_solver, device, loss=mse_loss):
    batch_loss = bce_discr_step(gen, discr, z, batch, discr_solver, device, loss=loss)
    return batch_loss


# Even Better GAN (Wasserstein loss)
def wass_gen_step():
    pass


def wass_discr_step():
    pass
