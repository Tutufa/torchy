import torch
from torch.nn import functional
from torchvision import utils as vutils


def vae_loss(x_hat, x, mean, log_var):
    reconstruction_loss = functional.binary_cross_entropy(x_hat, x, reduction='sum')
    latent_reg = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + latent_reg


def vae_train_step(model, batch, vae_loss, solver):
    model.zero_grad()

    batch_hat, mean, log_var = model(batch)

    loss = vae_loss(batch_hat, batch, mean, log_var)
    batch_loss = loss.detach().item()

    loss.backward()
    solver.step()

    return batch_loss


def vae_eval_step(model, batch, vae_loss):
    model.eval()

    batch_hat, mean, log_var = model(batch)

    loss = vae_loss(batch_hat, batch, mean, log_var)
    batch_loss = loss.detach().item()

    model.train()
    return batch_loss


def log_vae_step(step, logger, model=None, eval_batch=None, eval_shape=(1, 28, 28),
                 loss=None, train=True, img=False):

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
