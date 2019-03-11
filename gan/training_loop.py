import torch
import torch.optim as solvers
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils

from generators import MNISTGenerator
from discriminators import MNISTDiscriminator
from helpers import MNISTDataset

from tensorboardX import SummaryWriter


def train_step(gen: nn.Module, discr: nn.Module,
               batch, z,
               gen_solver, discr_solver,
               device, logger, step):

    # generator update
    gen.zero_grad()

    fake_examples = gen(z)
    discr_fake_probs = discr(fake_examples)

    gen_loss = loss(discr_fake_probs, Variable(torch.ones(batch.shape[0], 1).to(device)))
    gen_loss.backward()
    gen_solver.step()

    # discriminator update
    discr.zero_grad()

    fake_examples = gen(z)
    discr_fake_probs = discr(fake_examples)

    discr_real_probs = discr(batch)
    discr_real_loss = loss(discr_real_probs, Variable(torch.ones(batch.shape[0], 1).to(device)))
    discr_fake_loss = loss(discr_fake_probs, Variable(torch.zeros(batch.shape[0], 1).to(device)))

    discr_loss = discr_real_loss + discr_fake_loss
    discr_loss.backward()
    discr_solver.step()

    # logging
    if step % 100 == 0:
        logger.add_scalars('gan', {'gen_loss': gen_loss.detach().item(),
                                   'discr_loss': discr_loss.detach()}, step)
        print('Logging {}'.format(step))

    if step % 1000 == 0:
        z = Variable(torch.randn((8*3, z_dim)).view(-1, 1, 7, 7))
        fake_examples = gen(z)
        x = vutils.make_grid(fake_examples, normalize=True, scale_each=True)
        logger.add_image('Samples', x, step)


n_epoch = 1
z_dim = 7 * 7 * 1

gen = MNISTGenerator()
discr = MNISTDiscriminator()


gen_solver = solvers.Adam(gen.parameters(), lr=0.001)
discr_solver = solvers.Adam(discr.parameters(), lr=0.001)

loss = nn.BCELoss()
logger = SummaryWriter()


mnist_path = '/Users/kovalenko/PycharmProjects/torchy/data/mnist/mnist.pkl'
mnist = MNISTDataset(path_to_data=mnist_path)
train_dl, _ = mnist.get_iterators()


for epoch in range(0, n_epoch):
    for batch_n, (batch, _) in enumerate(train_dl):
        z = Variable(torch.randn((batch.shape[0], z_dim)).view(-1, 1, 7, 7))

        train_step(gen=gen, discr=discr, batch=batch, z=z,
                   gen_solver=gen_solver, discr_solver=discr_solver,
                   device='cpu', logger=logger, step=batch_n)

















