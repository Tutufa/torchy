import torch
import torch.optim as solvers
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from generators import MNISTGenerator
from discriminators import MNISTDiscriminator
from helpers import MNISTDataset
from helpers import bce_gen_step, bce_discr_step
from helpers import log_step


def train_step(gen_step, discr_step, gen, discr, batch, z, gen_solver, discr_solver, device, step, logger):
    # generator step
    gen_loss = gen_step(gen, discr, z, gen_solver, device)
    # discriminator update
    discr_loss = discr_step(gen, discr, z, batch, discr_solver, device)
    # logging
    log_step(step, gen, gen_loss, discr_loss, z_dim, device, logger)


n_epoch = 16
z_dim = 96
step_n = 0
device = 'cuda:0'
mnist_path = '/home/ubuntu/torchy/data/mnist/mnist.pkl'
writer = SummaryWriter(comment='DC_BCE_GAN')


mnist = MNISTDataset(path_to_data=mnist_path)
train_dl, valid_dl = mnist.get_iterators()

gen = MNISTGenerator()
discr = MNISTDiscriminator(capacity=32)
gen.to(device)
discr.to(device)

gen_solver = solvers.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999))
discr_solver = solvers.Adam(discr.parameters(), lr=0.001, betas=(0.5, 0.999))

for epoch in range(0, n_epoch):
    for batch_n, (batch, _) in enumerate(train_dl):

        z = Variable(torch.rand((batch.shape[0], z_dim)).to(device))
        batch = batch.to(device)

        train_step(gen_step=bce_gen_step, discr_step=bce_discr_step,
                   gen=gen, discr=discr, batch=batch, z=z,
                   gen_solver=gen_solver, discr_solver=discr_solver,
                   device=device, step=step_n, logger=writer)

        step_n += 1
