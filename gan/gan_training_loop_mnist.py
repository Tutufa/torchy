import torch
import torch.optim as solvers
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from gan_helpers import gan_train_step
from gan_helpers import bce_gen_step, bce_discr_step
from gan_helpers import ls_gen_step, ls_discr_step

from generators import MNISTGenerator
from discriminators import MNISTDiscriminator
from utils import MNISTDataset

import logging
logging.basicConfig(level=logging.INFO)
py_logger = logging.getLogger('gan_training_logging')


loss_type = 'bce'
n_epoch = 32
z_dim = 96
step_n = 0
mnist_path = '/storage_disk2/experiments/torchy/data/mnist/mnist.pkl'


if loss_type == 'bce':
    logger_prefix = '_DC_BCE_GAN'
    _gen_step = bce_gen_step
    _discr_step = bce_discr_step
elif loss_type == 'ls':
    logger_prefix = '_DC_LS_GAN'
    _gen_step = ls_gen_step
    _discr_step = ls_discr_step
else:
    raise NotImplemented

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter(comment=logger_prefix)


mnist = MNISTDataset(path_to_data=mnist_path)
train_dl, valid_dl = mnist.get_iterators()

gen = MNISTGenerator()
discr = MNISTDiscriminator(capacity=32)
gen.to(device)
discr.to(device)

gen_solver = solvers.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999))
discr_solver = solvers.Adam(discr.parameters(), lr=0.001, betas=(0.5, 0.999))

for epoch in range(0, n_epoch):
    py_logger.info('Epoch {}'.format(epoch))
    for batch_n, (batch, _) in enumerate(train_dl):

        z = Variable(torch.rand((batch.shape[0], z_dim)).to(device))
        batch = batch.to(device)

        gan_train_step(gen_step=_gen_step, discr_step=_discr_step,
                       gen=gen, discr=discr, batch=batch, z=z,
                       gen_solver=gen_solver, discr_solver=discr_solver,
                       device=device, step=step_n, logger=writer)

        step_n += 1
        if step_n % 250 == 0:
            py_logger.info('Epoch {} | Step {}'.format(epoch, step_n))
