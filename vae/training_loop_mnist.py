import torch
import torch.optim as solvers
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import MNISTDataset
from vae_models import VAE, MNISTDecoder, MNISTEncoder
from vae_helpers import vae_train_step

import logging
logging.basicConfig(level=logging.INFO)
py_logger = logging.getLogger('vae_training_logging')


n_epoch = 32
step_n = 0
mnist_path = '/Users/kovalenko/PycharmProjects/torchy/data/mnist/mnist.pkl'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter(comment='_VAE')


mnist = MNISTDataset(path_to_data=mnist_path)
train_dl, valid_dl = mnist.get_iterators()


enc = MNISTEncoder()
dec = MNISTDecoder()
model = VAE(encoder=enc, decoder=dec)
model.to(device)

vae_solver = solvers.Adam(model.parameters(), lr=1e-3)

for epoch in range(0, n_epoch):
    py_logger.info('Epoch {}'.format(epoch))
    for batch_n, (batch, _) in enumerate(train_dl):

        batch = batch.to(device)

        vae_train_step()

        step_n += 1
        if step_n % 250 == 0:
            py_logger.info('Epoch {} | Step {}'.format(epoch, step_n))


















