import torch
import torch.optim as solvers
from tensorboardX import SummaryWriter

from utils import MNISTDataset
from vae_models import VAE, MNISTDecoder, MNISTEncoder
from vae_helpers import vae_train_step, vae_loss, vae_eval_step, log_vae_step

import logging
logging.basicConfig(level=logging.INFO)
py_logger = logging.getLogger('vae_training_logging')


n_epoch = 64
step_n = 0
mnist_path = '/storage_disk2/experiments/torchy/data/mnist/mnist.pkl'


device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter(comment='_VAE')

mnist = MNISTDataset(path_to_data=mnist_path, flatten=True)
train_dl, valid_dl = mnist.get_iterators()


enc = MNISTEncoder()
dec = MNISTDecoder()
model = VAE(encoder=enc, decoder=dec)
model.to(device)

vae_solver = solvers.Adam(model.parameters(), lr=1e-3)

for epoch in range(0, n_epoch):
    py_logger.info('Epoch {}'.format(epoch))

    epoch_loss = 0
    eval_loss = 0

    # train
    for batch_n, (batch, _) in enumerate(train_dl):

        batch = batch.to(device)
        epoch_loss += vae_train_step(model, batch, vae_loss, vae_solver)

        step_n += 1
        if step_n % 250 == 0:
            py_logger.info('Epoch {} | Step {}'.format(epoch, step_n))

    log_vae_step(step_n, writer, model=model, loss=epoch_loss/len(train_dl.dataset))

    # eval
    for batch_n, (batch, _) in enumerate(valid_dl):
        batch = batch.to(device)
        eval_loss += vae_eval_step(model, batch, vae_loss)

        if batch_n == 0:
            log_vae_step(step_n, writer, model=model, eval_batch=batch, img=True)

    log_vae_step(step_n, writer, model=model, loss=eval_loss / len(valid_dl.dataset), train=False)
