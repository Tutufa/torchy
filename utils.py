import pickle
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision import utils as vutils


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

    if img_period % 250 == 0:
        gen.eval()
        z = Variable(torch.rand((10 * 10, z_dim)).to(device))
        fake_examples = gen(z)
        x = vutils.make_grid(fake_examples, nrow=10, normalize=True, scale_each=True)
        logger.add_image('Samples', x, step)

        gen.train()
