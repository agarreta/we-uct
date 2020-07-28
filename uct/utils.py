
from .neural_net_wrapper import NNetWrapper
import os
from pickle import Pickler, Unpickler
import torch
import random
import numpy as np

class Utils(object):

    def __init__(self, args):
        self.args = args

    def load_object(self, object):
        file_name = os.path.join(self.args.folder_name, object + '.pth.tar')
        print('Loading {}'.format(file_name))
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                return Unpickler(f).load()

    def save_object(self, object_name, object, folder =None):
        if self.args.active_tester:
            return
        folder = self.args.folder_name if folder is None else folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, object_name + '.pth.tar')
        with open(filename, "wb+") as f:
            Pickler(f).dump(object)

    def load_nnet(self, device, training:bool, load, folder='', filename='model.pth.tar'):
        nnet = NNetWrapper(self.args, device, training=training, seed=1)

        if load:
            nnet.load_checkpoint(folder=folder, filename=filename)

        nnet.set_optimizer_device(device)
        nnet.set_parameter_device(device)

        if training:
            nnet.model.train()
        else:
            nnet.model.eval()

        return nnet


def seed_everything(seed):
    # https://www.kaggle.com/hmendonca/fold1h4r3-arcenetb4-2-256px-rcic-lb-0-9759 cells 45-50
    # print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
