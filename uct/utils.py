
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
                # if object == 'time_log':
                #     self.time_log = Unpickler(f).load()
                #     logging.info(self.time_log)
                # if object == 'test_log':
                #     self.test_log = Unpickler(f).load()
                if object == 'arguments':
                    return Unpickler(f).load()
                # if object == 'avg_steps':
                #     self.avg_sat_steps_taken_per_level = Unpickler(f).load()
                if object == 'examples':
                    return Unpickler(f).load()

    def save_object(self, object_name, object, folder =None):
        if self.args.active_tester:
            return
        folder = self.args.folder_name if folder is None else folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, object_name + '.pth.tar')
        with open(filename, "wb+") as f:
            # if object == 'time_log':
            #     Pickler(f).dump(self.time_log)
            # if object == 'test_log':
            #     Pickler(f).dump(self.test_log)
            if 'arguments' in object_name:
                Pickler(f).dump(object)
                print('LEVEL: {}'.format(object.level))
            # if object == 'avg_steps':
            #     Pickler(f).dump(self.avg_sat_steps_taken_per_level)
            if object_name == 'examples':
                Pickler(f).dump(object)
                print('Number of entries in last level of train_examples_history: {}'.format(len(object[-1])))
        #filename = os.path.join(folder, object_name + '_level_' + str(self.args.level) + '.pth.tar')
        #with open(filename, "wb+") as f:
        #    # if object == 'time_log':
        #    #     Pickler(f).dump(self.time_log)
        #    # if object == 'test_log':
        #    #     Pickler(f).dump(self.test_log)
        #    if 'arguments' in object_name:
        #        Pickler(f).dump(object)
        #        print('LEVEL: {}'.format(object.level))
        #    # if object == 'avg_steps':
        #    #     Pickler(f).dump(self.avg_sat_steps_taken_per_level)
        #    if object_name == 'examples':
        #        Pickler(f).dump(object)
        #        print('Number of entries in last level of train_examples_history: {}'.format(len(object[-1])))

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
    #print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
