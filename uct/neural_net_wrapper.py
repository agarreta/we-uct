
import logging
import torch.optim as optim
import time
import numpy as np
import torch
import os
from .neural_net_models.newresnet import NewResnet2
from .neural_net_models.uniform_model import UniformModel
import random

def seed_everything(seed):
    # https://www.kaggle.com/hmendonca/fold1h4r3-arcenetb4-2-256px-rcic-lb-0-9759 cells 45-50
    #print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_log(folder):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=folder + f'/log.log')
    console = logging.StreamHandler()
    from PIL import PngImagePlugin
    logger = logging.getLogger(PngImagePlugin.__name__)
    logger.setLevel(logging.INFO)
    console.setLevel(logging.info)
    logging.getLogger('').addHandler(console)


class NNetWrapper( object ):
    def __init__(self, args, device, training, seed=None):
        self.args = args
        if seed is not None:
            seed_everything(seed)
            self.seed =seed
        else:
            assert False
        self.device = device
        self.training = training

        self.v_losses = []
        self.pi_losses = []
        self.train_scores = []
        self.test_scores = []

        if not self.args.oracle:
            m = NewResnet2
            self.model = m(args, channels = self.args.num_channels,
                                   blocks= self.args.num_resnet_blocks,
                                   device= self.device)
        else:
            self.model = UniformModel(self.args,self.args.num_actions)
        print(self.device)
        if self.args.active_tester:
            self.device = 'cpu'
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        self.set_optimizer_device(self.device)
        self.model.training = self.training
        if self.training:
            self.model.train()
        else:
            self.model.eval()
        self.device = self.args.train_device

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f'NUM TRAINABLE PARAMS: {params}')
        self.model.eval()
        for x in self.model.parameters():
            x.requires_grad_(False)

    def train(self, examples):

        """
        examples: list of examples, each example is of form (s_eq, pi, v) where s_eq is an equation
        formated for input to the nn
        """
        self.model.train()
        t = time.time()

        print(f'Num epochs {self.args.epochs}')
        for epoch in range(self.args.epochs):
            for param in self.model.parameters():
                param.requires_grad_(True)
            batch_idx = 0
            batch_v = []
            print(f'there are {len(examples)} examples to train on')
            while batch_idx < int(len(examples)/self.args.batch_size):
                batch_examples = [examples[i] for i in range(batch_idx * self.args.batch_size,
                                                                           (batch_idx + 1) * self.args.batch_size)]
                states_eq, pis, vs = list(zip(*batch_examples))
                l_v = self.core_train_function(states_eq, vs)
                batch_idx += 1
                batch_v.append(l_v.detach().cpu().numpy())
            self.v_losses.append(np.array(batch_v).mean())
            logging.info(f'EPOCH ::: {epoch}. LOSSES ::: {self.v_losses[-1]}')

        for param in self.model.parameters():
            param.requires_grad_(False)

        print(f'EPOCH ::: {epoch}. LOSSES :::  {self.v_losses[-1]}')
        logging.info(f'EPOCH ::: {epoch}. LOSSES :::  {self.v_losses[-1]}')
        print(f'Elapsed time during nn training: {round(time.time()-t, 2)}')
        logging.info(f'Elapsed time during nn training: {round(time.time()-t, 2)}')

        self.model.eval()
        for x in self.model.parameters():
            x.requires_grad_(False)

    def core_train_function(self, states_eq, vs):
        states_eq = list(states_eq)
        states_eq = torch.cat(states_eq)
        target_vs = torch.tensor(np.array(vs).astype(float),dtype=torch.float,device=self.device)
        self.optimizer.zero_grad()
        self.model.device = self.args.train_device

        states_eq = states_eq.to('cuda:0')
        target_vs = target_vs.to('cuda:0')

        out_v = self.model.forward(states_eq)
        l_v = self.loss_v(target_vs, out_v, None)
        total_loss =  l_v
        if not self.args.oracle:
            total_loss.backward()
            self.optimizer.step()
        return l_v

    def loss_v(self, targets, outputs, weights = None):
        if outputs.device != self.device:
            outputs = outputs.to(self.device)
        if not self.args.values_01:
            errors_original = (targets - outputs.view(-1))**2
            return torch.sum(errors_original)/targets.size()[0]


    def predict(self, s0, smt = None, ctx=None):
        """only used during mcts as a call on a single equation, hence the [0] at the end of the function"""
        self.model.eval()
        if self.args.test_mode and self.args.play_device == self.args.train_device:
            s0 = s0.to(self.args.train_device)
        with torch.no_grad():
            v = self.model.forward(s0, smt)
        return v[0]

    def set_optimizer_device(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def set_parameter_device(self, device):
        for param in self.model.parameters():
            param.to(device)

    def save_checkpoint(self, folder, filename):
        if not self.args.active_tester:
            self.set_optimizer_device('cpu')
            self.set_parameter_device('cpu')

            filepath = os.path.join(folder, filename)
            if not os.path.exists(folder):
                print(" Checkpoint Directory does not exist. Making directory {}".format(folder))
                os.mkdir(folder)
            else:
                print(" Checkpoint Directory exists ")

            torch.save({
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
            }, filepath)

    def load_checkpoint(self, folder, filename):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        print(filepath)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        print(filepath)
        checkpoint = torch.load(filepath)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.eval()
        for x in self.model.parameters():
            x.requires_grad_(False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
