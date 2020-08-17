import torch
import torch.nn as nn
import os
import numpy as np
import random
from utils import preprocess
from torch.distributions.normal import Normal
import torch.nn.functional as F

def print_num_params(model, model_name='model'):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num trainable params in {model_name}: {params}')
    model.num_parameters = params

def seed_everything(seed):
    # https://www.kaggle.com/hmendonca/fold1h4r3-arcenetb4-2-256px-rcic-lb-0-9759 cells 45-50
    # print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(10)

class ResNetBlock(nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super(ResNetBlock, self).__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels

        self.conv1 = nn.Conv2d(self.num_in_channels, self.num_out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_out_channels, self.num_out_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_out_channels)
        self.bn2 = nn.BatchNorm2d(self.num_out_channels)

    def forward(self, s):
        s_ = torch.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels (LEN_CORPUS) x 2 x MAX_LEN
        s_ = self.bn2(self.conv2(s_))  # batch_size x num_channels x 2 x length
        return torch.relu(s_ + s)


class BlockStack(nn.Module):
    def __init__(self, num_channels, num_blocks):
        super(BlockStack,self).__init__()
        self.init_conv = nn.Conv2d(1, num_channels, 3, 1, 1)
        self.blocks = nn.ModuleList([ResNetBlock(num_channels, num_channels)  for _ in range(num_blocks)])
    def forward(self,x):
        x = self.init_conv(x)
        for b in self.blocks:
            x = b(x)
        return x

class NormalModel_old(nn.Module):
    def __init__(self, channels, out_dim, dropout, in_channels=1):
        super(NormalModel_old, self).__init__()
        self.drop = dropout
        #self.init_conv = nn.Conv2d(1, normal_num_channels, 3, 1, 1)
        #self.blocks = nn.ModuleList([ResNetBlock(normal_num_channels, normal_num_channels) for _ in range(num_blocks)])
        #self.final_conv = nn.Conv2d(normal_num_channels, normal_num_channels, 3, 1, 1)
        #self.linear = nn.Linear(28*28*normal_num_channels, 2)
        self.c1 = nn.Conv2d(in_channels, channels, 5, 1, 2)
        self.c2 = nn.Conv2d(channels, channels, 5, 1, 2)
        self.c3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.c4 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.l1 = nn.Linear(7 * 7 * channels, 32)
        self.l2 = nn.Linear(32, out_dim)
        print_num_params(self, 'normal')


    def forward(self, x, drop=False):
        x = torch.dropout(torch.max_pool2d(self.c2(self.c1(x)), 2,2), p=self.drop, train=self.training)
        x = torch.dropout(torch.max_pool2d(self.c4(self.c3(x)), 2,2), p=self.drop, train=self.training)
        x= self.l2(torch.dropout(self.l1(x.view(x.shape[0], -1)), p=self.drop, train=self.training))
        if x.shape[-1]==1:
            x = x.squeeze(-1)
        return x

class NormalModel(nn.Module):
    def __init__(self, channels, out_dim, dropout, in_channels=1, num_blocks = None):
        super(NormalModel, self).__init__()
        self.drop = dropout
        #self.init_conv = nn.Conv2d(1, normal_num_channels, 3, 1, 1)
        #self.blocks = nn.ModuleList([ResNetBlock(normal_num_channels, normal_num_channels) for _ in range(num_blocks)])
        #self.final_conv = nn.Conv2d(normal_num_channels, normal_num_channels, 3, 1, 1)
        #self.linear = nn.Linear(28*28*normal_num_channels, 2)
        if num_blocks is not None:
            self.stack_block = BlockStack(2*channels, num_blocks)
        else:
            self.stack_block =None
        in_channels = in_channels if num_blocks is None else 2*channels
        self.c1 = nn.Conv2d(in_channels, channels, 5, 1, 2)
        self.c2 = nn.Conv2d(channels, channels, 5, 1, 2)
        self.c3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.c4 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.l1 = nn.Linear(7 * 7 * channels, 32)
        self.l2 = nn.Linear(32, out_dim)
        print_num_params(self, 'normal')


    def forward(self, x_, drop=False):
        if self.stack_block is not None:
            x_ = self.stack_block(x_)
        x = torch.dropout(torch.max_pool2d(self.c2(self.c1(x_)), 2,2), p=self.drop, train=self.training)
        x = torch.dropout(torch.max_pool2d(self.c4(self.c3(x)), 2,2), p=self.drop, train=self.training)
        x= self.l2(torch.dropout(self.l1(x.view(x.shape[0], -1)), p=self.drop, train=self.training))
        if x.shape[-1]==1:
            x = x.squeeze(-1)

        return x
LOG_STD_MIN = -20
LOG_STD_MAX = 1
class Actor(nn.Module):
    def __init__(self, num_channels=4, num_blocks=1, polyak=0.5, class_net=None):
        super(Actor, self).__init__()
        self.init_conv = nn.Conv2d(1, num_channels, 3, 1, 1)
        self.blocks = nn.ModuleList([ResNetBlock(num_channels, num_channels) for _ in range(num_blocks)])
        self.mu_layer1 = nn.Conv2d(num_channels, 1, 3, 1, 1)
        #self.mu_layer2 = nn.Conv2d(1, 1, 1, 0, 1)
        self.log_std_layer1 = nn.Conv2d(num_channels, 1, 3, 1, 1)
        #self.log_std_layer2 = nn.Conv2d(1, 1, 1, 0, 1)
        print_num_params(self, 'actor model')
        self.polyak =polyak
        self.class_net = class_net

    def forward(self, x, deterministic=False, with_logprob=True):
        #x = preprocess(x)
        #x_ = self.init_conv(x)
        #for block in self.blocks:
        #    x_ = block(x_)
        #x_= self.final_conv(x_)
        #if random.random() < 0.001:
        #    print('actor', x_.mean().item(), x_.max().item(),  x_.min().item(), x_.std().item())
        ##x__ = torch.clamp(x_,-1.,1.)
        #x__ = torch.sigmoid(x_)
#
        #return torch.clamp(self.polyak*x + (1-self.polyak)*x__, 0., 1.), x_
        if self.class_net is  None:
            x = preprocess(x)
            x_ = self.init_conv(x)
            for block in self.blocks:
                x_ = block(x_)
        else:
            x_ = self.class_net(x)
        #x_ = self.final_conv(x_)
        #mu = self.mu_layer2(torch.relu(self.mu_laye1r(x_)))
        mu = self.mu_layer1(x_)
        #log_std = self.log_std_layer2(torch.relu(self.log_std_layer1(x_)))
        log_std = self.log_std_layer1(x_)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        #std = F.softplus(log_std) #
        std = torch.exp(log_std)


        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action)
            if torch.isnan(logp_pi.mean()):
                print('boh')
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action)))
            logp_pi=logp_pi.mean([-1,-2])
            logp_pi = logp_pi.view(-1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        #pi_action = self.act_limit * pi_action

        pi_action = torch.clamp(self.polyak*x + (1-self.polyak)*pi_action, 0., 1.)

        return pi_action, logp_pi, mu, std

class Qfun(nn.Module):
    def __init__(self, channels=4, out=10, dropout=0):
        super(Qfun, self).__init__()
        self.main = NormalModel(channels, out, dropout, in_channels=1)
        print_num_params(self, 'Qfun model')

    def forward(self, x, drop=False):
        """
        x, y: two batch_size x 1 x 28 x 28 tensors
        """
        x = preprocess(x)

        #x = torch.cat([x,y], dim=1)  #TODO: I could take the difference instead of concatenating
        #x =  y-x
        x = self.main(x, drop)
        #x = torch.softmax(x,-1)
        return x
        #return 6.52*torch.tanh(x)


class ActorCritic(nn.Module):
    def __init__(self, channels, actor_channels, dropout, polyak,
                 max_num_steps, class_net, num_blocks):
        super(ActorCritic, self).__init__()
        self.channels = channels
        self.actor_channels = actor_channels
        self.dropout = dropout
        self.polyak = polyak
        self.max_num_steps = max_num_steps
        self.class_net= class_net
        self.num_blocks = num_blocks
        self.reset()

    def reset(self):
        self.qfun1 = Qfun(self.channels,1, self.dropout)
        self.qfun2 = Qfun(self.channels,1, self.dropout)
        nb = None if self.class_net is False else self.num_blocks
        self.classifier = Classifier(self.channels, 10, self.dropout, self.max_num_steps,1., nb)
        self.randomnet = Classifier(self.channels, 10, 0., self.max_num_steps)
        self.knowledge = Classifier(int(self.channels/4), 10, 0., self.max_num_steps, out_multiplyer=1)
        cn = None if self.class_net is None else self.classifier.main.stack_block
        self.actor = Actor(self.actor_channels,self.num_blocks, polyak=self.polyak, class_net=cn)



class Classifier(nn.Module):
    def __init__(self, channels=4, out=10, dropout=0., max_num_steps=10, out_multiplyer=1., num_blocks=None):
        super(Classifier, self).__init__()
        self.main = NormalModel(channels, out, dropout,in_channels=1, num_blocks=num_blocks)
        self.max_num_steps = max_num_steps
        self.out_multiplyer = out_multiplyer
        print_num_params(self, 'classifier model')

    def forward(self, x, drop=False):
        x = preprocess(x)
        x = self.main(x, drop=False)
        if False: #self.max_num_steps is not None:
            x=(1/(self.max_num_steps))*torch.softmax(x,-1)
        return self.out_multiplyer*x


class Rfun(nn.Module):
    def __init__(self, channels=4, out=10):
        super(Rfun, self).__init__()
        self.main = NormalModel(channels, out, in_channels=2)
        print_num_params(self, 'Rfun model')

    def forward(self, x, y, drop=False):
        """
        x, y: two batch_size x 1 x 28 x 28 tensors
        """
        x = torch.cat([x,y], dim=1)  #TODO: I could take the difference instead of concatenating
        #x =  y-x
        return self.main(x, drop)

