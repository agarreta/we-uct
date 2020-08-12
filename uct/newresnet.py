import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        s_ = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels (LEN_CORPUS) x 2 x MAX_LEN
        s_ = self.bn2(self.conv2(s_))  # batch_size x num_channels x 2 x length
        return F.relu(s_ + s)


class NewResnet2(nn.Module):

    def __init__(self, args, channels=None, blocks=None, device='cpu', one_head='', headless=False):
        super(NewResnet2, self).__init__()
        SIDE_MAX_LEN = args.SIDE_MAX_LEN if args.format_mode != 'cuts' else args.NNET_SIDE_MAX_LEN + 5
        np.random.seed(args.seed_class)
        torch.manual_seed(args.seed_class)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed_class)

        self.args = args
        self.device = device
        self.final_num_channels = 16

        self.num_in_channels = self.args.LEN_CORPUS
        self.num_channels = self.args.num_channels if channels is None else channels

        self.linear_input_size = int(self.final_num_channels * 2 * (SIDE_MAX_LEN))

        self.conv_initial = nn.Conv2d(self.num_in_channels, self.num_channels, 3, padding=1, stride=1)
        self.bn_initial = nn.BatchNorm2d(self.num_channels)

        num_resnet_blocks = self.args.num_resnet_blocks if blocks is None else blocks
        self.resnet_blocks = nn.ModuleList(
            [ResNetBlock(self.num_channels, self.num_channels) for _ in range(num_resnet_blocks)])

        self.conv_value = nn.Conv2d(self.num_channels, self.final_num_channels, 1, stride=1, padding=0)
        self.bn_value = nn.BatchNorm2d(self.final_num_channels)
        self.fc2_value = nn.Linear(self.linear_input_size, 1)  # self.args.linear_hidden_size)

    def process_output(self, output, batch_size):
        output = output.view(batch_size, self.final_num_channels, -1)
        output = output.view(batch_size, -1)
        return output

    def forward(self, state, stm=None):
        s = state
        batch_size = s.shape[0]
        s = F.relu(self.bn_initial(self.conv_initial(s)))
        for m in self.resnet_blocks:
            s = m(s)
        s_value = F.relu(self.bn_value(self.conv_value(s)))
        s_value = self.process_output(s_value, batch_size)
        s_value = torch.tanh(self.fc2_value(s_value))
        return s_value
