import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log
import random
from .gaussian_noise import GaussianNoise


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class NewResnet2___(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NewResnet2, self).__init__()
        num_layers, d_model, nhead, dim_feedforward = 7, 256, 32, 256
        self.d_model=d_model
        self.pos_encoder = PositionalEncoding(3+1+15+1)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,dim_feedforward)
        self.initial_linear = nn.Linear(3+1+15+1, d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, nn.LayerNorm(d_model))
        self.final_linear = nn.Linear(d_model, 1)

    def forward(self, x, ctx=None):
        """
        :param x: (batch_size, length, d_model)
        :return: value in [-1,1]
        """
        x = self.pos_encoder(x)
        x = self.initial_linear(x)
        x = self.encoder(x)
        x = self.final_linear(x)
        x = x.squeeze(-1)
        x = x.mean(dim=1)
        return torch.tensor([1]), x


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

    def __init__(self, args, channels=None, blocks=None, device='cpu', one_head = '', headless=False):
        super(NewResnet2, self).__init__()

        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        self.noise = GaussianNoise()

        self.args = args
        self.device = device
        self.final_num_channels = 16

        self.num_in_channels = self.args.LEN_CORPUS# if self.args.bound else 2 #1 + len(self.args.ALPHABET) + len(self.args.VARIABLES)
        self.num_channels = self.args.num_channels if channels is None else channels
        self.num_resnet_blocks = self.args.num_resnet_blocks if blocks is None else blocks

        self.linear_input_size = int(self.final_num_channels * 2*(self.args.SIDE_MAX_LEN))
        #self.linear_input_size = self.final_num_channels * int(self.args.SIDE_MAX_LEN/2) *int((self.args.LEN_CORPUS+1)/2)

        self.conv_initial = nn.Conv2d(self.num_in_channels, self.num_channels, 3, padding = 1, stride = 1)
        self.bn_initial = nn.BatchNorm2d(self.num_channels)
        self.bn_convred1 = nn.BatchNorm2d(self.num_channels)
        self.bn_convred2 = nn.BatchNorm2d(self.num_channels)
        #self.convred_final1 = nn.Conv2d(self.num_channels, self.num_channels, [3,2], padding=[1,0], stride=[1,2])
        #self.convred_final2 = nn.Conv2d(self.num_channels, self.num_channels, [3,2], padding=[1,0], stride=[1,2])
        #self.convred_initial = nn.Conv2d(self.num_channels, self.num_channels, [3,2], padding=[1,0], stride=[1,2])

        #
        self.maxpool_initial = nn.MaxPool2d(kernel_size=2)
        self.maxpool_final1 = nn.MaxPool2d(kernel_size=2)

        self.resnet_blocks = []
        for _ in range(self.num_resnet_blocks):
            self.add_module('resnet_block_' + str(_), ResNetBlock(self.num_channels, self.num_channels))

        self.conv_value = nn.Conv2d(self.num_channels, self.final_num_channels, 1, stride=1, padding=0)
        self.bn_value = nn.BatchNorm2d(self.final_num_channels)
        #self.maxpool_value = nn.MaxPool2d(kernel_size=2)
        if  self.args.bound:
            self.fc2_value = nn.Linear(self.linear_input_size,1) #self.args.linear_hidden_size)
        else:
            self.gru = nn.GRU(2*(self.final_num_channels),1,batch_first=True)
            self.bias = nn.Parameter(torch.normal(torch.tensor(0.)),requires_grad=True)
        #self.fc2_value2 = nn.Linear(self.args.linear_hidden_size, 1)

        if self.args.mcts_type != 'alpha0np':
            self.conv_pi = nn.Conv2d(self.num_channels, self.final_num_channels, 1, stride=1, padding=0)
            self.bn_pi = nn.BatchNorm2d(self.final_num_channels)
            self.maxpool_pi = nn.MaxPool2d(kernel_size=2)
            self.fc1_pi = nn.Linear(self.linear_input_size, self.args.linear_hidden_size)  # self.args.linear_hidden_size)
            self.fc2_pi = nn.Linear(self.args.linear_hidden_size, self.args.num_actions)

        self.block_names = [self.resnet_block_0]
        if self.num_resnet_blocks > 1:
            self.block_names += [ self.resnet_block_1]
        if self.num_resnet_blocks > 2:
            self.block_names += [self.resnet_block_2]
        if self.num_resnet_blocks > 3:
            self.block_names += [self.resnet_block_3]
        if self.num_resnet_blocks > 4:
            self.block_names += [self.resnet_block_4]
        if self.num_resnet_blocks > 5:
            self.block_names += [self.resnet_block_5, self.resnet_block_6, self.resnet_block_7]
        if self.num_resnet_blocks > 8:
            self.block_names += [self.resnet_block_8, self.resnet_block_9]
        if self.num_resnet_blocks > 10:
            self.block_names += [self.resnet_block_10, self.resnet_block_11,
                                 self.resnet_block_12,
                                 self.resnet_block_13, self.resnet_block_14]
        if self.num_resnet_blocks > 15:
            self.block_names += [self.resnet_block_15]
        if self.num_resnet_blocks > 19:
            self.block_names += [self.resnet_block_16, self.resnet_block_17,self.resnet_block_18,self.resnet_block_19]


    def process_output(self, output, batch_size):
        #w = math.floor(self.args.SIDE_MAX_LEN/4)
        output = output.view(batch_size, self.final_num_channels,-1)

        output = output.view(batch_size, -1)
        #output = output.view(batch_size, self.final_num_channels * 2*w*2)
        return output


    def forward(self, state, stm = None):
        s = state
        batch_size = s.shape[0]

        s = F.relu(self.bn_initial(self.conv_initial(s)))
        #s = self.convred_initial(s)
        #s=self.maxpool_initial(s)
        for m in self.block_names:
            s = m(s)
        #s = F.relu(self.bn_convred1(self.convred_final1(s)))
        #s = F.relu(self.bn_convred2(self.convred_final2(s)))
        #s= self.maxpool_final1(s)
        # head value

        if self.args.unsat_value == -1:
            if self.args.bound:
                s_value = F.relu(self.bn_value(self.conv_value(s)))
                s_value = self.process_output(s_value, batch_size)
                s_value = torch.tanh(self.fc2_value(s_value))
            else:
                s_value = F.relu(self.bn_value(self.conv_value(s)))
                s_value = s_value.view(s_value.shape[0], s_value.shape[1]*s_value.shape[2], s_value.shape[3])
                #s_value = s_value.view(s_value.shape[0], s_value.shape[1],s_value.shape[2]* s_value.shape[3])
                #s_value = self.process_output(s_value, batch_size)
                s_value = s_value.transpose(1,2)
                s_value, _ = self.gru(s_value)
                s_value = s_value[:,-1,:]
                #s_value = torch.mean(torch.tanh(s_value)).unsqueeze(0) #+ self.bias

        else:
            s_value = torch.tanh(self.fc2_value(s_value))

        if  self.args.mcts_type=='alpha0np':
            s_pi = F.softmax(torch.tensor([self.args.num_actions * [1.]]), dim=-1)
        else:
            s_pi = F.relu(self.bn_value(self.conv_pi(s)))
            s_pi = self.process_output(s_pi, batch_size)
            s_pi = torch.relu(self.fc1_pi(s_pi))
            s_pi = torch.softmax(self.fc2_pi(s_pi), dim=-1)


        if random.random() < 1/1000:
            print(s_value, s_pi)
        #print(s_value)
        return s_pi, s_value




class NewResnet2___(nn.Module):

    def __init__(self, args, channels=None, blocks=None, device='cpu', one_head = '', headless=False):
        super(NewResnet2____, self).__init__()

        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        self.noise = GaussianNoise()

        self.args = args
        self.device = device
        self.final_num_channels = 16
        self.num_in_channels = self.args.LEN_CORPUS#1 + len(self.args.ALPHABET) + len(self.args.VARIABLES)
        self.num_channels = self.args.num_channels if channels is None else channels
        self.num_resnet_blocks = self.args.num_resnet_blocks if blocks is None else blocks

        self.linear_input_size = int(self.final_num_channels * 2*(self.args.SIDE_MAX_LEN))
        #self.linear_input_size = self.final_num_channels * int(self.args.SIDE_MAX_LEN/2) *int((self.args.LEN_CORPUS+1)/2)

        self.conv_initial = nn.Conv2d(self.num_in_channels, self.num_channels, 3, padding = 1, stride = 1)
        self.bn_initial = nn.BatchNorm2d(self.num_channels)
        self.bn_convred1 = nn.BatchNorm2d(self.num_channels)
        self.bn_convred2 = nn.BatchNorm2d(self.num_channels)
        #self.convred_final1 = nn.Conv2d(self.num_channels, self.num_channels, [3,2], padding=[1,0], stride=[1,2])
        #self.convred_final2 = nn.Conv2d(self.num_channels, self.num_channels, [3,2], padding=[1,0], stride=[1,2])
        #self.convred_initial = nn.Conv2d(self.num_channels, self.num_channels, [3,2], padding=[1,0], stride=[1,2])

        #
        self.maxpool_initial = nn.MaxPool2d(kernel_size=2)
        self.maxpool_final1 = nn.MaxPool2d(kernel_size=2)

        self.resnet_blocks = []
        for _ in range(self.num_resnet_blocks):
            self.add_module('resnet_block_' + str(_), ResNetBlock(self.num_channels, self.num_channels))

        self.conv_value = nn.Conv2d(self.num_channels, self.final_num_channels, 1, stride=1, padding=0)
        self.bn_value = nn.BatchNorm2d(self.final_num_channels)
        #self.maxpool_value = nn.MaxPool2d(kernel_size=2)
        #self.fc2_value = nn.Linear(self.linear_input_size,1) #self.args.linear_hidden_size)
        #self.fc2_value2 = nn.Linear(self.args.linear_hidden_size, 1)


        self.block_names = [self.resnet_block_0]
        if self.num_resnet_blocks > 1:
            self.block_names += [ self.resnet_block_1]
        if self.num_resnet_blocks > 2:
            self.block_names += [self.resnet_block_2]
        if self.num_resnet_blocks > 3:
            self.block_names += [self.resnet_block_3]
        if self.num_resnet_blocks > 4:
            self.block_names += [self.resnet_block_4]
        if self.num_resnet_blocks > 5:
            self.block_names += [self.resnet_block_5, self.resnet_block_6, self.resnet_block_7]
        if self.num_resnet_blocks > 8:
            self.block_names += [self.resnet_block_8, self.resnet_block_9]
        if self.num_resnet_blocks > 10:
            self.block_names += [self.resnet_block_10, self.resnet_block_11,
                                 self.resnet_block_12,
                                 self.resnet_block_13, self.resnet_block_14]
        if self.num_resnet_blocks > 15:
            self.block_names += [self.resnet_block_15]



    def process_output(self, output, batch_size):
        #w = math.floor(self.args.SIDE_MAX_LEN/4)
        output = output.view(batch_size, self.final_num_channels,-1)

        output = output.view(batch_size, -1)
        #output = output.view(batch_size, self.final_num_channels * 2*w*2)
        return output


    def forward(self, state, stm = None):
        s = state
        batch_size = s.shape[0]

        s = F.relu(self.bn_initial(self.conv_initial(s)))
        #s = self.convred_initial(s)
        #s=self.maxpool_initial(s)
        for m in self.block_names:
            s = m(s)
        #s = F.relu(self.bn_convred1(self.convred_final1(s)))
        #s = F.relu(self.bn_convred2(self.convred_final2(s)))
        #s= self.maxpool_final1(s)
        # head value
        s_value = F.relu(self.bn_value(self.conv_value(s)))

        s_value = self.process_output(s_value, batch_size)
        if self.args.unsat_value == -1:
            s_value = torch.tanh(torch.mean(s_value))
        else:
            s_value = torch.sigmoid(torch.mean(s_value))

        if  self.args.mcts_type=='alpha0np':
            s_pi = F.softmax(torch.tensor([self.args.num_actions * [1.]]), dim=-1)
        else:
            s_pi = F.relu(self.bn_value(self.conv_pi(s)))
            s_pi = self.process_output(s_pi, batch_size)
            s_pi = torch.relu(self.fc1_pi(s_pi))
            s_pi = torch.softmax(self.fc2_pi(s_pi), dim=-1)


        if random.random() < 1/1000:
            print(s_value, s_pi)
        #print(s_value)
        return s_pi, s_value




class NewResnet2_(nn.Module):

    def __init__(self, args, channels=None, blocks=None, device='cpu', one_head = '', headless=False):
        super(NewResnet2_, self).__init__()

        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        self.lstm1 = nn.LSTM(input_size=2*args.LEN_CORPUS+2, hidden_size=8, num_layers=2, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=2*args.LEN_CORPUS+2, hidden_size=8, num_layers=2, bidirectional=True, batch_first=True)
        self.args = args
        self.device = device
        self.final_num_channels = 16
        self.num_in_channels = 2 #1 + len(self.args.ALPHABET) + len(self.args.VARIABLES)
        self.num_channels = self.args.num_channels if channels is None else channels
        self.num_resnet_blocks = self.args.num_resnet_blocks if blocks is None else blocks

        #self.linear_input_size = int(self.final_num_channels * 2*(self.args.SIDE_MAX_LEN))
        self.linear_input_size = 8

        self.conv_initial = nn.Conv2d(self.num_in_channels, self.num_channels, 3, padding = 1, stride = 1)
        self.bn_initial = nn.BatchNorm2d(self.num_channels)
        self.bn_convred1 = nn.BatchNorm2d(self.num_channels)
        self.bn_convred2 = nn.BatchNorm2d(self.num_channels)
        self.convred_final1 = nn.Conv2d(self.num_channels, 2, 3, padding=1, stride=1)
        #self.convred_final2 = nn.Conv2d(self.num_channels, self.num_channels, [3,2], padding=[1,0], stride=[1,2])
        #self.convred_initial = nn.Conv2d(self.num_channels, self.num_channels, [3,2], padding=[1,0], stride=[1,2])


        self.resnet_blocks = []
        for _ in range(self.num_resnet_blocks):
            self.add_module('resnet_block_' + str(_), ResNetBlock(self.num_channels, self.num_channels))



        self.conv_value = nn.Conv2d(self.num_channels, self.final_num_channels, 1, stride=1, padding=0)
        self.bn_value = nn.BatchNorm2d(self.final_num_channels)
        self.maxpool_value = nn.MaxPool2d(kernel_size=2)
        self.fc2_value = nn.Linear(self.linear_input_size,1) #self.args.linear_hidden_size)
        #self.fc2_value2 = nn.Linear(self.args.linear_hidden_size, 1)

        if self.args.mcts_type != 'alpha0':
            self.conv_pi = nn.Conv2d(self.num_channels, self.final_num_channels, 1, stride=1, padding=0)
            self.bn_pi = nn.BatchNorm2d(self.final_num_channels)
            self.maxpool_pi = nn.MaxPool2d(kernel_size=2)
            self.fc1_pi = nn.Linear(self.linear_input_size, self.args.linear_hidden_size)  # self.args.linear_hidden_size)
            self.fc2_pi = nn.Linear(self.args.linear_hidden_size, self.args.num_actions)

        self.block_names = [self.resnet_block_0]
        if self.num_resnet_blocks > 1:
            self.block_names += [ self.resnet_block_1]
        if self.num_resnet_blocks > 2:
            self.block_names += [self.resnet_block_2]
        if self.num_resnet_blocks > 3:
            self.block_names += [self.resnet_block_3]
        if self.num_resnet_blocks > 4:
            self.block_names += [self.resnet_block_4]
        if self.num_resnet_blocks > 5:
            self.block_names += [self.resnet_block_5, self.resnet_block_6, self.resnet_block_7]
        if self.num_resnet_blocks > 8:
            self.block_names += [self.resnet_block_8, self.resnet_block_9]
        if self.num_resnet_blocks > 10:
            self.block_names += [self.resnet_block_10, self.resnet_block_11,
                                 self.resnet_block_12,
                                 self.resnet_block_13, self.resnet_block_14]
        if self.num_resnet_blocks > 15:
            self.block_names += [self.resnet_block_15]



    def process_output(self, output, batch_size):
        #w = math.floor(self.args.SIDE_MAX_LEN/4)
        output = output.view(batch_size, self.final_num_channels,-1)

        output = output.view(batch_size, -1)
        #output = output.view(batch_size, self.final_num_channels * 2*w*2)
        return output


    def forward(self, state, stm = None):
        s = state
        batch_size, l, w = s.shape[0], s.shape[-1], s.shape[-2]

        s = s.transpose(1,2) # batch x 2 x len x lencorpus+1
        s = s.view(s.shape[0], s.shape[1]*s.shape[2], -1) # batch x 2*len x lencorpus+1
        s, _ = self.lstm1(s)   # batch x 2*len *2*hidden_size1
        s= s.view(s.hape[0],2, l, w )
        s.transpose(2,3)

        s = F.relu(self.bn_initial(self.conv_initial(s)))
        # s = self.convred_initial(s)
        for m in self.block_names:
            s = m(s)

        s = F.relu(self.bn_convred1(self.convred_final1(s)))

        s = s.transpose(1,2) # batch x 2 x len x lencorpus+1
        s = s.view(s.shape[0], s.shape[1]*s.shape[2], -1) # batch x 2*len x lencorpus+1
        s, _ = self.lstm2(s)   # batch x final_num_channels*(corpus+1) x 2*hidden_size


        #
        # s = F.relu(self.bn_convred2(self.convred_final2(s)))
        # head value

        s_value = self.process_output(s, batch_size)
        s_value = torch.tanh(self.fc2_value(s_value))

        if  self.args.mcts_type!='alpha0':
            s_pi = F.softmax(torch.tensor([self.args.num_actions * [1.]]), dim=-1)
        else:
            s_pi = F.relu(self.bn_value(self.conv_pi(s)))
            s_pi = self.process_output(s_pi, batch_size)
            s_pi = torch.relu(self.fc1_pi(s_pi))
            s_pi = torch.softmax(self.fc2_pi(s_pi), dim=-1)
        if random.random() < 1/1000:
            print(s_value, s_pi)
        #print(s_value)
        return s_pi, s_value


