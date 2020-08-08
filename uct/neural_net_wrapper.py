# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:30:05 2019

@author: garre
"""
import logging
import torch.optim as optim
import time
import numpy as np
import torch
import os
from .neural_net_models.neural_net import NeuralNet
import pandas as pd
import torch.nn.functional as F
#from .word_equation.GINgraph2 import GraphWeNet
#from .neural_net_models.graphsage import GraphSAGE
from .neural_net_models.newresnet import NewResnet2
from .neural_net_models.uniform_model import UniformModel
#from .neural_net_models.SatNet import SatNet
#from .utils import seed_everything
from .word_equation.word_equation_utils import seed_everything
#from .neural_net_models.attention import WordEquationAttention
from copy import deepcopy
# torch.set_default_tensor_type(torch.HalfTensor)


def init_log(folder, mode='train'):
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    #
    # # TODO: What is this for?
    # logging.getLogger("tensorflow").setLevel(logging.WARNING)

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=folder + f'/log.log')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # add the handler to the root logger
    # logging.getLogger('').addHandler(console)


class NNetWrapper( NeuralNet ):
    def __init__(self, args, device, training, seed=None):

        init_log(args.folder_name, 'test' if args.test_mode else 'train')
        # logging.info('Trainer test log info')
        # logging.warning('Trainer test log warning')

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


        if args.nnet_type == 'newresnet' and not args.cube:
            if not self.args.oracle:
                m = NewResnet2
                self.model = m(args, channels = self.args.num_channels,
                                       blocks= self.args.num_resnet_blocks,
                                       device= self.device, one_head='')
            else:
                self.model = UniformModel(self.args,self.args.num_actions)
            print(self.device)
            if self.args.active_tester:
                self.device = 'cpu'
            self.model.to(self.device)

            self.model.to(self.device)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        self.set_optimizer_device(self.device)
        self.model.training = self.training
        if self.training:
            self.model.train()
        else:
            self.model.eval()

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f'NUM TRAINABLE PARAMS: {params}')

        if self.args.test_mode:
            l = torch.nn.Linear(20, 10).to(self.args.train_device)
            t = torch.randn(20).to(self.args.train_device)
            try:
                print(l(t))
            except:
                print(l(t))

        self.model.eval()
        for x in self.model.parameters():
            x.requires_grad_(False)

    def train(self, examples, smt_values = None):

        """
        examples: list of examples, each example is of form (s_eq, pi, v) where s_eq is an equation
        formated for input to the nn
        """
        self.device = self.args.train_device
        self.model.train()

        l = torch.nn.Linear(20, 10).to(self.args.train_device)
        t = torch.randn(20).to(self.args.train_device)
        try:
            print(l(t))
        except:
            print(l(t))

        t = time.time()

        print(f'Num epochs {self.args.epochs}')
        for epoch in range(self.args.epochs):

            for param in self.model.parameters():
                param.requires_grad_(True)

            batch_idx = 0
            batch_pi = []
            batch_v = []

            print(f'there are {len(examples)} examples to train on')
            while batch_idx < int(len(examples)/self.args.batch_size):
                batch_examples = [examples[i] for i in range(batch_idx * self.args.batch_size,
                                                                           (batch_idx + 1) * self.args.batch_size)]
                if not self.args.rec_mcts:
                    states_eq, pis, vs = list(zip(*batch_examples))
                else:
                    states_eq, pis, vs = list(zip(*batch_examples))

                weights = None
                if self.args.learnable_smt:
                    smt_values_batch = [smt_values[i] for i in range(batch_idx * self.args.batch_size,
                                                                               (batch_idx + 1) * self.args.batch_size)]
                else:
                    smt_values_batch = None
                try:
                    l_pi, l_v = self.core_train_function(states_eq, pis, vs, smt_values_batch, epoch, weights)
                except:
                    print('Error, epoch {}, batch_id {}'.format(epoch, batch_idx))
                    l_pi, l_v = self.core_train_function(states_eq, pis, vs, smt_values_batch, epoch, weights)

                batch_idx += 1
                batch_pi.append(l_pi.detach().cpu().numpy())
                if self.args.use_value_nn_head:
                    batch_v.append(l_v.detach().cpu().numpy())
                else:
                    batch_v.append(l_v)

            self.pi_losses.append(np.array(batch_pi).mean())
            self.v_losses.append(np.array(batch_v).mean())
            logging.error(f'EPOCH ::: {epoch}. LOSSES ::: {self.pi_losses[-1]} + {self.v_losses[-1]}')
            print(f'EPOCH ::: {epoch}. LOSSES ::: {self.pi_losses[-1]} + {self.v_losses[-1]}')

        for param in self.model.parameters():
            param.requires_grad_(False)

        logging.error(f'EPOCH ::: {epoch}. LOSSES ::: {self.pi_losses[-1]} + {self.v_losses[-1]}')
        logging.error(f'Elapsed time during nn training: {round(time.time()-t, 2)}')
        print(f'Elapsed time during nn training: {round(time.time()-t, 2)}')
        self.model.eval()
        for x in self.model.parameters():
            x.requires_grad_(False)

    @staticmethod
    def augment_data(batch_ex):
        values = [ex[2] for ex in batch_ex]
        values = list((pd.Series(values).value_counts()).items())
        min_val = min([v[1] for v in values]) +1
        values = [[x[0], x[1]-min_val] for x in values]
        values = {x[0]: x[1] for x in values}
        new_batches = []
        for ex in batch_ex:
            new_batches += [[ex[0].clone(), ex[1].copy(), ex[2]] for _ in range(values[ex[2]])]
        return batch_ex

    def get_weights(self, batch_ex):
        values = [ex[2] for ex in batch_ex]
        values = list((pd.Series(values).value_counts()).items())
        values = [[x[0], x[1]] for x in values]
        values = {x[0]:  len(batch_ex)/(max(x[1],1)) for x in values}
        if 0 in values.keys():
            values[0] *= 1.2
        weights = torch.tensor([values[x[2]] for x in batch_ex], dtype=torch.float, device = self.device)
        print(values)
        return weights


    def core_train_function(self, states_eq, pis, vs, smt_values=None, epoch = 0, weights=None):
        # print(len(states_eq), states_eq.shape[0])
        #if self.args.rec_mcts:
        #    pass
        #else:
        if (not self.args.sat) and self.args.nnet_type!= 'graphwenet':
            states_eq = list(states_eq)
        else:
            states_eq = states_eq[0]

        if self.args.nnet_type not in ['GIN', 'satnet', 'graphwenet', 'wordgamenet'] or self.args.nnet_type in ['lstmpe']:
            maxlen = max([x.shape[1] for x in states_eq])

        if self.args.nnet_type in ['resnet', 'recnewnet', 'newresnet','resnet_double', 'resnet1d', 'lstmpe',
                                   'wordgamenet','hanoinet', 'cubenet']:# or 'supernet':
            states_eq = torch.cat(states_eq)



        target_pis = torch.tensor(np.array(pis).astype(float),
                                  dtype=torch.float,
                                  device=self.device)

        if self.args.use_value_nn_head:
            target_vs = torch.tensor(np.array(vs).astype(float),
                                     dtype=torch.float,
                                     device=self.device)

        if smt_values is not None:
            smt_values = torch.tensor(smt_values, dtype = torch.float,  device =self.device)

        self.optimizer.zero_grad()
        self.model.device = self.args.train_device

        if self.args.nnet_type != 'supernet':
            if not self.args.rec_mcts:
                try:
                    out_pi, out_v = self.model.forward(states_eq, smt_values)
                except:
                    try:
                        out_pi, out_v = self.model.forward(states_eq, smt_values)
                    except:
                        out_pi, out_v = self.model.forward(states_eq, smt_values)
            else:
                try:
                    out_pi, out_v, ctxt = self.model.forward(states_eq, smt_values, from_nn_train = True)
                except:
                    try:
                        out_pi, out_v, ctxt = self.model.forward(states_eq, smt_values, from_nn_train = True)
                    except:
                        out_pi, out_v, ctxt = self.model.forward(states_eq, smt_values, from_nn_train = True)
        elif self.args.nnet_type == 'supernet':
            try:
                out_pi, out_v = self.model.forward(states_eq, grad=True, device = self.args.train_device)
            except:
                try:
                    out_pi, out_v = self.model.forward(states_eq, grad=True, device = self.args.train_device)
                except:
                    out_pi, out_v = self.model.forward(states_eq, grad=True, device = self.args.train_device)

        l_pi = self.loss_pi(target_pis, out_pi)
        if self.args.use_value_nn_head:
            l_v, l_v_original = self.loss_v(target_vs, out_v, None)

            total_loss = l_pi + l_v
            #total_loss.register_hook(lambda grad: print(grad))
            #out_pi.register_hook(lambda grad: print(grad))# grad + torch.normal(torch.tensor(0.), torch.tensor(1./((1+epoch)**0.55)))
            if not self.args.oracle:
                total_loss.backward()
                #torch.normal(torch.tensor(0.), torch.tensor(1/((1+epoch)**0.55))).to(self.device))
            if self.args.nnet_type in ['GIN','attention']:

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-2)
                #torch.nn.utils.clip_grad_value_(self.model.parameters(), 1e-1)
                pass
            print('gradient', total_loss.grad, l_pi.grad, l_v.grad)
            if self.args.nnet_type == 'GIN' and self.args.quadratic_mode:
                pass
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-4)
                #torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.)
            if False:#self.args.nnet_type == 'GIN' and self.num_mlp_layers > 1:
                print('grads first linear layer')
                print(self.model.mlps[0].linears[0].weight.grad.mean(), self.model.mlps[0].linears[0].weight.grad.max())
                print('grads last hidden linear layer')
                print(self.model.mlps[-1].linears[-1].weight.grad.mean(), self.model.mlps[-1].linears[-1].weight.grad.max())
                print('weights first linear layer')
                print(self.model.mlps[0].linears[0].weight.mean(), self.model.mlps[0].linears[0].weight.max())
                print('weights last linear layer')
                print(self.model.mlps[-1].linears[-1].weight.mean(),
                      self.model.mlps[-1].linears[-1].weight.max())
                #print('weights hidden lstm layer')
                #print([x.mean() for x in self.model.hidden_lstm.all_weights[0]],
                #      [x.max() for x in self.model.hidden_lstm.all_weights[0]])
                #print('grads hidden lstm layer')
                #print([x.grad.mean() for x in self.model.hidden_lstm.all_weights[0]],
                #      [x.grad.max() for x in self.model.hidden_lstm.all_weights[0]])
                #print('weights equality lstm layer')
                #print([ x.mean() for x in self.model.equality_lstm.all_weights[0]],
                #      [ x.max() for x in self.model.equality_lstm.all_weights[0]])
                #print('grads equality lstm layer')
                #print([ x.grad.mean() for x in self.model.equality_lstm.all_weights[0]],
                #      [ x.grad.max() for x in self.model.equality_lstm.all_weights[0]])
                #print('grads last equality linear layer')
                #print(self.model.equality_linear.linears[-1].weight.grad.mean(),
                #      self.model.equality_linear.linears[-1].weight.grad.max())
                #print('weights last equality linear layer')
                #print(self.model.equality_linear.linears[-1].weight.mean(),
                #      self.model.equality_linear.linears[-1].weight.max())
                #print('\n')

            elif False:# self.args.nnet_type == 'GIN' and self.num_mlp_layers == 1:
                print(self.model.mlps[0].linear.weight.grad.mean(), self.model.mlps[0].linear.weight.grad.max())
                print(self.model.mlps[-1].linear.weight.grad.mean(), self.model.mlps[-1].linear.weight.grad.max())
            #print(self.model.equality_mlp.linears[0].weight.grad.mean())
            else:
                #print(self.model.block_names[0].conv1.weight.grad.mean(), self.model.block_names[0].conv1.weight.grad.max())
                #print(self.model.block_names[-1].conv2.weight.grad.mean(), self.model.block_names[-1].conv2.weight.grad.max())
                pass
            if not self.args.oracle:
                if (not self.args.sat) and self.args.nnet_type != 'graphwenet':
                    print(out_pi[:1], [x[0] for x in out_v])
                else:
                    print(out_pi, out_v)

            #print(l_pi)
            #print('\n')

            #self.model.float()
            if not self.args.oracle:

                self.optimizer.step()
            return l_pi, l_v_original
        else:
            total_loss = l_pi
            if not self.args.oracle:
                total_loss.backward()
                self.optimizer.step()
            return l_pi, 0.

    def loss_pi(self, targets, outputs):
        if type(outputs) == list:
            outputs = outputs[0]
        if outputs.device != self.device:
            outputs = outputs.to(self.device)
        return -torch.sum(targets * torch.log(outputs + 1e-31*torch.ones(outputs.shape,
                                                                         device=self.device,
                                                                         dtype=torch.float))) / targets.size()[0]

    def loss_v(self, targets, outputs, weights = None):
        if outputs.device != self.device:
            outputs = outputs.to(self.device)
        if not self.args.values_01:
            errors_original = (targets - outputs.view(-1))**2
            if weights is not None:
                errors = weights* errors_original
                return torch.sum(errors) / targets.size()[0], torch.sum(errors_original)/targets.size()[0]
            else:
                t = torch.sum(errors_original)/targets.size()[0]
                return t,t
        else:
            errors_original = (targets - outputs.view(-1))**2
            t = torch.sum(errors_original) / targets.size()[0]
            return t, t
            #def expand(t):
            #    return torch.tensor([[1-x, x] for x in t], dtype = torch.float, device = self.device)
            #targets = expand(targets)
            #outputs = expand(outputs)
            #t = -torch.sum(targets * torch.log(outputs + 1e-31 * torch.ones(outputs.shape,
            #                                                            device=self.device,
            #                                                            dtype=torch.float))) / targets.size()[0]
            #return t, t

    def predict(self, s0, smt = None, ctx=None):
        """only used during mcts as a call on a single equation, hence the [0] at the end and the eval call"""
        self.model.eval()
        if smt is not None:
            smt = torch.tensor([[smt]], dtype = torch.float)
        if self.args.nnet_type == 'GIN':
            s0 = [s0]
        if self.args.nnet_type in ['resnet','recnewnet', 'newresnet', 'resnet_double', 'resnet1d', 'GIN',
                                   'satnet', 'graphwenet', 'attention','lstmpe','hanoinet', 'wordgamenet']:
            if self.args.test_mode and self.args.play_device == self.args.train_device:
                s0 = s0.to(self.args.train_device)
                if self.args.rec_mcts:
                    ctx = ctx.to(self.args.train_device)
            if self.args.rec_mcts:
                s0 = [s0, ctx]
            if not self.args.rec_mcts:
                with torch.no_grad():
                    try:
                        pi, v = self.model.forward(s0, smt)
                    except:
                        pi, v = self.model.forward(s0, smt)
                if not self.args.sat and self.args.nnet_type != 'graphwenet':
                    return pi[0], v[0]
                else:
                    return pi, v
            else:
                with torch.no_grad():
                    try:
                        pi, v, ctx = self.model.forward(s0, smt)
                    except:
                        pi, v, ctx = self.model.forward(s0, smt)
                return pi[0], v[0], ctx
        elif self.args.nnet_type == 'supernet':
            with torch.no_grad():
                try:
                    pi, v = self.model.forward(s0, grad=False)
                except:
                    pi, v = self.model.forward(s0, grad=False)
            return pi[0], v[0]

    def detailed_predict(self, eq, state, top_moves = 10):
        pi, v = self.predict(eq.format_state(state))
        pi = pi.numpy()
        moves = pd.DataFrame(np.zeros(eq.getActionSize()))
        moves['probabilities'] = pi
        moves['description'] = [action['description'] for action in eq.actions.values()]
        moves = moves.sort_values('probabilities', ascending= False)
        print('The best' + str(top_moves) + ' moves for ' + state + ' are: ')
        print(moves.iloc[:top_moves])
        print('The value of the equation is : ', str(v))

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

        # if self.args.load_level:
            # self.args.level = checkpoint['level']
            # self.args.update_parameters()

    #def init_log(self, folder, mode='train'):
#
    #    # TODO: What is this for?
    #    logging.getLogger( "tensorflow" ).setLevel( logging.WARNING )
#
    #    # set up logging to file
    #    logging.basicConfig( level=logging.DEBUG,
    #                         format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    #                         datefmt='%m-%d %H:%M',
    #                         filename=folder + f'/log_{mode}.log' )
    #    # define a Handler which writes INFO messages or higher to the sys.stderr
    #    console = logging.StreamHandler()
    #    console.setLevel( logging.INFO )
    #    # add the handler to the root logger
    #    logging.getLogger( '' ).addHandler( console )
#