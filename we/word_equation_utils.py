from cvxopt import matrix, solvers
import cvxopt
import torch
# torch.set_default_tensor_type(torch.HalfTensor)
#import torch_geometric as tg
#from torch_geometric.data import Data
import torch
import numpy as np
import multiprocessing as mp
import random
import math
#from .GINgraph import GINgraph
#from .GINgraph2 import WEtoGraph
from math import sin
from .word_equation_transformations import WordEquationTransformations
from .word_equation import WordEquation
from we.SMTSolver import SMT_eval
import re
import os

class WordEquationUtils(object):
    def __init__(self, args, seed=0):
        self.args = args
        if args.nnet_type == 'GIN':
            self.GINgraph = GINgraph(self.args.VARIABLES, self.args.ALPHABET)

        if args.few_channels:
            self.value_symbol = {symbol: i+1 for i,symbol in enumerate(self.args.VARIABLES)}
            self.value_symbol.update( {symbol: i+1 for i,symbol in enumerate(self.args.ALPHABET)})
            self.value_symbol.update({'.':0})
        self.transformations = WordEquationTransformations(args)
        self.create_action_dict()
        self.create_fast_action_dict()
        if seed is not None:
            seed_everything(seed)
            # self.get_afterstates()

    def are_equal(self, eq1, eq2):
        # print(eq1.get_string_form(), eq2.get_string_form())
        return eq1.get_string_form() == eq2.get_string_form()

    def weighted_length(self, word, weights):
        weights['.'] = 0
        return sum([weights[letter] for letter in word])

    def one_hot_encode_resnet(self, s, length, first = False):
        """encoding is in different dimension order than usual due to CNN reading
        data in dimensions (batch, channels, length), so here each column
        is a one-hot encoding of a letter
        """
        if self.args.few_channels:
            t = torch.zeros(1, 2, 1, length)
            for i, x in enumerate(s):
                ch = 0 if x in self.args.VARIABLES else 1
                t[0,ch,0,i]=self.value_symbol[x]
            return t
        elif self.args.nnet_type != 'resnet1d':
            t = torch.zeros(1, self.args.LEN_CORPUS, 1, length)
            try:
                for i, x in enumerate(s): t[0, self.args.symbol_indices[x], 0, i] = 1.
            except:
                print('ERROR: ', t.shape, s, len(s), len(self.args.symbol_indices))
            return t
        elif not self.args.few_channels:
            #if first:
            t = torch.zeros(1, self.args.LEN_CORPUS+1, length)
            try:
                for i, x in enumerate(s):
                    if x != '=':
                        t[0, self.args.symbol_indices[x], i] = 1.
                    else:
                        t[0, len(self.args.symbol_indices), i] = 1.
            except:
                print('ERROR: ', t.shape, s, len(s), len(self.args.symbol_indices), x, i)
            return t

    def encode_newresnet(self, s, length, height =None, first = False):
        if not self.args.recurrent_blocks:
            height = max(len(self.args.variable_indices), len(self.args.alphabet_indices))
        t = torch.zeros(1, 2, height, length)
        for i, x in enumerate(s):
            if x in self.args.VARIABLES:
                t[0, 0, self.args.variable_indices[x], i] = 1.
            elif x != '.':
                t[0, 1, self.args.alphabet_indices[x], i] = 1.
        #t[0, 2, [2*i+1 for i in range(int(height/2))], :] = -1.
        return t


    def one_hot_encode_dnc(self, s):
        """encoding is in different dimension order than usual due to CNN reading
        data in dimensions (batch, channels, length), so here each column
        is a one-hot encoding of a letter
        """
        t = torch.zeros(1, len(s), self.args.LEN_CORPUS)
        for i, x in enumerate(s): t[0, i, self.args.symbol_indices[x]] = 1.
        return t

    def format_lc_part_in_cnn_style(self, eq, length):

        tensor = torch.zeros((1, 1 + len(self.args.VARIABLES) + len(self.args.ALPHABET), 2, length),
                             dtype=torch.float)

        sign = int(eq.ell < 0)
        tensor[0, 0, sign, :eq.ell] = 1.

        for i, x in enumerate(self.args.VARIABLES):
            coef_variable = eq.coefficients_variables_lc[x]
            sign = int(coef_variable < 0)
            tensor[0, 1 + i, sign, : coef_variable] = 1.

        for i, x in enumerate(self.args.ALPHABET):
            weight_constant = eq.weights_constants_lc[x]
            tensor[0, 1 + len(self.args.VARIABLES) + i, 0, : weight_constant] = 1.
        return tensor

    @staticmethod
    def sin_PE(pos, dim, model_dim):
        return sin(pos / (10000 ** (dim / model_dim)))

    @staticmethod
    def cos_PE(pos, dim, model_dim):
        return sin(pos / (10000 ** (dim / model_dim)))

    def encode(self, x):
        """x is a tensor of shape batch_size x length x w.
        returns a tensor of shape batch_size x length x 2w
        with positional encodings on the last w coordinates"""
        mdim = x.shape[2]
        t = [[self.sin_PE(pos, dim, mdim) for dim in range(mdim)] for pos in range(x.shape[1])]
        t = torch.tensor(t)
        return torch.cat([x, t], dim=2)


    def format_state(self, eq, device='cpu'):
        if self.args.oracle:
            return torch.tensor([[[0.]]])
        if self.args.nnet_type in ['attention']:
            def attention_fun(eq):
                mdim = 5 + len(self.args.VARIABLES) + len(self.args.ALPHABET)
                t = torch.zeros(1, len(eq), 3*mdim, dtype =torch.float)
                if eq == 'error':
                    return t
                t[0, 0, 0] = 1.
                found_eq = False
                for i, symbol in enumerate(eq):
                    if symbol != '=':
                        t[0, i, 4 + self.args.symbol_indices[symbol]]=1.
                        t[0, i, 2 + int(found_eq)] = 1.
                    else:
                        found_eq=True
                        t[0, i, 1] = 1.
                    t[0, i, mdim :] = torch.tensor([self.sin_PE(i, dim, mdim) for dim in range(mdim)] +
                                                   [self.cos_PE(i, dim, mdim) for dim in range(mdim)])
                return t

            if False:
                afterstates = self.get_afterstates(eq)
                transf_aft = []
                for e in afterstates:
                    if e == 0:
                        mdim = 4 + len(self.args.VARIABLES) + len(self.args.ALPHABET)
                        t = torch.zeros(1, 1, 2 * mdim, dtype=torch.float)
                        transf_aft.append(t)
                    else:
                        transf_aft.append(attention_fun(e.w))
                out = [attention_fun(eq.w), transf_aft]
                return out
            return attention_fun(eq.w)
        if self.args.nnet_type in ['lstmpe']:
            tensor = torch.zeros(1, len(eq.w), len(self.args.symbol_indices))
            for i, x in enumerate(eq.w):
                tensor[0, i, self.args.symbol_indices[x]] = 1.
            return tensor
        if self.args.nnet_type in ['resnet', 'resnet_double', 'newresnet']:# and \
                #('noagg' in self.args.folder_name or 'meanagg' in self.args.folder_name):

            s1, s2 = eq.w.split('=')


            maxlen = self.args.SIDE_MAX_LEN

            tensor1, tensor2 = self.one_hot_encode_resnet(s1, maxlen).to(device), self.one_hot_encode_resnet(s2, maxlen).to(device)
            tensor = torch.cat((tensor1, tensor2), dim=2).to(device)

            if self.args.use_length_constraints:
                tensor_lc = self.format_lc_part_in_cnn_style(eq, maxlen)

                tensor = torch.cat((tensor, tensor_lc), dim = 1).to(device)

            return tensor

        if self.args.nnet_type == 'resnet1d':
            s1, s2 = eq.w.split('=')
            maxlen = 2*self.args.SIDE_MAX_LEN+1

            tensor = self.one_hot_encode_resnet(eq.w, maxlen).to(device)

            if self.args.use_length_constraints:
                tensor_lc = self.format_lc_part_in_cnn_style(eq, maxlen)

                tensor = torch.cat((tensor, tensor_lc), dim=1).to(device)

            return tensor

        if self.args.nnet_type in [ 'recnewnet', 'newresnet'] and \
                ('fullagg' in self.args.folder_name):

            s1, s2 = eq.w.split('=')
            if self.args.recurrent_blocks or self.args.maxpool_input:
                #print('hi')
                maxlen = max(len(s1), len(s2))#self.args.SIDE_MAX_LEN
                height = max(len([x for x in self.args.VARIABLES if x in eq.w]),
                                 len([x for x in self.args.ALPHABET if x in eq.w]))
            else:
                maxlen = self.args.SIDE_MAX_LEN
                height = None
            tensor1, tensor2 = self.encode_newresnet(s1, maxlen, height).to(device), self.encode_newresnet(s2, maxlen, height).to(device)
            tensor = torch.cat((tensor1, tensor2), dim=1).to(device)

            return tensor

        if self.args.nnet_type == 'GIN':
            g = WEtoGraph(eq.w, self.args.VARIABLES, self.args.ALPHABET)
            return g.G
        if False:
            g = GINgraph(self.args.VARIABLES, self.args.ALPHABET)
            g.get_graph(eq.w)
            return g

        if self.args.nnet_type == 'graphwenet':
            g = WEtoGraph(eq.w, self.args.VARIABLES, self.args.ALPHABET)
            if True:
                eq.we_to_graph = g
            return g

        if self.args.nnet_type == 'supernet':
            s1, s2 = eq.w.split('=')
            num_chunks = int(math.ceil(max(len(s1), len(s2)) / self.args.chunk_size))
            maxlen = self.args.chunk_size
            result = torch.zeros(1,num_chunks, self.args.LEN_CORPUS, 2, maxlen, device = 'cpu', dtype = torch.float32)

            chunks = self.split_into_chunks(eq.w, num_chunks)
            for i, chunk in enumerate(chunks):
                tensor1 = self.one_hot_encode_resnet(chunk[0], maxlen).to(device)
                tensor2 = self.one_hot_encode_resnet(chunk[1], maxlen).to(device)
                tensor = torch.cat((tensor1, tensor2), dim=2).to(device)
                result[0, i, :, :,  :] = tensor

            if self.args.use_length_constraints:
                tensor_lc = self.format_lc_part_in_cnn_style(eq, maxlen)
                tensor = torch.cat((tensor, tensor_lc), dim = 1).to(device)

            return result

        elif self.args.nnet_type == 'pgnn':
            converter = WEtoGraph(eq.w, self.args, self.args.node_dict)
            dists_array = converter.precompute_dist_data(converter.G.edge_index, converter.G.num_nodes, approximate=self.args.pgnn_approximate)
            converter.G.dists = torch.tensor(dists_array, dtype=torch.float)
            converter.preselect_anchor(
                data = converter.G,
                #layer_num= ,
                #anchor_num= ,
                #anchor_size_num= ,
                device ='cpu'
            )
            return converter.G


        elif self.args.nnet_type == 'dnc' or self.args.nnet_type == 'sam':
            """takes equation and one-hot-encodes it for input to CNN"""
            tensor = self.one_hot_encode_dnc(eq.w).to(device)
            return tensor

    def split_into_chunks(self, w, num_chunks):
        """Returns a list of smaller equations"""

        def get_subword(i, word):
            sw = word[-(i+1)*cs: -i*cs] if i != 0 else word[-cs:]
            if len(sw) == 0:
                return '.'
            else:
                return sw
        cs = self.args.chunk_size
        s1, s2 = w.split('=')
        chunks = [[get_subword(i, s1), get_subword(i, s2)] for i in range(num_chunks)]
        return chunks

    def check_sat_wrap(self, eq, time = 500):
        eq = self.check_satisfiability(eq, time)
        if eq.sat == 0:
            eq.sat = 'unknown'
        elif eq.sat == -1:
            eq.sat = 0
        return eq

    def get_eq_list(self, eq):
        eq_split = eq.w.split('=')
        lefts_w = eq_split[0].split('+')
        rights_w = eq_split[1].split('+')
        assert len(lefts_w) == len(rights_w)
        w_list = [lefts_w[i]+'='+rights_w[i] for i in range(len(lefts_w))]
        eq_list = [WordEquation(args=self.args, w=w) for w in w_list]
        return eq_list

    def check_satisfiability(self, eq, time=500, smt_time=None):
        if '+' in eq.w:
            eq_list = self.get_eq_list(eq)
            sats = []
            for e in eq_list:
                sats.append(self.main_check_satisfiability(e))
            if -1 in sats:
                eq.sat = -1
            elif 0 in sats:
                eq.sat = 0
            else:
                eq.sat = 1
            return eq
        else:
            return self.main_check_satisfiability(eq, smt_time=smt_time)

    def main_check_satisfiability(self, eq, time=500, num= 0, smt_time=None):

        if self.args.use_length_constraints:
            if not self.LC_is_sat(eq):
                eq.sat = -1
                return eq

        w_split = eq.w.split('=')
        if w_split[0] == w_split[1]:
            if not self.args.use_length_constraints:
                eq.sat = 1
                return eq
            else:
                if self.LC_is_sat(eq):
                    # if True:
                    eq.sat = 1
                    return eq
                else:
                    eq.sat = -1
                    return eq
        else:
            # the following if can be done with "'X' not in eq.w" assuming eq.w is in normal form, which should be the case
            if self.is_constant(eq.w):
                eq.sat = -1
                return eq
            else:
                if w_split[0] in self.args.VARIABLES and self.is_constant(w_split[1]):
                    if self.treat_case_variable_side(eq, w_split[0], w_split[1]):
                        eq.sat = 1
                        return eq
                    else:
                        eq.sat = -1
                        return eq
                if w_split[1] in self.args.VARIABLES and self.is_constant(w_split[0]):
                    if self.treat_case_variable_side(eq, w_split[1], w_split[0]):
                        eq.sat = 1
                        return eq
                    else:
                        eq.sat = -1
                        return eq

            # now we know equation is not constant and no side is a single variable
            eq = self.unsat_by_incompatible_extremes(eq)
            if eq.sat == -1:
                return eq

            # if False:
            if self.args.check_LP:
                if not self.LP_is_sat(eq):
                    eq.sat = -1
                    return eq

            if self.args.use_oracle:
                out = SMT_eval(self.args, eq, timeout=smt_time)
                if out > 0:
                    eq.sat = 1
                    return eq
                elif out < 0:
                    eq.sat = -1
                    return eq
                else:
                    eq.sat=0
        return eq



    def LP_is_sat(self, eq):
        cvxopt.solvers.options['show_progress'] = False

        # todo: can the next call be removed (for efficiency?)
        eq.find_LP()
        b = matrix(eq.lp['bb'])
        G = matrix(eq.lp['GG'])
        h = matrix(eq.lp['hh'])
        A = matrix(eq.lp['AA'])
        c = matrix(eq.lp['cc'])
        # print(b, 'yyy')
        sol = solvers.lp(c, G, h, A, b, solver='glpk', options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
        # print(sol['x'])
        return not sol['x'] is None

    def treat_case_variable_side(self, eq, var_side, other_side):
        """

        :param eq:
        :param var_side:
        :param other_side:
        :return: True or False depending on wheter a WEWLC with word equation of the form X=ct has a solution or not
        """
        assert len(var_side) == 1
        if self.args.use_length_constraints:
            eq = self.update_LC_with_var_subtitution(eq, var_side, other_side)
            if self.LC_is_sat(eq):
                eq.sat = 1
                return True
            else:
                eq.sat = -1
                return False
        else:
            eq.sat = 1
            return True

    def LC_is_sat(self, eq):
        sats = []
        for num in range(len(eq.coefficients_variables_lc)):
            sats.append(self.main_LC_is_sat(eq, num))
        if False in sats:
            return False
        else:
            return True

    def main_LC_is_sat(self, eq, num=0):
        """
        Determines if the length constraint of eq is satisfiable assuming the word equation is empty
        lc always has the form: linear_combination_of_weighted_lengths_of_variables >= ell
        """
        if eq.ell[num] <= 0:
            # in this case we can set all variables to be empty
            return True
        elif any([eq.coefficients_variables_lc[num][var] > 0 for var in self.args.VARIABLES]):
            # we can make a variable with positive coefficient as large as needed
            return True
        else:
            # the only option left is that all coefficients are <= 0 and ell > 0, so unsolvable
            return False
    def update_LC_with_var_subtitution(self, eq, var, subst):
        for _ in range(len(eq.coefficients_variables_lc)):
            eq = self.main_update_LC_with_var_subtitution(eq, var, subst, _)
        return eq

    def main_update_LC_with_var_subtitution(self, eq, var, subst,num):
        # assert var in self.args.VARIABLES
        # assert len(self.args.VARIABLES) == 5
        eq.ell[num] = eq.ell[num] - eq.coefficients_variables_lc[num][var] * self.count_ctts(subst)
        # eq.coefficients_variables_lc.update({
        #     x: eq.coefficients_variables_lc[x] + eq.coefficients_variables_lc[var] * subst.count(x) for x in self.args.VARIABLES if x != var
        # })
        eq.coefficients_variables_lc[num].update({
            var: 0
        })
        # if len(eq.not_normalized) == 0:
        #     eq.not_normalized = '-treated case var=...'
        # else:
        #     eq.not_normalized += '-treated case var=...'
        return eq

    def count_ctts(self, word):
        return len([1 for x in word if x in self.args.ALPHABET])

    def unsat_by_incompatible_extremes(self, eq):
        w_l, w_r = eq.w.split('=')
        if w_l[0] != w_r[0]:
            if w_l[0] not in self.args.VARIABLES and w_r[0] not in self.args.VARIABLES:
                eq.sat = -1
                return eq
        if w_l[-1] != w_r[-1]:
            if w_l[-1] not in self.args.VARIABLES and w_r[-1] not in self.args.VARIABLES:
                eq.sat = -1
                return eq
        return eq

    def is_constant(self, word):
        return all([x in self.args.ALPHABET + ['.'] for x in word])



    def delete_var(self, eq, eq_side, word_side):
        try:
            eq_split = eq.w.split('=')
            var = eq_split[eq_side][word_side]
            # let = eq_split[1 - side][0]
            if var not in self.args.VARIABLES:
                eq.attempted_wrong_move = True
                return eq
            else:
                new_w = re.sub(var, '', eq.w)
                eq.w = new_w
                eq.attempted_wrong_move = False
                return eq
        except:
            eq.attempted_wrong_move = True
            return eq

    def compress(self, eq):
        if self.args.ALPHABET[-1] in eq.w:
            return eq
        left_allowed = {letter: True for letter in self.args.ALPHABET[:-1]}
        right_allowed = {letter: True for letter in self.args.ALPHABET[:-1]}
        for letter in self.args.ALPHABET:
            split = eq.w.split(letter)
            left_split = [x[-1:] for x in split[:-1]]  # symbols on the left of letter
            right_split = [x[:1] for x in split[1:]]
            if any([x in self.args.VARIABLES for x in left_split]):
                right_allowed[letter] = False
            if any([x in self.args.VARIABLES for x in right_split]):
                left_allowed[letter] = False
        allowed_tuples = [l1 + l2 for l1, val1 in left_allowed.items() if val1 for l2, val2 in right_allowed.items()
                          if
                          val2]
        if len(allowed_tuples) == 0:
            eq.attempted_wrong_move = True
            return eq
        else:
            counts = [[eq.w.count(pair), pair] for pair in allowed_tuples]
            if all([x == 0 for x in counts]):
                eq.attempted_wrong_move = True
                return eq
            else:
                counts.sort()
                new_w = re.sub(counts[-1][-1], self.args.ALPHABET[-1], eq.w)
                eq.w = new_w
                return eq

    def move(self, eq, eq_side, word_side):
        # side = 0 if side == 'left' else 1
        eq_split = eq.w.split('=')
        word_side_aux = 0 if word_side == 0 else -1
        var = eq_split[eq_side][word_side_aux]
        let = eq_split[1 - eq_side][word_side_aux]
        if var not in self.args.VARIABLES:
            eq.attempted_wrong_move = True
            return eq
        else:
            new_w = re.sub(var, ((1 - word_side) * let) + var + ((word_side) * let), eq.w)
            eq.w = new_w
            eq.attempted_wrong_move = False
            return eq

    def act(self, eq, action_num, verbose=0):
        eq.attempted_wrong_move = False
        new_eq = eq.deepcopy()
        action = self.actions[action_num]
        type_of_action = action['type']

        if type_of_action == 'delete':
            eq_side = action['eq_side']
            word_side = action['word_side']
            new_eq = self.delete_var(new_eq, eq_side, word_side)

        if type_of_action in ['move_0', 'move_-1']:
            eq_side = action['eq_side']
            word_side = action['word_side']
            new_eq = self.move(new_eq, eq_side, word_side)

        if new_eq.attempted_wrong_move:
            return new_eq

        new_eq.not_normalized = new_eq.get_string_form()
        new_eq = self.transformations.normal_form(new_eq)

        if self.are_equal(new_eq, eq):
            new_eq.attempted_wrong_move = True
            return new_eq
        else:
            eq.attempted_wrong_move = False
            new_eq.get_id()
            return new_eq

    def get_afterstates(self, eq):
        afterstates = []
        for action_idx in self.actions.keys():
            new_eq = self.act(eq, action_idx)
            if new_eq.attempted_wrong_move:
                afterstates.append(0)
            else:
                afterstates.append(new_eq)
        return afterstates

    def get_valid_actions(self, eq):
        afterstates = self.get_afterstates(eq)
        valid_actions = torch.ones(len(self.actions), dtype=torch.float, requires_grad=False,
                                   device=self.args.play_device)
        for i, x in enumerate(afterstates):
            if type(x) == int:
                valid_actions[i] = 0.
        return valid_actions, afterstates

    def create_action_dict(self):
        """
        Creates a dictionary with possible actions.
        Each entry is a dictionary with different entries depending on the type of action
        """
        actions_delete = [{'description': f"delete_{eq_side}_{word_side} ",
                           'type': 'delete',
                           'eq_side': eq_side,
                           'word_side': word_side,
                           } for eq_side in [0, 1] for word_side in [0, 1]]
        actions_move = [{'description': f"move_{eq_side}_{word_side}",
                         'type': f'move_{word_side}',
                         'eq_side': eq_side,
                         'word_side': word_side,
                         } for eq_side in [0, 1] for word_side in [0, 1]]

        self.actions = actions_delete + actions_move
        self.actions = {i: self.actions[i] for i in range(len(self.actions))}

    def get_action_size(self):
        self.create_action_dict()
        return len(self.actions)

    def create_fast_action_dict(self):
        'returns a dictionary that allows easy access to the number index each action has in self.action'
        alph = self.args.ALPHABET
        vars = self.args.VARIABLES
        fast_dict = {}
        fast_dict['delete'] = {'left': len(alph), 'right': len(alph)}
        fast_dict['move'] = {'left': len(alph), 'right': len(alph)}
        self.fast_dict = fast_dict



def seed_everything(seed):
    # https://www.kaggle.com/hmendonca/fold1h4r3-arcenetb4-2-256px-rcic-lb-0-9759 cells 45-50
    #print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
