import numpy as np
import math
import re
import random
import networkx as nx
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
import random
from copy import deepcopy
import torch
import re
import numpy as np
import os
from pickle import Pickler, Unpickler
import logging
#gym.logger.set_level(logging.INFO)


VARS =list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿĀāĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ')
FVARS = list('ĀāĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ')


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

            if not self.args.bounded_model:
                maxlen = max([len(x) for x in [s1, s2]]
                             + [eq.ell]
                             + [eq.coefficients_variables_lc[x] for x in self.args.VARIABLES]
                             + [eq.weights_constants_lc[x] for x in self.args.ALPHABET])
            else:
                maxlen = self.args.SIDE_MAX_LEN

            tensor1, tensor2 = self.one_hot_encode_resnet(s1, maxlen).to(device), self.one_hot_encode_resnet(s2, maxlen).to(device)
            tensor = torch.cat((tensor1, tensor2), dim=2).to(device)

            if self.args.use_length_constraints:
                tensor_lc = self.format_lc_part_in_cnn_style(eq, maxlen)

                tensor = torch.cat((tensor, tensor_lc), dim = 1).to(device)

            return tensor

        if self.args.nnet_type == 'resnet1d':
            s1, s2 = eq.w.split('=')

            if not self.args.bounded_model:
                maxlen = max([len(x) for x in [s1, s2]]
                             + [eq.ell]
                             + [eq.coefficients_variables_lc[x] for x in self.args.VARIABLES]
                             + [eq.weights_constants_lc[x] for x in self.args.ALPHABET])
            else:
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

    def check_satisfiability(self, eq, time=500):

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

            if self.args.z3_is_final:
                try:
                    z3 = WeToSmtZ3(self.args.VARIABLES, self.args.ALPHABET, time)
                    out = z3.eval(eq.w)
                    z3.solver.reset()
                    if out > 0:
                        eq.sat = 1
                        return eq
                    elif out < 0:
                        eq.sat = -1
                        return eq
                except:
                    try:
                        z3 = WeToSmtZ3(self.args.VARIABLES, self.args.ALPHABET, time)
                        out = z3.eval(eq.w)
                        z3.solver.reset()
                        if out > 0:
                            eq.sat = 1
                            return eq
                        elif out < 0:
                            eq.sat = -1
                            return eq

                    except:
                        z3 = WeToSmtZ3(self.args.VARIABLES, self.args.ALPHABET, time)
                        out = z3.eval(eq.w)
                        z3.solver.reset()
                        if out > 0:
                            eq.sat = 1
                            return eq
                        elif out < 0:
                            eq.sat = -1
                            return eq


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
        """
        Determines if the length constraint of eq is satisfiable assuming the word equation is empty
        lc always has the form: linear_combination_of_weighted_lengths_of_variables >= ell
        """
        if eq.ell <= 0:
            # in this case we can set all variables to be empty
            return True
        elif any([eq.coefficients_variables_lc[var] > 0 for var in self.args.VARIABLES]):
            # we can make a variable with positive coefficient as large as needed
            return True
        else:
            # the only option left is that all coefficients are <= 0 and ell > 0, so unsolvable
            return False

    def update_LC_with_var_subtitution(self, eq, var, subst):
        # assert var in self.args.VARIABLES
        # assert len(self.args.VARIABLES) == 5
        eq.ell = eq.ell - eq.coefficients_variables_lc[var] * self.count_ctts(subst)
        # eq.coefficients_variables_lc.update({
        #     x: eq.coefficients_variables_lc[x] + eq.coefficients_variables_lc[var] * subst.count(x) for x in self.args.VARIABLES if x != var
        # })
        eq.coefficients_variables_lc.update({
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


class WordEquationTransformations(object):

    # todo: throughout all project I do a lot of split('='). Optimize?

    def __init__(self, args):

        self.args = args

    def normal_form(self, eq, minimize=True, mode='play'):
        if self.args.use_normal_forms:
            if minimize is None:
                minimize = not self.args.use_true_symmetrization
            # print(self.eq.lp['abelian_form'].items())
            if self.args.automatic_compress and mode != 'generation':
                new_eq = ''
                while new_eq != eq.w:
                    new_eq = eq.w
                    eq = self.compress(eq)

            if (self.args.generation_mode != 'constant_side') and (not self.args.quadratic_mode):
                new_w_split = eq.w.split('=')
                new_w_split.sort()
                new_w = new_w_split[0] + '=' + new_w_split[1]
                eq.w = new_w

            if self.args.use_length_constraints:
                eq = self.treat_lc(eq, mode)

            if mode != 'generation':
                eq = self.del_pref_suf(eq)

            if minimize and self.args.use_normal_forms:
                auto = self.get_automorphism(eq)
                eq = self.apply_automorphism(eq, auto, mode)
            return eq
        else:
            if mode != 'generation':
                eq = self.del_pref_suf(eq)
            return eq

            def normal_form_large_version(w, alphabet, variables, num_digits):

                def substitution_fun(char, auto, size_auto, letter_list):
                    if char not in auto.keys():
                        new_char = letter_list[size_auto]
                        auto[char] = new_char
                        size_auto += 1
                    else:
                        new_char = auto[char]
                    return new_char, auto, size_auto

                alph_auto = {}
                size_alph_auto = 0
                var_auto = {}
                size_var_auto = 0
                new_w = ''
                pointer = 0
                while pointer < len(w):
                    char = w[pointer]
                    if char in alphabet:
                        pointer += 1
                        new_char, alph_auto, size_alph_auto = substitution_fun(char, alph_auto, size_alph_auto,
                                                                               alphabet)
                        new_w += new_char
                    elif char == '=':
                        pointer += 1
                        new_w += '='
                    else:
                        char = w[pointer: pointer + num_digits]
                        pointer += num_digits
                        new_char, var_auto, size_var_auto = substitution_fun(char, var_auto, size_var_auto, variables)
                        new_w += new_char
                return new_w

            eq.w = normal_form_large_version(eq.w, self.args.ALPHABET, self.args.VARIABLES, self.args.num_var_digits_large_mode)

        return eq

    def compress(self, eq):
        if self.args.ALPHABET[-1]  in eq.w:
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
        allowed_tuples = [l1 + l2 for l1, val1 in left_allowed.items() if val1 for l2, val2 in right_allowed.items() if
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

    def get_automorphism(self, eq):

        order = ''.join(OrderedDict.fromkeys(eq.w).keys())

        char_auto = re.sub( '[' + ''.join(self.args.VARIABLES + ['.', '=']) + ']', '' , order)
        var_auto = re.sub('[' + ''.join(self.args.ALPHABET + ['.', '=']) + ']', '', order)

        for x in self.args.ALPHABET:
            if x not in char_auto:
                char_auto += x

        for x in self.args.VARIABLES:
            if x not in var_auto:
                var_auto += x

        automorphism = {'.':'.', '=':'='}
        automorphism.update({var_auto[i]: self.args.VARIABLES[i] for i in range(len(var_auto))})
        automorphism.update({char_auto[i]: self.args.ALPHABET[i] for i in range(len(char_auto))})
        return automorphism

    def apply_automorphism_to_dictionary(self,dictionary, auto):
        return {auto[key]: value for key,value in dictionary.items()}

    def apply_automorphism(self, eq, auto, mode = 'play'):
        eq.w = eq.w.translate(str.maketrans(auto))
        if self.args.use_length_constraints:
            eq.weights_constants_lc = self.apply_automorphism_to_dictionary(eq.weights_constants_lc, auto)
            # { auto[key] : value for key, value in eq.weights_constants_lc.items() }
            eq.coefficients_variables_lc = self.apply_automorphism_to_dictionary(eq.coefficients_variables_lc, auto)
            # { auto[key] : value for key, value in eq.coefficients_variables_lc.items() }
            if mode == 'generation':
                eq.num_letters_per_variable = { auto[key]: self.apply_automorphism_to_dictionary(value, auto) for
                                                key, value in eq.num_letters_per_variable.items() }
        # eq.candidate_sol = { auto[key[0]] + key[1]: value for key, value in eq.candidate_sol.items() }
        return eq

    def del_pref_suf(self, eq):
        """note this does not change the LP (abelian form)"""

        def length_longest_common_prefix(strs):

            if not strs: return 0
            shortest_str = min(strs, key=len)
            length = 0
            for i in range(len(shortest_str)):
                if all([x.startswith(shortest_str[:i + 1]) for x in strs]):
                    length += 1
                else:
                    break

            return length

        def length_longest_common_sufix(strs):

            if not strs: return 0
            shortest_str = min(strs, key=len)
            length = 0
            for i in range(len(shortest_str)):
                if all([x.endswith(shortest_str[-i - 1:]) for x in strs]):
                    length += 1
                else:
                    break

            return length

        w_split = eq.w.split('=')
        l_prefix = length_longest_common_prefix(w_split)
        w_l = w_split[0][l_prefix:]
        w_r = w_split[1][l_prefix:]

        l_suf = length_longest_common_sufix([w_l, w_r])
        if l_suf > 0:
            w_l = w_l[:-l_suf]
            w_r = w_r[:-l_suf]

        if len(w_l) == 0:
            w_l = '.'
        if len(w_r) == 0:
            w_r = '.'

        eq.w = w_l + '=' + w_r
        return eq


    def treat_lc(self, eq, mode = 'play'):
        # if eq.ell <= 0:
        # if mode == 'generation':
        #     if all([eq.coefficients_variables_lc[x] <= 0 for x in self.args.VARIABLES]):
        #         for x in self.args.VARIABLES:
        #             # todo: a bit unefficient since we checked <= for all vars before. improve?
        #             if eq.coefficients_variables_lc[x] < eq.ell:
        #                 if x in eq.w:
        #                     eq.w = re.sub(x, '', eq.w)
        #                     eq.coefficients_variables_lc[x] = 0
        #                     # eq.lp['abelian_form'][x] = 0
        #             eq.ell = 0
        #         # return eq
        #
        #     elif all([eq.coefficients_variables_lc[x] >= 0 for x in self.args.VARIABLES]):
        #         if eq.ell <= 0:
        #             eq.nullify_lc()
        #             # return eq
        #
        #     elif all([eq.coefficients_variables_lc[x] == 0 for x in self.args.VARIABLES]):
        #         eq.ell = 0
        #         # return eq
        #
        #
        # else:
        for x in self.args.VARIABLES:
            coef = eq.coefficients_variables_lc[x]
            if coef != 0:
                if x not in eq.w:
                    if coef > 0:
                        eq.nullify_lc()
                    elif coef < 0:
                        eq.coefficients_variables_lc[x] = 0

        for x in self.args.ALPHABET:
            coef = eq.weights_constants_lc[x]
            if coef != 0:
                if x not in eq.w:
                    eq.weights_constants_lc[x] = 1

        return eq


def seed_everything(seed):
    # https://www.kaggle.com/hmendonca/fold1h4r3-arcenetb4-2-256px-rcic-lb-0-9759 cells 45-50
    #print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class WordEquationMovesQuadratic(object):

    def __init__(self, args, seed=0):
        self.args = args
        self.utils = WordEquationUtils(args)
        self.transformations = WordEquationTransformations(args)
        self.create_action_dict()
        self.create_fast_action_dict()
        if seed is not None:
            seed_everything(seed)

            # self.get_afterstates()

    def delete_var(self, eq, eq_side, word_side):
        eq_split = eq.w.split('=')
        #print(eq_split, eq_side, word_side)
        var = eq_split[eq_side][word_side]
        #let = eq_split[1 - side][0]
        if var not in self.args.VARIABLES:
            eq.attempted_wrong_move = True
            return eq
        else:
            new_w = re.sub(var, '', eq.w)
            eq.w = new_w
            eq.attempted_wrong_move = False
            return eq

    def compress(self, eq):
        if self.args.ALPHABET[-1]  in eq.w:
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
        allowed_tuples = [l1 + l2 for l1, val1 in left_allowed.items() if val1 for l2, val2 in right_allowed.items() if
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
        let = eq_split[1-eq_side][word_side_aux]
        if var not in self.args.VARIABLES:
            eq.attempted_wrong_move = True
            return eq
        else:
            new_w = re.sub(var, ((1-word_side)*let)+var+((word_side)*let), eq.w)
            eq.w =new_w
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

        if self.utils.are_equal(new_eq, eq):
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
        valid_actions = torch.ones(len(self.actions), dtype=torch.float, requires_grad=False, device=self.args.play_device)
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
                           } for eq_side in [0,1]for word_side in [0,-1]]
        actions_move = [{'description': f"move_{eq_side}_{word_side}",
                           'type': f'move_{word_side}',
                           'eq_side': eq_side,
                           'word_side': word_side,
                           } for eq_side in [0,1] for word_side in [0,1]]

        self.actions =  actions_move  + actions_delete
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





class WordEquation(object):
    """
    w : word equation, e.g. Xab = abX
    lc : length constraint: an integer l
    coefficients_variables_lc : a dictionary of integers c_i, one for each variable. This and previous stands for sum_i l_ix_i geq l
    constant_weights : dictionary with a nonzero integer for each letter in the alphabet
    sat: if the welc is sat (+1) or unsat (-1) or unknown (0)
    recorded_variable_length:
            used in EquationGenerator to keep track of the length of the variables in the solution used to construct the equation
            dictionary: variable x to dictionary_x, where dictionary_x maps constant to number of times the constant appears in x
    attempted_wrong_move: if agent has attempted an invalid move. Used only during MCTS to mask invalid moves
    is_constant_left/right: if True then equation has a constant left/right side. If False then the left/right side may or may not be constant.
                            Used for optimization during automorphisms operations
    """
    def __init__(self, args, w='a=a', seed=None):
        if seed is not None:
            seed_everything(seed)


        self.args = args
        self.w = w
        self.ell = 0
        self.coefficients_variables_lc = dict({x: 0 for x in self.args.VARIABLES})
        self.weights_constants_lc = dict({x: 1 for x in self.args.ALPHABET})
        self.lp = {}
        self.sat = 0 if not args.values_01 else 'unknown'
        self.ctx = None
        # The next are only used in equation generator
        self.num_letters_per_variable = {x: {y: 0 for y in self.args.ALPHABET} for x in self.args.VARIABLES}
        self.used_alphabet_only_equation = []
        self.used_alphabet = []
        self.used_variables_only_equation = []
        self.used_variables = []
        self.dictionary_translation= self.get_dict_translation()
        self.initial_state  = False
        # self.update_used_symbols()
        if self.args.check_LP:
            self.find_LP()

        self.attempted_wrong_move = False

        self.candidate_sol = dict({x + side : '' for x in self.args.VARIABLES for side in ['l', 'r']})
        self.not_normalized = ''
        self.level = 0
        self.id = self.get_string_form() + str(round(random.random(), 5))[1:]

    def get_id(self):
        self.id = self.get_string_form() + str(round(random.random(), 5))[1:]

    def get_dict_translation(self):
        dc = {s: s for s in VARS}
        for i,s in enumerate(FVARS):
            dc[s] = '+V' +'+'
        return dc

    def update_used_symbols(self):
        self.used_variables = [x for x in self.args.VARIABLES if x in self.w]
        self.available_vars = [x for x in self.args.VARIABLES if x not in self.used_variables]
        self.used_alphabet = [x for x in self.args.ALPHABET if x in self.w]
        self.available_alphabet = [x for x in self.args.ALPHABET if x not in self.used_alphabet]

    def find_LP(self):
        """
        constructs dictionary with
        'abelian_form'
        'A'
        'G'
        'h'
        'c'
        :return:
        """

        w_split = self.w.split('=')
        w_left = w_split[0]
        w_right = w_split[1]


        ab_form = {x : w_left.count(x) - w_right.count(x) for x in self.args.VARIABLES + self.args.ALPHABET}
        self.lp['abelian_form'] = ab_form
        variable_vec = [ab_form[var] for var in self.args.VARIABLES]
        zero_vec = len(self.args.VARIABLES)*[0.]

        for i in range(len(self.args.ALPHABET)):
            if i == 0:
                A = [ variable_vec + (len(self.args.ALPHABET)-1) * zero_vec ]
            else:
                A += [i* zero_vec +  variable_vec  + (len(self.args.ALPHABET) -i-1) * zero_vec]
        self.lp['AA'] = np.array(A)
        num_vars = len(self.args.VARIABLES) * len(self.args.ALPHABET)
        self.lp['cc'] = num_vars * [0.]
        self.lp['hh'] = num_vars * [0.]
        G = np.zeros((num_vars, num_vars))

        for i in range(num_vars):
            G[i][i] = -1.

        self.lp['GG'] = G
        # print(ab_form)
        try:
            self.lp['bb'] =[float(-ab_form[x]) for x in self.args.ALPHABET]
        except:
            print(self.lp)
            print(self.args.ALPHABET)

        # print(self.lp['b'])
        # print(self.lp['b'])
        # print(G.shape, self.lp['A'].shape)

    # def delete_var_from_lp(self, var):
    #     self.lp[var] = 0
    #     var_index = np.argmax([x == var for x in self.args.VARIABLES])
    #     num_vars = len(self.args.VARIABLES)
    #     for i in range(len(A)):
    #         self.lp['A'][i][i*num_vars + var_index] = 0.

    def get_string_form(self):
        if self.args.use_length_constraints:
            lc_string = ''
            for x in self.args.VARIABLES:
                lc_string += str(self.coefficients_variables_lc[x])
            weight_string = ''
            for x in self.args.ALPHABET:
                weight_string += str(self.weights_constants_lc[x])
            return self.w + ':' + str(self.ell) + ':' + lc_string + ':' + weight_string
        else:
            return self.w

    def get_string_form_for_print(self):
        return self.get_string_form()
        # if self.args.use_length_constraints:
        #     lc_string = ''
        #     for x in self.args.VARIABLES:
        #         lc_string += str(self.coefficients_variables_lc[x])
        #     weight_string = ''
        #     for x in self.args.ALPHABET:
        #         weight_string += str(self.weights_constants_lc[x])
        #     return self.w + ':' + str(self.ell) + ':' + lc_string + ':' + weight_string
        # elif not self.args.large:
        #     return self.w
        # else:
        #     translation = self.w
        #     translation.translate(str.maketrans(self.dictionary_translation))
        #     return translation

    def deepcopy(self, copy_id = False):
        # todo: find better way
        new_eq = WordEquation(self.args, w = self.w)
        if self.args.use_length_constraints:
            new_eq.ell = self.ell
            new_eq.coefficients_variables_lc = deepcopy(self.coefficients_variables_lc)
            new_eq.weights_constants_lc = deepcopy(self.weights_constants_lc)
            new_eq.num_letters_per_variable = deepcopy(self.num_letters_per_variable)

        if self.args.check_LP:
            new_eq.lp['abelian_form'] = deepcopy(self.lp['abelian_form'])
            new_eq.lp['AA'] = deepcopy(self.lp['AA'])
            new_eq.lp['bb'] = deepcopy(self.lp['bb'])
            new_eq.lp['GG'] = deepcopy(self.lp['GG'])
            new_eq.lp['hh'] = deepcopy(self.lp['hh'])
        new_eq.sat = self.sat

        new_eq.used_alphabet_only_equation = deepcopy(self.used_alphabet_only_equation)
        new_eq.used_alphabet = deepcopy(self.used_alphabet)
        new_eq.used_variables_only_equation = deepcopy(self.used_variables_only_equation)
        new_eq.used_variables = deepcopy(self.used_variables)

        new_eq.candidate_sol = deepcopy(self.candidate_sol)
        new_eq.not_normalized = deepcopy(self.not_normalized)
        new_eq.level = self.level
        if copy_id:
            new_eq.id = self.id
            new_eq.s_eq = deepcopy(self.s_eq)
        return new_eq

    def simplify_candidate_sol(self):
        # print(self.candidate_sol)
        sol = []
        for x in self.args.VARIABLES:
            bit = self.candidate_sol[x+'l'] + self.candidate_sol[x+'r']
            # print(bit)
            if len(bit) > 0:
                sol.append(x + ':' + bit)
        # print(sol)
        return sol

    def nullify_lc(self):
        self.coefficients_variables_lc = {x: 0 for x in self.args.VARIABLES}
        self.ell = 0

    def valid_lengths(self):
        return all([len(x) <= self.args.SIDE_MAX_LEN for x in self.w.split('=')])
#
from collections import OrderedDict

class Args():
    def __init__(self):
        self.num_discs=12
        self.nnet_type = 'newresnet'
        self.few_channels = False
        self.ALPHABET = ['a','b','c']
        self.VARIABLES = ['X','Y','Z','W','U','V']
        self.VARS = self.VARIABLES
        self.automatic_compress = True
        self.values_01 = False
        self.check_LP = True
        self.use_length_constraints = False
        self.use_normal_forms = True
        self.generation_mode = 'standard'
        self.quadratic_mode = True
        self.SIDE_MAX_LEN = 10
        self.bounded_model = False

args = Args()

Moves = WordEquationMovesQuadratic(args)


def multinomial(n, k, num):
    return math.factorial(n) / (math.factorial(k) ** num)


def moves_wordgame():
    mo = [(a + b, b + a) for a in alphabet for b in alphabet]
    return mo


def moves_hanoi():
    pass


def generate_graph(word, nx_graph=False):
    # moves = [(a + b, b + a) for a in alphabet for b in alphabet]
    # moves = [(a + b, b + a) for i,a in enumerate(alphabet)
    #         for j, b in enumerate(alphabet) if i > j]
    moves = range(Moves.get_action_size())

    states = set({word.w})
    edges = set({})
    words_to_check = set({word})
    if nx_graph:
        G = nx.Graph()
        G.add_node(word)
    else:
        G = None

    def process(word):
        # words_to_check.remove(word)
        for move in moves:
            new_word = Moves.act(word, move)

            if new_word.w != word.w:
                edges.update({(word.w, move, new_word.w)})
                if nx_graph:
                    G.add_edge(word.w, new_word.w)
                    G.add_edge(new_word.w, word.w)

            if new_word.w not in states:
                print(new_word.w)
                states.update({new_word.w})
                if nx_graph:
                    G.add_node(new_word.w)
                words_to_check.update({new_word})
                # generate_states(new_word, states)

    while len(words_to_check) > 0:
        process(words_to_check.pop())
        if len(states) % 1000 == 0:
            print(f'{len(states)} states created')
    return states, edges, G


init_word = 'aabbccdd'  # 2500, 64%
init_word = 'aaaaabbbbbccccc'  # 620.000, 63%
init_word = 'aaaabbbbcccc'  # 30000, 61%
init_word = 'aaaaaaaabbbbbbbbc'  # 176.000, 48%

init_word = 'aaaaaaabbbbbbbcc'  # 366.000, 67%
init_word = 'aaaaaabbbbbbbcc'  # 383.000, 57%
init_word = 'abcdefghi'  # 362.000, 75%

init_words = [
    'aaaaaabbbbbbc',  # 10.000, 47%

]

for init_word in init_words:
    print(init_word)
    alphabet = list(set(init_word))

    states, edges, G = generate_graph(WordEquation(args=args, w= 'aZYXWbaaabaVUaa=UYaaabaZbaVWaaX'),
                                      nx_graph=True)
    # print(states)
    # print(edges)
    print('number of states: ', len(states))

    print('number of edges: ', len(edges) / 2)
    print(f'number of clique edges: {0.5 * len(states) * (len(states) - 1)}')
    print(f'cyclomatic number: {(len(edges) - len(states) + 1) / ( len(edges))}')
    # nx.draw(G, with_labels=True)
    # plt.show()
    print('\n')