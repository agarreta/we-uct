# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:14:19 2019

@author: garre
"""

import numpy as np
from copy import deepcopy
import random
from string import ascii_lowercase
import os
import torch


def seed_everything(seed):
    # https://www.kaggle.com/hmendonca/fold1h4r3-arcenetb4-2-256px-rcic-lb-0-9759 cells 45-50
    #print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


VARS =list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿĀāĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ')
FVARS = list('ĀāĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ')

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
        self.coefficients_variables_lc = [dict({x: 0 for x in self.args.VARIABLES})]
        self.weights_constants_lc = [dict({x: 1 for x in self.args.ALPHABET})]
        self.lp = {}
        self.sat = 0 if not args.values_01 else 'unknown'
        self.ctx = None
        # The next are only used in equation generator
        self.num_lcs=0
        self.ell = [0 for _ in range(self.num_lcs)]
        self.num_letters_per_variable = [{x: {y: 0 for y in self.args.ALPHABET} for x in self.args.VARIABLES} for _ in range(self.num_lcs)]
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
            self.w += ':'
            lc_string = ''
            for i,y in enumerate(self.coefficients_variables_lc):
                for x in self.args.VARIABLES:
                    if y[x] > 0:
                        lc_string += str(y[x]) + x
                if len(self.ell) > 0:
                    self.w +=  lc_string +'>' +str(self.ell[i]) + ':'
            return self.w
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

    def nullify_lc(self, x,i):
        self.coefficients_variables_lc[i][x] = {x: 0 for x in self.args.VARIABLES}
        self.ell[i] = 0

    def valid_lengths(self):
        return all([len(x) <= self.args.SIDE_MAX_LEN for x in self.w.split('=')])
#
# from we import *
#
# import numpy as np
# from collections import deque
# from string import ascii_lowercase
# import torch
# from keras.preprocessing.text import Tokenizer
# import sys
# from dotmap import DotMap
# import multiprocessing
# import logging as lg
# import time
# from torch.nn import Linear
#
# from we.word_equation_moves import WordEquationMoves
#
# sys.setrecursionlimit(3000)
# sys.getrecursionlimit()
#
# MAXLEN_ALPHABET = 10
# VARIABLES = ['X', 'Y', 'Z', 'W', 'U']
#
# args = DotMap({
#     'train_device': 'cuda:0',
#     'play_device': 'cpu',
#
#     'VARIABLES': VARIABLES,
#     'MAXLEN_ALPHABET': MAXLEN_ALPHABET,
#     'MAX_ALPHABET': [x for x in ascii_lowercase[:MAXLEN_ALPHABET]],
#     'SPECIAL_CHARS': ['=', '.'],
#     'LEN_CORPUS': MAXLEN_ALPHABET + len(VARIABLES),
#     'SIDE_MAX_LEN': 30,
#     'side_maxlen_pool': 25,
#
#     'numIters': 10000,
#     'prevNumIters': 0,
#     'max_level': 1,
#     'num_mcts_sims_not_test': int(round(10 * 3)),  # number MCTS simulations
#     'test_num_mcts_sims': 200,
#     'batch_size': 128,
#     'numEps_train': 11,  # after * num_cpu's = how many equations to face (episodes) before training nn
#     'numEps_test': 5,
#     'epochs': 10,
#     'num_cpus': multiprocessing.cpu_count() - 3,
#
#     'maxlenOfQueue': 200000,  # max number of training examples
#     'cpuct': 1,  # exploration factor in MCTS
#     'temp': 1,
#
#     'load_model': False,
#     'save_model': True,
#     'train_model': True,
#     'checkpoint': './temp/',
#     'load_folder_file': ('temp', 'checkpoint_inf.pth.tar'),
#     'numItersForTrainExamplesHistory': 20,
#
#     'dropout': 0.2,
#     'num_channels': 256,
#     'num_resnet_blocks': 5,
#
#     'num_attempts_generated_eqs': 50000,
#
#     'timeout_time': 200,
#     'score_to_level_up': 0.95,
#
#     'pool_max_initial_length': 5,
#     'pool_max_initial_constants': 7,
#     'allowance_factor': 1.25,
#     'test_max_steps': 25,
#     'evaluation_after_exceeded_steps_or_time': 0.,
#     'learning_rate': 0.02,
#
#     'num_workers': 24,
#
#     'nnet_type': 'resnet',
#     'length_constraints': True,
#     # 'resnet', 'dnc', 'sam'
# })
# args.numEps_train = int(round(100 / args.num_cpus))
# args.numEps_test = int(round(20 / args.num_cpus))
#
# if args.nnet_type == 'resnet':
#     args.LEN_CORPUS += 1
#     tokenizer = Tokenizer(num_words=args.LEN_CORPUS, char_level=True, lower=False)
#     tokenizer.fit_on_texts(3 * ['.'] + 2 * args.VARIABLES + list(args.MAX_ALPHABET))
#
# elif args.nnet_type == 'dnc' or args.nnet_type == 'sam':
#     args.LEN_CORPUS += 2
#     tokenizer = Tokenizer(num_words=args.LEN_CORPUS, char_level=True, lower=False)
#     tokenizer.fit_on_texts(3 * ['=', '.'] + 2 * args.VARIABLES + list(args.MAX_ALPHABET))
#
# symbol_indices = tokenizer.word_index
# for x in symbol_indices.keys():
#     symbol_indices[x] -= 1
# symbol_indices_reverse = {i: x for i, x in enumerate(symbol_indices.keys())}
# args.symbol_indices = symbol_indices
# args.symbol_indices_reverse = symbol_indices_reverse
#
# args.num_mcts_simulations = int(round(50 * np.log(1 + args.max_level)))
#
# a = WordEquation(args,'aXX=bYc')
# a.coefficients_variables_lc['U'] = 5
# a.num_letters_per_variable['W']['c'] = 3
# a.update_used_symbols()
# print(a.used_variables, a.used_alphabet)
# print(a.lp)
# #
# # b = WordEquationEnvironment(args,)
# # aa = b.normal_form(a)
# # aa.show()