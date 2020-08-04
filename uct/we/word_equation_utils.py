from cvxopt import matrix, solvers
import cvxopt
import torch
# torch.set_default_tensor_type(torch.HalfTensor)
#import torch_geometric as tg
#from torch_geometric.data import Data
import torch
import numpy as np
import multiprocessing as mp
import networkx as nx
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
        self.transformations = WordEquationTransformations(args)
        self.create_action_dict()
        self.create_fast_action_dict()
        if seed is not None:
            seed_everything(seed)
            # self.get_afterstates()

    def are_equal(self, eq1, eq2):
        return eq1.get_string_form() == eq2.get_string_form()

    def one_hot_encode_resnet(self, s, length, first = False):
        t = torch.zeros(1,self.args.LEN_CORPUS,1, length)
        try:
            for i, x in enumerate(s):
                if x != '=':
                    t[0, self.args.symbol_indices[x],0, i] = 1.
                else:
                    t[0,len(self.args.symbol_indices),0, i] = 1.
        except:
            print('ERROR: ', t.shape, s, len(s), len(self.args.symbol_indices), x, i)
        return t

    def encode_newresnet(self, s1, s2, length, height =None):
        if not self.args.recurrent_blocks:
            height = 3*max(len(self.args.variable_indices), len(self.args.alphabet_indices))+1
        t = torch.zeros(1, 2, height, length)
        for i, x in enumerate(s1):
            if x in self.args.VARIABLES:
                t[0, 0, 3*self.args.variable_indices[x]+1, i] = 1.
            elif x != '.':
                t[0, 1, 3*self.args.alphabet_indices[x]+1, i] = 1.
        for i, x in enumerate(s2):
            if x in self.args.VARIABLES:
                t[0, 0, 3*self.args.variable_indices[x] +2, i] = 1.
            elif x != '.':
                t[0, 1, 3*self.args.alphabet_indices[x] +2, i] = 1.

        t[0, 0, [i for i in range(height)[::3]], :] = -1.
        t[0, 1, [i for i in range(height)[::3]], :] = -1.
        return t


    def format_state(self, eq, device='cpu'):
        if self.args.oracle:
            return torch.tensor([[[0.]]])
        if self.args.nnet_type in ['resnet', 'resnet_double', 'newresnet']:# and \
                #('noagg' in self.args.folder_name or 'meanagg' in self.args.folder_name):


            if self.args.using_attention:
                return self.attention_fun(eq.w)


            s1, s2 = eq.w.split('=')
            if self.args.format_mode == 'cuts':
                #l = int(self.args.NNET_SIDE_MAX_LEN/2)
                #s1 = s1[:l] + '|' + s1[-l:]
                #s2 = s2[: l] + '|'  + s2[-l:]

                cuts = []
                relevant_vars = [x for x in [s1[0], s1[-1], s2[0], s2[-1]] if x in self.args.VARIABLES]
                cuts += [[s1[:4], s2[:4]]]
                side_relevant_vars = [x for x in relevant_vars if x in [s1[0],s1[-1]] and x in [s2[0], s2[-1]]]
                inner_relevant_vars = [x for x in relevant_vars if x not in side_relevant_vars]
                for x in inner_relevant_vars:
                    p1 = s1.find(x,1,-1 )
                    if p1 == -1:
                        p1 = s2.find(x, 1,-1)
                    cuts += [[s1[p1-1:p1+2], s2[p1-1:p1+2]]]
                cuts +=  [[s1[-4:], s2[-4:]]]
                new_s1, new_s2 = '',''
                for c in cuts:
                    new_s1 += c[0] + '|'
                    new_s2 += c[1] + '|'
                s1 = new_s1[:-1]
                s2 = new_s2[:-1]
                if random.random() < 1/2000:
                    print(s1,s2)


            maxlen = self.args.SIDE_MAX_LEN if self.args.bound else max(len(s1), len(s2))
            maxlen = maxlen if self.args.format_mode != 'cuts' else self.args.NNET_SIDE_MAX_LEN+5
            if True:#self.args.bound:
                tensor1, tensor2 = self.one_hot_encode_resnet(s1, maxlen).to(device), self.one_hot_encode_resnet(s2, maxlen).to(device)
                tensor = torch.cat((tensor1, tensor2), dim=2).to(device)

            else:
                tensor = self.encode_newresnet(s1,s2, maxlen).to(device)


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
            if self.args.unsat_value in sats:
                eq.sat = self.args.unsat_value
            elif self.args.unknown_value in sats:
                eq.sat = self.args.unknown_value
            else:
                eq.sat = self.args.sat_value
            return eq
        else:
            return self.main_check_satisfiability(eq, smt_time=smt_time)

    def main_check_satisfiability(self, eq, time=500, num= 0, smt_time=None):

        if self.args.use_length_constraints:
            if not self.LC_is_sat(eq):
                eq.sat = self.args.unsat_value
                return eq

        w_split = eq.w.split('=')
        if w_split[0] == w_split[1]:
            if not self.args.use_length_constraints:
                eq.sat = self.args.sat_value
                return eq
            else:
                if self.LC_is_sat(eq):
                    # if True:
                    eq.sat = self.args.sat_value
                    return eq
                else:
                    eq.sat = self.args.unsat_value
                    return eq
        else:
            # the following if can be done with "'X' not in eq.w" assuming eq.w is in normal form, which should be the case
            if self.is_constant(eq.w):
                eq.sat = self.args.unsat_value
                return eq
            else:
                if w_split[0] in self.args.VARIABLES and self.is_constant(w_split[1]):
                    if self.treat_case_variable_side(eq, w_split[0], w_split[1]):
                        eq.sat = self.args.sat_value
                        return eq
                    else:
                        eq.sat = self.args.unsat_value
                        return eq
                if w_split[1] in self.args.VARIABLES and self.is_constant(w_split[0]):
                    if self.treat_case_variable_side(eq, w_split[1], w_split[0]):
                        eq.sat = self.args.sat_value
                        return eq
                    else:
                        eq.sat = self.args.unsat_value
                        return eq

            # now we know equation is not constant and no side is a single variable
            eq = self.unsat_by_incompatible_extremes(eq)
            if eq.sat == self.args.unsat_value:
                return eq

            # if False:
            if self.args.check_LP:
                if not self.LP_is_sat(eq):
                    eq.sat = self.args.unsat_value
                    return eq

            if self.args.use_oracle:
                out = SMT_eval(self.args, eq, timeout=smt_time)
                if out > self.args.unknown_value:
                    eq.sat = self.args.sat_value
                    return eq
                elif out < self.args.unknown_value:
                    eq.sat = self.args.unsat_value
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
                eq.sat = self.args.sat_value
                return True
            else:
                eq.sat = self.args.unsat_value
                return False
        else:
            eq.sat = self.args.sat_value
            return True

    def count_ctts(self, word):
        return len([1 for x in word if x in self.args.ALPHABET])

    def unsat_by_incompatible_extremes(self, eq):
        w_l, w_r = eq.w.split('=')
        if w_l[0] != w_r[0]:
            if w_l[0] not in self.args.VARIABLES and w_r[0] not in self.args.VARIABLES:
                eq.sat = self.args.unsat_value
                return eq
        if w_l[-1] != w_r[-1]:
            if w_l[-1] not in self.args.VARIABLES and w_r[-1] not in self.args.VARIABLES:
                eq.sat = self.args.unsat_value
                return eq
        return eq

    def is_constant(self, word):
        return all([x in self.args.ALPHABET + ['.'] for x in word])



def seed_everything(seed):
    # https://www.kaggle.com/hmendonca/fold1h4r3-arcenetb4-2-256px-rcic-lb-0-9759 cells 45-50
    #print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
