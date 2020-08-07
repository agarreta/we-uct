from cvxopt import matrix, solvers
import cvxopt
import torch
import numpy as np
import random
from .word_equation_transformations import WordEquationTransformations
from .word_equation import WordEquation
from uct.SMTSolver import SMT_eval
import os

class WordEquationUtils(object):
    def __init__(self, args, seed=0):
        self.args = args
        self.transformations = WordEquationTransformations(args)
        if seed is not None:
            seed_everything(seed)

    def are_equal(self, eq1, eq2):
        return eq1.get_string_form() == eq2.get_string_form()

    def one_hot_encode_resnet(self, s, length):
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

    def format_state(self, eq, device='cpu'):
        if self.args.oracle:
            return torch.tensor([[[0.]]])
        s1, s2 = eq.w.split('=')
        maxlen = self.args.SIDE_MAX_LEN if self.args.bound else max(len(s1), len(s2))
        maxlen = maxlen if self.args.format_mode != 'cuts' else self.args.NNET_SIDE_MAX_LEN+5
        tensor1, tensor2 = self.one_hot_encode_resnet(s1, maxlen).to(device), self.one_hot_encode_resnet(s2, maxlen).to(device)
        tensor = torch.cat((tensor1, tensor2), dim=2).to(device)

        return tensor


    def get_eq_list(self, eq):
        eq_split = eq.w.split('=')
        lefts_w = eq_split[0].split('+')
        rights_w = eq_split[1].split('+')
        assert len(lefts_w) == len(rights_w)
        w_list = [lefts_w[i]+'='+rights_w[i] for i in range(len(lefts_w))]
        eq_list = [WordEquation(args=self.args, w=w) for w in w_list]
        return eq_list

    def check_satisfiability(self, eq, time=500, smt_time=None):
        return self.main_check_satisfiability(eq, smt_time=smt_time)

    def main_check_satisfiability(self, eq, smt_time=None):

        w_split = eq.w.split('=')
        if w_split[0] == w_split[1]:
            eq.sat = self.args.sat_value
            return eq
        else:
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

    def treat_case_variable_side(self, eq, var_side):
        """

        :param eq:
        :param var_side:
        :param other_side:
        :return: True or False depending on wheter a WEWLC with word equation of the form X=ct has a solution or not
        """
        assert len(var_side) == 1
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
