import numpy as np
from copy import deepcopy
import random

class WordEquation(object):
    """
    CLASS ATRIBUTES
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

        self.args = args
        self.w = w
        self.lp = {}
        self.sat = 0
        self.attempted_wrong_move = False
        self.level = 0
        self.id = self.get_string_form() + str(round(random.random(), 5))[1:]

        self.num_lcs=0
        self.num_letters_per_variable = [{x: {y: 0 for y in self.args.ALPHABET} for x in self.args.VARIABLES} for _ in range(self.num_lcs)]
        self.used_alphabet_only_equation = []
        self.used_alphabet = []
        self.used_variables_only_equation = []
        self.used_variables = []
        self.initial_state  = False
        if self.args.check_LP:
            self.find_LP()


    def get_id(self):
        self.id = self.get_string_form() + str(round(random.random(), 5))[1:]

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

    def get_string_form(self):
        return self.w

    def get_string_form_for_print(self):
        return self.get_string_form()

    def deepcopy(self, copy_id = False):
        # TODO:  why not use the deepcopy function from the copy library?
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

    def valid_lengths(self):
        return all([len(x) <= self.args.SIDE_MAX_LEN for x in self.w.split('=')])
