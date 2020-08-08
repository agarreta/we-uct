# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:15:59 2019

@author: garre
"""
import re
import torch
from copy import deepcopy
from we.word_equation.word_equation_utils import WordEquationUtils, seed_everything
from we.word_equation.word_equation_transformations import WordEquationTransformations
# torch.set_default_tensor_type(torch.HalfTensor)

class WordEquationMoves(object):

    def __init__(self, args, seed = None):
        self.args = args
        self.utils = WordEquationUtils(args)
        self.transformations = WordEquationTransformations(args)
        self.create_action_dict()
        self.create_fast_action_dict()
        if seed is not None:
            seed_everything(seed)


    def delete_var(self, eq, variable):
        if variable in eq.w:
            eq.w = re.sub(variable, '', eq.w)
            eq.coefficients_variables_lc[variable] = 0
            eq.attempted_wrong_move = False
            # new_eq = self.treat_lone_variables_in_lc(new_eq, variable)
            return eq
        eq.attempted_wrong_move = True
        return eq


    def pop(self, eq, variable, letter, side, verbose):
        if letter not in eq.w or variable not in eq.w:
            eq.attempted_wrong_move = True
            return eq
        if side == 'left':
            new_string = letter + variable
        else:
            new_string = variable + letter
        new_w = re.sub(variable, new_string, eq.w)
        w_split = new_w.split('=')

        def aux_function(eq,new_w):
            eq.w = new_w
            if self.args.use_length_constraints:
                eq.ell = eq.ell - (eq.coefficients_variables_lc[variable] * eq.weights_constants_lc[letter])
            # eq.lp[letter] = eq.lp[letter] + w_split[0].count(variable) - w_split[1].count(variable)
            eq.attempted_wrong_move = False
            if verbose == 1:
                if side == 'left':
                    eq.candidate_sol[variable + 'l'] = eq.candidate_sol[variable + 'l'] + letter
                else:
                    eq.candidate_sol[variable + 'r'] = letter + eq.candidate_sol[variable + 'r']
                # print('pop', eq.candidate_sol)
            return eq
        if len(w_split[0]) <= self.args.SIDE_MAX_LEN and len(w_split[1]) <= self.args.SIDE_MAX_LEN:
            eq =aux_function(eq,new_w)
            return eq
        if not self.args.bounded_model:
            eq = aux_function(eq,new_w)
            return eq
        eq.attempted_wrong_move = True
        return eq

    def pair_compression(self, eq, letter1, letter2, verbose):
        pair = letter1 + letter2
        new_letter = self.args.ALPHABET[-1]
        if new_letter not in eq.w:
            if pair in eq.w:
                eq.w = re.sub(pair, new_letter, eq.w)
                if self.args.use_length_constraints:
                    eq.weights_constants_lc[new_letter] = eq.weights_constants_lc[letter1] + eq.weights_constants_lc[
                        letter2]
                eq.attempted_wrong_move = False
                if verbose == 1:
                    eq.candidate_sol = {key: re.sub(pair, new_letter, value) for key, value in eq.candidate_sol.items()}
                return eq
        eq.attempted_wrong_move = True
        return eq



    def act(self, eq, action_num, verbose = 0):
        eq.attempted_wrong_move = False
        new_eq = eq.deepcopy()
        action = self.actions[action_num]
        type_of_action = action['type']

        if type_of_action == 'delete':
            variable = action['variable']
            new_eq = self.delete_var(new_eq, variable)

        if type_of_action == 'compress':
            letter1, letter2 = action['letter1'], action['letter2']
            new_eq = self.pair_compression(new_eq, letter1, letter2, verbose)

        if type_of_action == 'pop':
            variable = action['variable']
            letter = action['letter']
            side = action['side']
            new_eq = self.pop(new_eq, variable, letter, side, verbose)
            # if not new_eq.valid_lengths():
            #     print('pop of too much length', new_eq.attempted_wrong_move, self.args.SIDE_MAX_LEN)

        if new_eq.attempted_wrong_move:
            return new_eq

        # if verbose == 1:
        new_eq.not_normalized = new_eq.get_string_form()

        new_eq = self.transformations.normal_form(new_eq)
        # if not new_eq.valid_lengths():
        #     logging.error('normalization of too much length {}, {}'.format(new_eq.attempted_wrong_move, self.args.SIDE_MAX_LEN))
        # if verbose:
        #     print('after norm', new_eq.candidate_sol)

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
            if type(x) == int: valid_actions[i] = 0.
        return valid_actions, afterstates

    def create_action_dict(self):
        """
        Creates a dictionary with possible actions. Each entry is a dictionary with different entries depending on the type of action
        """
        actions_delete = [{'description': "delete " + str(self.args.VARIABLES[i]),
                           'type': 'delete',
                           'variable': self.args.VARIABLES[i]
                           } for i in range(len(self.args.VARIABLES))]



        actions_compress, actions_pop = [], []
        if not self.args.automatic_compress:
            for i in range(len(self.args.ALPHABET[:-1])):
                for j in range(len(self.args.ALPHABET[:-1])):
                    actions_compress.append(
                        {'description': "compress " + str(self.args.ALPHABET[i]) + " and " + str(
                            self.args.ALPHABET[j]),
                         'type': 'compress',
                         'letter1': self.args.ALPHABET[i],
                         'letter2': self.args.ALPHABET[j]
                         })


        for i in range( len( self.args.ALPHABET) ):
            for k in range(len(self.args.VARIABLES)):
                for side in ['left', 'right']:
                    actions_pop.append(
                        {'description': "pop " + str(
                            self.args.ALPHABET[i]) + " from " + side + " of " + str(
                            self.args.VARIABLES[k]),
                         'type': 'pop',
                         'letter': self.args.ALPHABET[i],
                         'side': side,
                         'variable': self.args.VARIABLES[k]
                         })

        self.actions = actions_delete + actions_compress + actions_pop  # + actions_rename + actions_reset + actions_swap
        self.actions = {i: self.actions[i] for i in range(len(self.actions))}

    def get_action_size(self):
        self.create_action_dict()
        return len(self.actions)

    def create_fast_action_dict(self):
        'returns a dictionary that allows easy access to the number index each action has in self.action'
        alph = self.args.ALPHABET
        vars = self.args.VARIABLES
        fast_dict = {}
        fast_dict['delete'] = {var: i for i,var in enumerate(vars)}
        if not self.args.automatic_compress:
            fast_dict['compress'] = {l1: {l2 : len(vars) + len(alph[:-1])*i + j for j,l2 in enumerate(alph[:-1])} for i,l1 in enumerate(alph[:-1])}
        fast_dict['pop'] = {l: {s: {var: len(vars) + len(alph[:-1])*len(alph[:-1]) + i*len(vars)*2 + k*len(vars) + j for j,var in enumerate(vars) } for k,s in enumerate(['left', 'right'])} for i,l in enumerate(alph)}
        self.fast_dict = fast_dict




