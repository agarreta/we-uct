# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:42:10 2019

@author: garre
"""
import random
import numpy as np
import re
from .word_equation import WordEquation
from .word_equation_transformations import WordEquationTransformations
from .word_equation_utils import WordEquationUtils, seed_everything
from string import ascii_lowercase
from copy import deepcopy
import time
import logging
from pickle import Pickler
import os
#from .SATtoWordEquation import SATtoWordEquation


def with_update_used_symbols(fun):

    def wrapped_fun(*args, **kwargs):
        args[1].update_used_symbols()
        values = fun(*args, **kwargs)
        if type(values) == tuple:
            values[0].update_used_symbols()
        else:
            values.update_used_symbols()
        return values

    return wrapped_fun

class WordEquationGenerator(object):

    def __init__(self, args, seed=None):
        if seed is not None:
            seed_everything(seed)

        self.args = args
        # self.args.init_log()
        self.utils = WordEquationUtils(args)
        self.transformations = WordEquationTransformations(args)
        self.pool = {}
        self.mode  = self.args.generation_mode  # standard, constant_side, quadratic, regular_orderered


    def random_word(self, length=3, alphabet=['a', 'b']):
        word = ''
        while True:
            for _ in range(length):
                word += random.choice(alphabet)
            if len(set(word)) > 1 or random.random()> -1.1:  # avoids having words with one single letter (which eventually produces a lot of equations beggining with ZZ
                #    print(f'-{word}')
                return word

    def init_eq(self, length, alph):
        eq = WordEquation(self.args)
        w = self.random_word(length, alph)
        eq.w = w + '=' + w
        eq.update_used_symbols()
        if self.args.check_LP:
            eq.find_LP()
        return eq

    @with_update_used_symbols
    def add_variable(self, eq, prob_insertion=0.4, letter = None, side = None, first_time = False):
        # if len(eq.used_alphabet_only_equation) > 0:
        """
        The first_time argument serves to make a special move if it is the first time a variable is added. In this case
        we transform something like aba=aba into Zaba=abZa. It is not allowed that Zaba=ZabZa for example. The point is
        to avoid deleting common prefixes and suffixes afterwards.
        :param eq:
        :param prob_insertion:
        :param letter:
        :param side:
        :param first_time:
        :return:
        """

        def insert_variable_regular_oriented(word):
            occurrences_of_letter = [i for i in range(len(word)) if word[i] == letter]
            if random.random() < prob_insertion:
                position = random.sample(occurrences_of_letter, 1)[0]
                word = word[:position] + new_string + word[position + 1:]
            return word

        variable = random.sample(eq.available_vars, 1)[0]
        letter = random.sample(eq.used_alphabet, 1)[0] if letter is None else letter
        side = random.choice(['left', 'right']) if side is None else side
        w = eq.w
        if side == 'right':
            new_string = letter + variable
        else:
            new_string = variable + letter

        if self.mode == 'standard':
            new_eq = ''
            for i, l in enumerate(w):
                if l == letter and random.random() < prob_insertion:
                    new_eq += new_string
                else:
                    new_eq += l
            eq.w = new_eq

        elif self.mode == 'constant_side':
            new_eq = ''
            on_left_side = True
            for i, l in enumerate(w):
                if l == '=':
                    on_left_side = False
                if l == letter and random.random() < prob_insertion:
                    if (self.mode != 'constant_side') or on_left_side:
                        new_eq += new_string
                    else:
                        new_eq += l
                else:
                    new_eq += l
            eq.w = new_eq
        elif self.mode == 'quadratic-oriented':
            w0, w1 = w.split('=')
            possible_letters = set([x for x in w0 if x in self.args.ALPHABET]).intersection(
                set([x for x in w1 if x in self.args.ALPHABET]))
            if len(possible_letters) > 0:
                letter = random.sample(possible_letters, 1)[0]
                if side == 'right':
                    new_string = letter + variable
                else:
                    new_string = variable + letter
                w0 = insert_variable_regular_oriented(w0)
                w1 = insert_variable_regular_oriented(w1)
                new_eq = w0 + '=' + w1
                eq.w = new_eq
            else:
                pass
                # eq.attempted_wrong_move = True

        elif self.mode == 'regular-ordered':
            def get_maximal_constant_chunk(word):
                if len([x for x in word if x in self.args.VARIABLES]):
                    index = int(max([i for i in range(len(word)) if word[i] in self.args.VARIABLES]) +1)
                else:
                    index = 0
                return word[:index], word[index:]

            w0, w1 = w.split('=')
            w0_chunk0, w0_chunk = get_maximal_constant_chunk(w0)
            w1_chunk0, w1_chunk = get_maximal_constant_chunk(w1)
            if len(w0_chunk) == 0 and len(w1_chunk) == 0:
                # eq.attempted_wrong_move = True
                new_eq = ''
                pass
            else:
                possible_letters = set([x for x in w0_chunk if x in self.args.ALPHABET]).intersection(
                    set([x for x in w1_chunk if x in self.args.ALPHABET]))
                if len(possible_letters) > 0:
                    letter = random.sample(possible_letters, 1)[0]
                    if side == 'right':
                        new_string = letter + variable
                    else:
                        new_string = variable + letter
                    w0_chunk = insert_variable_regular_oriented(w0_chunk)
                    w1_chunk = insert_variable_regular_oriented(w1_chunk)
                    new_eq = w0_chunk0+ w0_chunk + '=' + w1_chunk0+w1_chunk
                    eq.w = new_eq
                else:
                    pass
            #print(w, new_eq)
        return eq

    def get_paired_string_and_variables(self, eq):
        """
        :param eq:
        :return: a random tuple (letter, var, side) such that each occurrence of var in eq.w has letter on its left or
        right, depending on side. If no such tuple exists it returns (False, False, False)
        """
        permuted_variables = np.random.permutation( eq.used_variables )
        permuted_used_alphabet = np.random.permutation( eq.used_alphabet )
        permuted_sides = np.random.permutation( ['left', 'right'] )
        for variable in permuted_variables:
            for letter in permuted_used_alphabet:
                for side in permuted_sides:
                    if self.var_always_starts_or_ends_with_letter( eq.w, variable, letter, side ):
                        return letter, variable, side
        return False, False, False

    @with_update_used_symbols
    def anti_compress(self, eq, initial_letter=None, mode = 'anti_pop_compatible'):

        if mode == 'anti_pop_compatible':
            if len(eq.used_variables)==0:
                return eq, False

            letter1, variable, side = self.get_paired_string_and_variables(eq)
            valid = False if letter1 == False else True

            if not valid:
                possible_initial_letters = eq.used_alphabet
                if len(possible_initial_letters) == 0:
                    return eq, False
                letter1 = random.sample(possible_initial_letters,1)[0]

        else:
            possible_initial_letters = eq.used_alphabet
            if len(possible_initial_letters) == 0:
                return eq, False
        # if initial_letter is None:
            letter1 = random.sample(possible_initial_letters, 1)[0]
        # else:
        #     letter1 = initial_letter

        possible_letters = list([x for x in self.args.ALPHABET if x != letter1])
        if len(possible_letters) == 0:
            return eq, False

        letter2 = np.random.choice(possible_letters)
        letter3 = np.random.choice(possible_letters)

        eq.w = re.sub(letter1, letter2 + letter3, eq.w)
        if self.args.use_length_constraints:
            for x in self.args.VARIABLES:
                num_original_letter = eq.num_letters_per_variable[x][letter1]
                eq.num_letters_per_variable[x][letter2] += num_original_letter
                eq.num_letters_per_variable[x][letter3] += num_original_letter
                eq.num_letters_per_variable[x][letter1] = 0
                eq.ell = eq.ell + eq.coefficients_variables_lc[x] * num_original_letter

        return eq, True

    def var_always_starts_or_ends_with_letter(self, w, variable,letter,side):
        """
        to check that all occurrences of 'variable' are preceded by 'letter' we count
        the number of occurences of 'variable' and of 'old_string'
        """
        if side == 'left':
            old_string = letter + variable
        else:
            old_string = variable + letter

        new_eq = re.sub(old_string, variable, w)
        return len(new_eq) == len(re.sub(variable, '', w))

    @with_update_used_symbols
    def anti_pop(self, eq):

        def main_anti_pop_function(variable, letter, side):
            w = eq.w
            if side == 'left':
                old_string = letter + variable
            else:
                old_string = variable + letter

            new_eq = re.sub(old_string, variable, w)
            if self.var_always_starts_or_ends_with_letter(eq.w,variable, letter, side):
                valid = True
                eq.w = new_eq
                if self.args.use_length_constraints:
                    eq.num_letters_per_variable[variable][letter] += 1
                    eq.ell = eq.ell + eq.coefficients_variables_lc[variable]
            else:
                valid = False

            return eq, valid

        letter, variable, side = self.get_paired_string_and_variables(eq)
        if letter == False:
            return eq, False
        else:
            eq, valid = main_anti_pop_function( variable, letter, side )
            return eq, valid

    def compute_variable_length(self, eq, variable):
        return sum([eq.num_letters_per_variable[variable][x] for x in self.args.ALPHABET])

    @with_update_used_symbols
    def add_variable_to_lc(self, eq):

        def main_add_variable_to_lc_function( eq, variable, sign):
            eq.coefficients_variables_lc[variable] += sign
            eq.ell = eq.ell + sign * self.compute_variable_length(eq, variable)
            return eq

        variable = random.choice(eq.used_variables)
        num_pos = sum([int(eq.coefficients_variables_lc[x] > 0) for x in eq.used_variables])
        num_neg = len(eq.used_variables) - num_pos

        if num_neg == 0 and num_pos == 0:
            sign =-1
            # if len(eq.used_variables) >= 2:
            #     variable = random.choice([x for x in eq.used_variables if x != variable])
            #     eq = main_add_variable_to_lc_function( eq, variable, 1 )
        elif num_neg != 0 and num_pos == 0:
            sign = 1
        elif num_neg ==0 and num_pos != 0:
            sign = -1
        else:
            sign = np.random.choice([-1, 1])
            # sign = random.choice([-1, 1])
        eq = main_add_variable_to_lc_function( eq, variable, sign )

        return eq

    def decrease_ell(self, eq):
        eq.ell -= 1
        return eq


    def generate_sequence_of_eqns(self, level, initial_alphabet, initial_length=2, prob_insertion=0.4, min_num_vars=1):
        i=0
        self.log = []
        current_level = 0
        num_consecutive_equal_eqns = 0
        eq = self.init_eq(initial_length, initial_alphabet)
        eq.update_used_symbols()

        while eq.level < level:
            #print(eq.w)
            available_actions, distribution = [], []
            # or len(eq.used_variables_only_equation) < min_num_vars:
            i+=1
            if i == 1:
                available_actions = ['add_variable']
                distribution = [1]
                first_time = True

            else:
                first_time = False
                if len(eq.available_vars) > 0 and len(eq.used_alphabet) > 0:
                    available_actions += ['add_variable']
                    if self.mode == 'standard' or self.mode == 'constant_side':
                        distribution += [0.25]
                    elif self.mode == 'regular-ordered' or self.mode == 'quadratic-oriented':
                        distribution += [1]

                if len(eq.used_variables) > 0 and len(eq.used_alphabet) > 0:
                    available_actions += ['anti_pop']
                    distribution += [1]  # favor anti-pop, otherwise generated equations tend to be solvable just by deleting its variables

                if  len(eq.used_alphabet)>0 and len(eq.used_variables)>0:
                    # and len(eq.available_alphabet) > 0:
                    available_actions += ['anti_compress']
                    distribution += [1]

                if len(available_actions) == 0:
                    break

            distribution = np.array(distribution)
            step_type = np.random.choice(available_actions, 1, p=distribution / (distribution.sum()))[0]
            original_eq = eq.w

            if step_type == 'anti_pop':
                eq, valid = self.anti_pop(eq)
                if valid and eq.w != original_eq:
                    if self.args.use_length_constraints:
                        eq = self.add_variable_to_lc(eq)
                    eq.level += 1

            elif step_type == 'add_variable':
                prob_insertion = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                eq = self.add_variable(eq, prob_insertion, first_time=False)
                if eq.w != original_eq:
                    eq.level += 1

            elif step_type == 'anti_compress':
                eq, valid = self.anti_compress(eq)
                if valid and eq.w != original_eq:
                    if self.args.use_length_constraints:
                        eq = self.add_variable_to_lc(eq)
                    eq.level += 1

            if original_eq == eq.w:
                num_consecutive_equal_eqns += 1
            else:
                num_consecutive_equal_eqns = 0
            if num_consecutive_equal_eqns >= 3:
                eq = self.init_eq(initial_length, initial_alphabet)
                eq.update_used_symbols()
                break

            if i >= 2:
                # print(eq.w)
                eq = self.transformations.normal_form(eq, minimize=True, mode = 'generation')
                if eq.level >=4 and self.args.generate_z3_unsolvable:
                    eq = self.utils.check_satisfiability(eq, time=50)
                    if float(eq.sat) != 0.:
                        break

                eqsplit = eq.w.split('=')
                if self.args.bounded_model:
                    if len(eqsplit[0]) > self.args.side_maxlen_pool or len(eqsplit[1]) > self.args.side_maxlen_pool:
                        break
                if len(eqsplit[0]) <= 1 or len(eqsplit[1]) <= 1:
                    break
                # print('post normal',eq.w)
                neq = eq.deepcopy()

                # eq = self.transformations.normal_form(eq, mode = 'generation')
                self.log.append(neq)

                #self.utils.check_satisfiability(neq)
                #if neq.sat == -1:
                #    print(f'Error: unsatisfiable equation {neq.w} found in '
                #          f'generated pool. Pool: '
                #          f'{[x.get_string_form() for x in self.log]}' )
                #    raise Exception

    def is_simple(self, eq, vars_to_check):
        """Simple : if it can be solved by removing all variables or one side has length less than 3 """
        # TODO: can be optimized

        if eq.w.split('=')[0] == eq.w.split('=')[1]:
            return True

        for var in vars_to_check:
            if var in eq.w:
                neq = eq.deepcopy()
                neq.w = re.sub(var, '', neq.w)
                neq = self.transformations.normal_form(neq)
                neq_split = neq.w.split('=')
                if neq_split[0] == neq_split[1]:
                    if self.args.use_length_constraints:
                        if self.utils.LC_is_sat(eq):
                            return True
                    else:
                        return True
                if neq_split[0] in self.args.VARIABLES and self.is_constant(neq_split[1]):
                    if self.args.use_length_constraints:
                        if self.utils.treat_case_variable_side(neq.deepcopy(), neq_split[0], neq_split[1]):
                            return True
                    else:
                        return True
                if neq_split[1] in self.args.VARIABLES and self.is_constant(neq_split[0]):
                    if self.args.use_length_constraints:
                        if self.utils.treat_case_variable_side(neq.deepcopy(),neq_split[1], neq_split[0]):
                            return True
                    else:
                        return True
                if self.is_simple(neq, vars_to_check):
                    return True
                #  TODO: this else False makes this function much faster but it may miss simple forms of the type X=ctt
                # else:
                #     return False

        #eq_split = eq.w.split('=')
        #if eq_split[0].count('Z') == len(eq_split[0]) or eq_split[1].count('Z') == len(eq_split[1]):
        #    if random.random() < -0.95:
        #        return True
        return False


        # for var in self.args.VARIABLES:
        #     if var in neq.w:
        #         neq.w = re.sub(var, '', w)
        #         neq = self.transformations.normal_form(neq)
        #         if self.is_final_state(neq) == 1:
        #             return True
        #         if self.is_simple(neq):
        #             return True
        # else:
        #     return False

    def is_constant(self, w):
        return not bool(np.array([int(x in w) for x in self.args.VARIABLES]).sum())

    def generate_pool(self, size, level_list):
        p = []
        t = time.time()
        if self.args.test_mode:
            print(level_list)
        for i, level in enumerate(level_list):
        #while len(p) < size:
            # min_num_vars = random.choice([1])
            if self.args.generation_mode == 'standard':
                prob_insertion = random.choice([0.2,0.3,0.4,0.5,0.6,0.7, 0.8,0.9])
                initial_length = self.args.pool_max_initial_length  # random.choice([x+1 for x in range(self.args.pool_max_initial_length-1, self.args.pool_max_initial_length)])
            elif self.args.generation_mode == 'alternative':
                prob_insertion = random.choice([0.1,0.2,0.8,0.9])
                initial_length = self.args.pool_max_initial_length
            elif self.args.generation_mode == 'constant_side':
                prob_insertion = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                initial_length = 3*self.args.pool_max_initial_length
            elif self.args.generation_mode == 'quadratic-oriented':
                prob_insertion = random.choice([0.8])
                initial_length = 2 #max(1,int(self.args.pool_max_initial_length/4))
            elif self.args.generation_mode == 'regular-ordered':
                prob_insertion = random.choice([0.66])
                initial_length = int(self.args.pool_max_initial_length/2)

            #print(self.args.generation_mode, self.args.test_mode)
            if self.args.generation_mode == 'standard' and not self.args.test_mode:
                #random_level = level #level_list[-1]  # random.choice([x for x in level_list])
                if i == 0:
                    random_level = max(level-1, self.args.min_level)
                elif i == 1:
                    random_level = level
                else:
                    random_level = random.choice([x for x in range(3, level+1)])
            else:
                random_level = level
            find_next_equation = False
            while not find_next_equation:
                self.generate_sequence_of_eqns(level=random_level,
                                               initial_alphabet=[x for x in self.args.ALPHABET[:self.args.pool_max_initial_constants]],
                                               initial_length=initial_length,
                                               prob_insertion=prob_insertion,
                                               min_num_vars=1)

                if len(self.log)> 1:
                    candidate_eq = self.log[-1]
                    if candidate_eq.level >= random_level:
                        candidate_eq = self.transformations.normal_form(candidate_eq, minimize=True)
                        if len(candidate_eq.w.split('=')[0]) >= 2 and len(candidate_eq.w.split('=')[1]) >= 2:
                            if not self.is_simple(candidate_eq, self.args.VARIABLES):
                                # if not self.z3solvable(candidate_eq):
                                if self.args.generate_z3_unsolvable:
                                    candidate_eq = self.utils.check_satisfiability(candidate_eq)
                                else:
                                    candidate_eq.sat = 0
                                if candidate_eq.sat == 0:
                                    p.append(candidate_eq)
                                    find_next_equation = True
                                    if self.args.test_mode:
                                        print(len(p), candidate_eq.get_string_form(), candidate_eq.level)

        logging.info(f'Elsapsed time generating pool: {round(time.time()-t,2)}')
        self.pool_generation_time = round(time.time()-t, 2)
        self.pool = p
        self.save_pool(level_list, size)

    def z3solvable(self, eq):
        if self.args.generate_z3_unsolvable:
            z3 = WeToSmtZ3(self.args.VARIABLES, self.args.ALPHABET, 30000)
            out = z3.eval(eq.w)
            z3.solver.reset()
            if float(out) != 0.:
                return True

            else:
                #print(eq.w)
                return False
        else:
            return False


    def save_pool(self, level_list, size):
        folder = self.args.folder_name + '/pools'
        if not os.path.exists(folder):
            os.makedirs(folder)
        pool_names = os.listdir(folder)

        if not self.args.test_mode:
            filename = os.path.join(folder, f'pool{len(pool_names)}_lvl_{level_list[0]}_{level_list[-1]}_size_{size}.pth.tar')
        else:
            filename = os.path.join('benchmarks', f'pool_lvl_{level_list[0]}_{level_list[-1]}_size_{size}_{self.args.generation_mode}_{self.args.size_type}.pth.tar')

        with open(filename, "wb+") as f:
            Pickler(f).dump(self.pool)
        f.close()