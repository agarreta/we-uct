
import random
import numpy as np
import re
from .word_equation import WordEquation
from .word_equation_transformations import WordEquationTransformations
from .word_equation_utils import WordEquationUtils, seed_everything
import time
import logging
from pickle import Pickler
import os

def with_update_used_symbols(fun):
    """
    Function wrapper that updates the used sympbols in the equation that is currently being generated
    """
    def wrapped_fun(*args, **kwargs):
        args[1].update_used_symbols()
        values = fun(*args, **kwargs)
        if type(values) == tuple:
            values[0].update_used_symbols()
        else:
            values.update_used_symbols()
        return values
    return wrapped_fun

class WordEquationGeneratorQuadratic(object):
    def __init__(self, args, seed=None):
        if seed is not None:
            seed_everything(seed)
        self.args = args
        self.SIDE_MAX_LEN = self.args.SIDE_MAX_LEN
        self.VARIABLES  = self.args.VARIABLES
        self.ALPHABET = self.args.ALPHABET
        self.utils = WordEquationUtils(args, seed)
        self.transformations = WordEquationTransformations(args)
        self.pool = {}
        self.mode = self.args.generation_mode  # standard, constant_side, quadratic, regular_orderered

    def random_word(self, length=3, alphabet=['a', 'b']):
        word = ''
        while True:
            for _ in range(length):
                word += random.choice(alphabet)
            if len(set(word)) > 1:  # this together with the 'while True' avoids having initial words with only no two distinct letters. Such initial words tend to produce  quite easy equations
                return word

    def init_eq(self, length, alph):
        eq = WordEquation(self.args)
        if self.args.generation_mode == 'constant_side':
            length -= int(len(self.VARIABLES)/2)
        w = self.random_word(length, alph)
        eq.w = w + '=' + w
        eq.update_used_symbols()
        if self.args.check_LP:
            eq.find_LP()
        return eq

    def get_previous_letter(self, eq_w, letter):
        split = [x for x in eq_w.split(letter)][:-1]
        split = [x for x in split if len(x) > 0]
        previous = [x[-1] for x in split if x[-1] != '=']
        #print(previous)
        # print(eq_w, letter, split, previous)
        if len(previous) == 0:
            return ''
        assert len(np.unique(previous) )<= 1
        return previous[0]

    @with_update_used_symbols
    def inverse_move(self, eq):
        w1, w2 = eq.w.split('=')
        l1, l2 = w1[0], w2[0]
        choice = [[x,i] for i, x in enumerate([l1, l2]) if x in self.VARIABLES]
        if len(choice) == 0:
            return eq
        letter = random.choice(choice)
        #if letter[0] in self.VARIABLES:
        #    return eq
        if self.mode in ['quadratic-' \
                        'oriented', 'quadratic-oriented-linear','alternative'] :
            side = letter[1]
            letter = letter[0]
            main_eq_side = eq.w.split('=')[1-side]
            previous_letter = self.get_previous_letter(main_eq_side, letter)
            if previous_letter == '':#t or previous_letter in self.VARIABLES:
                return eq
            main_eq_side = previous_letter + re.sub(previous_letter+letter, letter, main_eq_side)
            if side == 0:
                new_w = w1 + '=' + main_eq_side
            else:
                new_w = main_eq_side + '=' + w2
            eq.w = new_w
            return eq
        elif self.mode == 'constant_side':
            side = 1
            letter = letter[0]
            main_eq_side = eq.w.split('=')[1-side]
            previous_letter = self.get_previous_letter(main_eq_side, letter)
            if previous_letter == '':# or previous_letter in self.VARIABLES:
                return eq
            main_eq_side =   re.sub(previous_letter+letter, letter, main_eq_side)
            new_w = main_eq_side + '='+ previous_letter + w2
            eq.w = new_w
            return eq
        else:
            assert False
        # new_w_split = new_w.split('=')
        # assert new_w_split[0][0] in self.VARIABLES or new_w_split[1][0] in self.VARIABLES
        # side_with_previous_letter = [x[0] == letter for x in new_w.split('=')]  # [bool, bool]
        # print(eq.w, new_w, side_with_previous_letter)
        # if side_with_previous_letter[0]:  # left starts with letter
        #     new_w = new_w_split[0] + '=' + previous_letter + new_w_split[1]
        # else:
        #     new_w = previous_letter + new_w_split[0] + '=' + new_w_split[1]
        # eq.w = new_w
        # return eq

    @with_update_used_symbols
    def add_variable(self, eq):
        side = random.choice([0, 1]) if self.mode != 'constant_side' else 0
        other_side = 1-side if self.mode!= 'constant_side' else side
        if len(eq.available_vars) > 0:
            var = random.choice(eq.available_vars)
        else:
            return eq
        eq_split = eq.w.split('=')
        eq_side = eq_split[other_side]
        position = random.choice(range(len(eq_side)))
        new_eq_side = eq_side[:position] + var + eq_side[position:]
        if self.mode in ['quadratic-oriented','alternative']:
            if side == 0:
                new_w = var + eq_split[side] + '=' + new_eq_side
            else:
                new_w = new_eq_side + '=' + var + eq_split[side]
        elif self.mode == 'constant_side':
            new_w = eq_split[side] + '=' + var + new_eq_side

        elif self.mode == 'quadratic-oriented-linear':
            lin = random.choice([1,1,1,0])
            if any([len(x)>= self.SIDE_MAX_LEN-2 for x in eq_split]) and side ==1:
                lin = max(1, lin)
            if lin == 2:
                let = new_eq_side[position-1]
                for i, x in enumerate(new_eq_side):
                    if i  != position+1 and x == let:
                        new_eq_side_ = new_eq_side[:i]  + var  +new_eq_side[i:]
                new_eq_side = new_eq_side_

            if side == 0:
                new_w = lin*var + eq_split[side] + '=' + new_eq_side

            else:
                new_w = new_eq_side + '=' + lin*var + eq_split[side]

        elif self.mode == 'constant_side':
            new_w = var + new_eq_side + '=' + eq_split[1]
        else:
            assert False

        eq.w = new_w
        return eq

    def compute_variable_length(self, eq, variable):
        return sum([eq.num_letters_per_variable[variable][x] for x in self.ALPHABET])

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
        while eq.level <  level:
            #print(eq.w,eq.level, level)
            available_actions, distribution = [], []
            # or len(eq.used_variables_only_equation) < min_num_vars:
            i+=1
            if i == 1:
                available_actions = ['add_variable']
                distribution = [1]
                first_time = True

            else:
                first_time = False

                if self.args.quadratic_mode:
                    available_actions = ['add_variable', 'inverse_move']
                    distribution = [0.4, 0.6]
                if len(available_actions) == 0:
                    #print('no more available actions')
                    break

            distribution = np.array(distribution)
            step_type = np.random.choice(available_actions, 1, p=distribution / (distribution.sum()))[0]
            original_eq = eq.w
            #print(eq.w, step_type)

            if step_type == 'inverse_move':
                eq = self.inverse_move(eq)
                if eq.w != original_eq:
                    if self.args.use_length_constraints:
                        eq = self.add_variable_to_lc(eq)
                    eq.level += 1

            elif step_type == 'add_variable':
                eq = self.add_variable(eq)
                if eq.w != original_eq:
                    eq.level += 1

            if original_eq == eq.w:
                num_consecutive_equal_eqns += 1
            else:
                num_consecutive_equal_eqns = 0
            if num_consecutive_equal_eqns >= 3:
                eq = self.init_eq(initial_length, initial_alphabet)
                eq.update_used_symbols()
                #print('3 consecutive repeated equtions' )
                break

            if i >= 2:
                #print(eq.w)
                #eq = self.transformations.normal_form(eq, minimize=True, mode = 'generation')
                eq = self.transformations.normal_form(eq, minimize=True, mode ='generation')#, mode = 'generation')
                if eq.level >=4 and self.args.generate_z3_unsolvable:
                    if not self.args.values_01:
                        eq = self.utils.check_satisfiability(eq, time=100)
                        if eq.sat !=0:
                            #print('unsatisfiable equation')
                            break
                    else:
                        eq = self.utils.check_sat_wrap(eq, time=100)
                        if eq.sat != 'unknown':
                            break


                eqsplit = eq.w.split('=')

                if len(eqsplit[0]) > self.args.side_maxlen_pool or len(eqsplit[1]) > self.args.side_maxlen_pool:
                    #print('side too big')
                    break
                if len(eqsplit[0]) <= 1 or len(eqsplit[1]) <= 1:
                   # print('side too small')
                    break
                # print('post normal',eq.w)
                neq = eq.deepcopy()

                #eq = self.transformations.normal_form(eq, mode = 'generation')
                self.log.append(neq)

                self.utils.check_satisfiability(neq)
                if neq.sat == -1:
                    print(f'Error: unsatisfiable equation {neq.w} found in '
                          f'generated pool. Pool: '
                          f'{[x.get_string_form() for x in self.log]}' )
                    self.utils.check_satisfiability(neq)

                    raise Exception(neq.w)

    def is_simple(self, eq, vars_to_check):
        """Simple : if it can be solved by removing all variables or one side has length less than 3 """
        # TODO: can be optimized


        if eq.w.split('=')[0] == eq.w.split('=')[1]:
            return True

        if self.args.quadratic_mode:
            return False

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
                if neq_split[0] in self.VARIABLES and self.is_constant(neq_split[1]):
                    if self.args.use_length_constraints:
                        if self.utils.treat_case_variable_side(neq.deepcopy(), neq_split[0], neq_split[1]):
                            return True
                    else:
                        return True
                if neq_split[1] in self.VARIABLES and self.is_constant(neq_split[0]):
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


        # for var in self.VARIABLES:
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
        return not bool(np.array([int(x in w) for x in self.VARIABLES]).sum())

    def generate_pool(self, size, level_list):

        p = []
        t = time.time()
        if self.args.generation_mode == 'standard':
            prob_insertion = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            initial_length = self.args.pool_max_initial_length  # random.choice([x+1 for x in range(self.args.pool_max_initial_length-1, self.args.pool_max_initial_length)])
        elif self.args.generation_mode == 'alternative':
            prob_insertion = random.choice([0.1, 0.2, 0.8, 0.9])
            initial_length = self.args.pool_max_initial_length
        elif self.args.generation_mode == 'constant_side':
            prob_insertion = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            initial_length = 3 * self.args.pool_max_initial_length
        elif self.args.generation_mode == 'quadratic-oriented':
            prob_insertion = random.choice([0.8])
        elif self.args.generation_mode == 'quadratic-oriented-linear':
            prob_insertion = random.choice([0.2, 0.9, 0.5])
            initial_length = 2  # max(1,int(self.args.pool_max_initial_length/4))
        elif self.args.generation_mode == 'regular-ordered':
            prob_insertion = random.choice([0.66])
            initial_length = int(self.args.pool_max_initial_length / 2)
        if self.args.quadratic_mode:
            initial_length = self.SIDE_MAX_LEN - len(self.VARIABLES)
            self.args.pool_max_initial_constants = len(self.ALPHABET)

        if self.args.test_mode:
            print(level_list)
        level_slots = self.args.train_level_slots

        while len(p) < len(level_list):
            for i, level_slot in enumerate(level_slots):
                if len(p) >= len(level_list):
                    break
                #while len(p) < size:
                # min_num_vars = random.choice([1])
                random_level = random.choice(level_slot)
                # num_initial_letters = random.choice([x + 1 for x in range(len(self.ALPHABET))])
                #if not self.args.test_mode:
                #    #random_level = level #level_list[-1]  # random.choice([x for x in level_list])
                #    if i == 0:
                #        random_level = max(level-1, self.args.min_level)
                #    elif i == 1:
                #        random_level = level
                #    else:
                #        random_level = random.choice([x for x in range(3, level+1)])
                #else:
                #    random_level = level
                find_next_equation = False

                while not find_next_equation:
                    #print('hola')
                    self.generate_sequence_of_eqns(level=random_level,
                                                   initial_alphabet=[x for x in self.ALPHABET[:self.args.pool_max_initial_constants]],
                                                   initial_length=initial_length,
                                                   prob_insertion=prob_insertion,
                                                   min_num_vars=1)

                    if len(self.log) > 1:
                        candidate_eq = self.log[-1]
                        if candidate_eq.level >= random_level:
                            candidate_eq = self.transformations.normal_form(candidate_eq, minimize=True)
                            if len(candidate_eq.w.split('=')[0]) >= 2 and len(candidate_eq.w.split('=')[1]) >= 2:
                                if not self.is_simple(candidate_eq, self.VARIABLES):
                                    if self.args.test_mode or (candidate_eq.w not in self.args.test_ws):
                                        if self.args.generate_z3_unsolvable:
                                            if not self.args.values_01:
                                                candidate_eq = self.utils.check_satisfiability(candidate_eq)
                                            else:
                                                candidate_eq = self.utils.check_sat_wrap(candidate_eq)
                                                if candidate_eq.sat == 'unknown':
                                                    p.append(candidate_eq)
                                                    find_next_equation = True
                                                    if self.args.test_mode:
                                                        print(len(p), candidate_eq.get_string_form(), candidate_eq.level)
                                        else:
                                            candidate_eq.sat = 0
                                        if candidate_eq.sat == 0:
                                            p.append(candidate_eq)
                                            find_next_equation = True
                                            if self.args.test_mode:
                                                print(len(p), candidate_eq.get_string_form(), candidate_eq.level)
                                    else:
                                        self.args.num_collisions_test += 1
        print([eq.level for eq in p] )
        logging.info(f'Elapsed time generating pool: {round(time.time()-t,2)}')
        self.pool_generation_time = round(time.time()-t, 2)
        self.pool = p


        self.save_pool(level_list, size)

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
            if not self.args.test_mode:
                Pickler(f).dump([x.w for x in self.pool])
            else:
                Pickler(f).dump(self.pool)
        f.close()