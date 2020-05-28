import time
import gc
import numpy as np

from .mcts import MCTS
from .word_equation.we import WE
from .word_equation.word_equation import WordEquation
from copy import deepcopy
import logging
import random
import os
from pickle import Pickler, Unpickler
from .SMTSolver import SMT_eval
import torch
from .utils import seed_everything


class Player(object):

    def __init__(self, args, nnet, mode='train', name='player', pool=[], previous_attempts_pool = [], seed=None):
        if seed is not None:
            self.seed = seed
            seed_everything(self.seed)
        else:
            self.seed = None
        T = 30
        args.timeout_time = T
        if args.hanoi or args.wordgame:
            args.timeout_time = T
            if args.active_tester:
                args.hanoi_max_lvl=16
        if args.active_tester:
            args.timeout_time=T
        args.timeout_mcts = args.timeout_time
        self.args = args

        #if not self.args.test_mode or mode == 'BENCHMARK': # in test mode having init log here doubles the displayed logging in the console
        self.init_log(args.folder_name, mode)#'test' if mode == 'test' else 'train')
        self.nnet = nnet
        self.we = WE(args, self.seed)
        print(f'seed {self.seed}')
        z3_timeout = 10000 if not self.args.active_tester else int(self.args.timeout_time*1000)
        #self.z3converter = SMTSolver(self.args.VARIABLES, self.args.ALPHABET, z3_timeout, self.args.use_length_constraints)
        self.train_examples = []

        self.num_attempts_state = {}
        self.num_solution_state = {}

        self.level = 40# self.args.level #if mode != 'BENCHMARK' else self.args.max_level_benchmark
        if self.args.hanoi:
            self.level=self.args.hanoi_max_lvl
            print('hanoi max lvl: ', self.args.hanoi_max_lvl)
        self.mode = mode
        self.device = self.args.play_device
        self.type_of_benchmark = self.args.type_of_benchmark
        #
        # if mode == 'train':
        self.temp =  args.temp
        self.max_steps = args.allowance_factor * self.level
        self.num_equations = args.num_equations_train
        self.pool_filename = None
        if mode == 'train':
            self.timeout_time = self.args.timeout_time
            self.timeout_mcts = self.args.timeout_mcts
            self.num_mcts_simulations = self.args.num_mcts_simulations
        elif mode == 'BENCHMARK':
            self.timeout_time = self.args.timeout_time_benchmark
            self.timeout_mcts = self.args.timeout_mcts_benchmark
            self.num_mcts_simulations = self.args.num_mcts_simulations_benchmark
            if self.args.type_of_benchmark == 'standard':
                self.pool_filename = self.args.test_mode_pool_filename
            elif self.args.type_of_benchmark == 'regular-ordered':
                self.pool_filename = 'benchmarks/pool_lvl_6_30_size_100_regular-ordered.pth.tar'
        elif mode == 'test':
            self.num_mcts_simulations = self.args.num_mcts_simulations
            self.timeout_time =  self.args.timeout_time
            self.timeout_mcts =  self.args.timeout_mcts
            print(f'test mode, timeout: {self.args.timeout_time}, sims: {self.num_mcts_simulations}')
            if self.args.type_of_benchmark == 'standard':
                self.pool_filename = self.args.test_mode_pool_filename
            elif self.args.type_of_benchmark == 'regular-ordered':
                self.pool_filename = 'benchmarks/pool_lvl_6_30_size_100_regular-ordered.pth.tar'

        if mode != 'train':
            self.temp = 0.
            # if self.args.mcgs_type <= 3 or self.args.nnet_type != 'mcgs':
            #     self.args.mcgs_persistence = 0.9999

        self.score = self.initiate_container([])
        self.sat_steps_taken = []
        self.search_times = self.initiate_container([])
        self.action_indices = self.initiate_container([])
        self.num_pop_actions =self.initiate_container(0)
        self.num_compress_actions = self.initiate_container(0)
        self.num_delete_actions = self.initiate_container(0)
        self.percentage_actions = [self.initiate_container(0) for _ in range(3)]
        self.num_total_actions = self.initiate_container(0)
        self.num_successful_plays_at_level = self.args.num_successful_plays_at_level
        self.num_plays_at_current_level = self.args.num_plays_at_current_level
        self.num_timeouts = self.initiate_container(0)
        self.num_fails = self.initiate_container(0)
        self.percentage_timeouts_solved=self.initiate_container([])
        self.percentage_timeouts_failed=self.initiate_container([])
        self.z3score = self.initiate_container([])
        self.z3mctstimes_sat = self.initiate_container([])
        self.z3mctstimes_unsat = self.initiate_container([])
        self.eqs_solved = []
        self.eqs_solved_z3 = []
        self.times_eqs_solved_Z3 = []
        if self.args.type_of_benchmark is not None:
            self.name = name + '_' + self.args.type_of_benchmark
        else:
            self.name = name
        self.player_index = int(self.name.split('_')[1])
        self.failed_pool = pool
        self.previous_attempts_pool = previous_attempts_pool

        self.truncate = False
        self.eqs = []
        self.pool = None if pool==[] else pool
        self.pool_generation_time = None

        self.discount = args.discount

    def initiate_container(self, value):
        l = self.level if not self.args.hanoi else self.args.hanoi_max_lvl
        if type(value) == list:
            return [value.copy() for _ in range(l)]
        else:
            return [value for _ in range(l)]

    def update_action_logs(self, action, level):
        try:
            self.action_indices[level-1].append(action)
        except:
            print('h')


    def init_log(self, folder, mode='train'):

        # TODO: What is this for?
        #logging.getLogger("tensorflow").setLevel(logging.WARNING)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=folder + f'/log_{mode}.log')
        console = logging.StreamHandler()
        from PIL import PngImagePlugin
        logger = logging.getLogger(PngImagePlugin.__name__)
        logger.setLevel(logging.INFO)  # tame the "STREAM" debug messages

        console.setLevel(logging.ERROR)
        logging.getLogger('').addHandler(console)

    def timeout(self, elapsed_time, episode_step, level, eq):
        return elapsed_time > self.args.timeout_time#*1.25

    def execute_episode(self, eq=None, verbose=0):

        self.mcts.initial_time = time.time()

        train_examples = []
        episode_step = 0
        num_repetitions = 0
        if verbose == 1:
            eq.not_normalized = eq.w

        log = [eq.w]
        t = time.time()
        level = eq.level
        eq.s_eq=None

        eq.sat = 0


        eq.initial_state = True
        prev_state=eq.deepcopy(copy_id = True)
        prev_state.initial_state = True
        self.mcts.previous_state_context[prev_state.id] = torch.zeros(1, self.args.linear_hidden_size, dtype = torch.float)
        self.mcts.is_leaf[eq.w] = True
        self.mcts.is_leaf[prev_state.w] = True


        while True:
            if not self.args.use_oracle:
                s_eq = self.we.utils.format_state(eq)
            else:
                s_eq = None
            eq.s_eq = s_eq
            pi = self.mcts.get_action_prob(eq, s_eq, temp=self.temp, previous_state=prev_state)
            if pi == 'already_sat':
                return [(0,0,1)], ['already_solved']
            examples = [(s_eq, pi, None)]
            train_examples += examples

            if type(pi) == int:
                print('what is happening?')
            action = np.random.choice(len(pi), p=np.array(pi))

            prev_state = eq.deepcopy(copy_id = True)

            new_eq = self.we.moves.act(eq, action, verbose)
            eq = new_eq

            if self.args.forbid_repetitions:
                if eq.w in log:
                    num_repetitions +=1

            episode_step += 1
            tt = round(time.time() - t, 2)
            log += [eq.get_string_form_for_print()]

            if self.timeout(tt, episode_step, level, eq) or\
                    (self.args.forbid_repetitions and num_repetitions > 1/2):
                r =  self.args.evaluation_after_exceeded_steps_or_time
                return [(x[0], x[1], r*(self.discount**i)) for i, x in enumerate(train_examples[::-1])], log

            if self.mcts.found_sol[0]:
                if self.mode == 'test' or self.args.oracle:
                    print('found early solution')
                    if verbose == 1:
                        candidate_sol = eq.simplify_candidate_sol()
                        log.append(candidate_sol)
                    return [(x[0], x[1], self.discount**(i)) for i,x in enumerate(train_examples[::-1])],  log  #+ self.mcts.found_sol[1]*['early_sol']

            #eq = self.we.utils.check_satisfiability(eq, smt_time=self.args.mcts_smt_time_max)
            if eq.w in self.mcts.final_state_value:
                eq.sat = self.mcts.final_state_value[eq.w]
            else:
                eq = self.we.utils.check_satisfiability(eq, smt_time=self.args.mcts_smt_time_max)

            if eq.sat != 0:
                if verbose == 1:
                    candidate_sol = eq.simplify_candidate_sol()
                    log.append(candidate_sol)
                if eq.sat == 1:
                    return [(x[0], x[1], self.discount**i) for i, x in enumerate(train_examples[::-1])], log
                else:
                    return [(x[0], x[1], -self.discount**i) for i, x in enumerate(train_examples[::-1])], log


    def play(self, level_list=None):

        self.nnet.model.eval()

        if self.pool is not None:
            pool = self.pool
            print(pool)
        else:
            if self.pool_filename is None:  # or (not self.args.test_mode):
                level_list = 10*[self.level] if level_list is None else level_list
                print(self.args.VARIABLES)
                print('Generating pool..')
                self.we.generator.generate_pool(self.num_equations, level_list)#, self.seed)
                pool=self.we.generator.pool
                #pool = self.transform_failed_pool() + list(self.we.generator.pool)[len(self.failed_pool):]
                self.pool_generation_time = self.we.generator.pool_generation_time
                print('pool generated')
            else:
                file_name = self.pool_filename
                print('Loading {}'.format(file_name))
                if os.path.exists(file_name):
                    with open(file_name, "rb") as f:
                        pool = Unpickler(f).load()
                        self.pool_generation_time = 0

            level_order = [[eq.level, i] for i,eq in enumerate(pool)]
            level_order.sort(reverse= True)
            pool = [pool[lvl[1]] for lvl in level_order]

        self.execution_times_sat = self.initiate_container([])
        self.execution_times_unsat = self.initiate_container([])
        self.sat_steps_taken = []
        absolute_initial_time = time.time()
        new_failed_pool = []
        new_previous_attempts_pool = []
        print(pool)
        for i, eq in enumerate(pool):
            eq_original=eq.w
            if i > 0:
                self.nn_outputs.update(self.mcts.nn_outputs)
            else:
                self.nn_outputs = {}

            if type(eq) == str:
                print('what', eq)
                break

            self.mcts = MCTS(self.nnet,
                             self.args,
                             self.num_mcts_simulations,
                             self.timeout_mcts,
                             self.args.max_steps(eq.level), #self.args.allowance_factor * self.level,
                             self.level,
                             self.nn_outputs,
                             self.mode,
                             self.seed)


            self.mcts.name = self.name

            # todo: not needed?
            verbose=0

            z3out = 0
            if self.args.active_tester and self.args.test_solver:
                t = time.time()
                z3out = SMT_eval(self.args, eq)
                z3time = time.time()-t
                if z3out > 0:
                    self.eqs_solved_z3.append([eq.level, eq.w, z3time])
            else:
                z3out = 0

            self.z3score[eq.level-1].append(z3out)
            if z3out != 0:
                if z3out > 0:
                    self.z3mctstimes_sat[eq.level-1].append(z3time)
            initial_local_time = time.time()
            eq = self.we.transformations.normal_form(eq)
            examples, log = self.execute_episode(eq, verbose)

            if False:
                if z3out <= 0:
                    examples, log = self.execute_episode(eq, verbose)
                    #
                    #examples, log = [[10, None, -1], [10, None, -1], [10, None, -1]], ['fail', 'fail','fail']
                else:
                    examples, log = [[10,None, 1], [10, None, 1], [10, None, 1]], ['already_solved', 'already_solved', 'already_solved']
            reps = self.mcts.num_reps

            local_execution_time = round(time.time() - initial_local_time, 4)
            len_eq = len(log)

            printout_info = [i, '', eq.level, len_eq, local_execution_time,
                             round(self.timeout_time,2),
                             round(self.timeout_mcts,4),
                             self.num_mcts_simulations,
                             0, z3out]
            if not self.args.cnf_benchmark:
                printout_info += [log]

            if examples[0][2] >= 1  or log[-1] == 'early_sol':
                self.score[eq.level-1].append(1)
                printout_info[1] = 'New sol'
                logging.error(self.execution_printout(*printout_info))
                self.execution_times_sat[eq.level-1].append(local_execution_time)
                self.sat_steps_taken.append(len_eq)
                self.search_times[eq.level-1]+= self.mcts.search_times
                self.z3mctstimes_sat[eq.level-1].append(self.mcts.meanz3times[0])
                if self.args.active_tester:
                    self.eqs_solved.append([eq.level, eq_original, local_execution_time])


            elif (not self.args.values01 and examples[0][2] == 0):
                self.score[eq.level-1].append(0)
                printout_info[1] = 'Stopped'
                self.num_timeouts[eq.level-1] += 1
                logging.error(self.execution_printout(*printout_info))

            elif examples[0][2] < 0:

                self.score[eq.level-1].append(0)
                self.z3mctstimes_unsat[eq.level-1].append(self.mcts.meanz3times[0])

                printout_info[1] = 'Fail'
                self.num_fails[eq.level-1] += 1
                logging.error(self.execution_printout(*printout_info))

            if log[0] != 'already_solved':
                self.train_examples += examples

         #   else:
         #       if self.get_score() < self.args.min_eqs_for_successful_test_player and self.args.test_mode:
         #           self.truncate = True
         #           print('TRUNCATE!')
         #       else:
         #           print('Finished level early!')

        gc.collect()
        self.absolute_execution_time = round(time.time()-absolute_initial_time, 2)
        self.failed_pool = new_failed_pool
        self.previous_attempts_pool = new_previous_attempts_pool

        if self.mode == 'BENCHMARK':
            logging.error(f'*+*+*+*+*+*+*\n'
                         f'benchmark for {self.name} score of {self.score} in {self.absolute_execution_time} seconds\n')

    def execution_printout(self, i, event, level,
                           steps, local_execution_time, timout_time, timeout_mcts, num_mcts, previous_attempts, z3out =None, log =[], proportion_evals=None,z3side1 = None, z3side2 = None):
        if not self.args.use_solver:
            return f'{self.mcts.num_reps}-{self.level} - {level} - {event} - {self.num_successful_plays_at_level}/{self.num_plays_at_current_level} - {self.name} - Eq: ' \
                f'{self.get_score()}/{i + 1}/{self.num_equations}: Steps {steps}, Time: {local_execution_time}, Timeout (episode/mcts): {timout_time}/{timeout_mcts},' \
                f' Num mcts sims: {num_mcts}, Previous attempts: {previous_attempts}, {log}'
        else:
            if not self.args.check_ztimeout3_conversion:
                return f'{proportion_evals}-{self.level} - {level} - {event} - {self.num_successful_plays_at_level}/{self.num_plays_at_current_level} - {self.name} - Eq: ' \
                    f'{self.get_score()}/{i + 1}/{self.num_equations}: Steps {steps}, Time: {local_execution_time}, Timeout (episode/mcts): {timout_time}/{timeout_mcts},' \
                    f' Num mcts sims: {num_mcts}, Previous attempts: {previous_attempts}, {log}'
            else:
                return f'{z3out}//{z3side1}={z3side2}:::::{self.level} - {level} - {event} - {self.num_successful_plays_at_level}/{self.num_plays_at_current_level} - {self.name} - Eq: ' \
                    f'{self.get_score()}/{i + 1}/{self.num_equations}: Steps {steps}, Time: {local_execution_time}, Timeout (episode/mcts): {timout_time}/{timeout_mcts},' \
                    f' Num mcts sims: {num_mcts}, Previous attempts: {previous_attempts}, {log}'

    def len_list_for_div(self, l):
        return len(l) if len(l) != 0 else 1


    def get_score(self):
        return sum([sum(x) for x in self.score])


    #def do_symmetrized_examples(self, eq, probs):
    #    symmetrized_examples = [(self.we.utils.format_state(eq), probs, None)]
    #    eq.update_used_symbols()
    #    max_letter_idx = max([self.args.symbol_indices[x] for x in eq.used_alphabet]) if len(eq.used_alphabet) != 0 else 0
    #    max_var_idx = max([self.args.symbol_indices[x] for x in eq.used_variables]) if len(eq.used_variables) != 0 else 0
    #    max_letter = self.args.symbol_indices_reverse[max_letter_idx]
    #    max_variable = self.args.symbol_indices_reverse[max_var_idx]
    #    max_letter_idx = np.argmax([max_letter == letter for letter in self.args.ALPHABET])
    #    max_variable_idx = np.argmax([max_variable == variable for variable in self.args.VARIABLES])
    #    reached_max_letter = max_letter_idx == len(self.args.ALPHABET)-1
    #    reached_max_var = max_variable_idx == len(self.args.VARIABLES)-1
    #    while not reached_max_letter or not reached_max_var:
    #        if not reached_max_letter and not reached_max_var:
    #            mode = 'all'
    #            max_letter_idx += 1
    #            max_variable_idx += 1
    #        elif not reached_max_letter:
    #            mode = 'letters'
    #            max_letter_idx += 1
    #        else:
    #            mode = 'variables'
    #            max_variable_idx += 1
    #        eq = self.increase_symbols_by_one(eq, mode).deepcopy()
    #        probs = self.get_new_actions(probs, mode)
    #        symmetrized_examples.append((self.we.utils.format_state(eq), probs, None))
    #        reached_max_letter = max_letter_idx == len(self.args.ALPHABET) - 1
    #        reached_max_var = max_variable_idx == len(self.args.VARIABLES) - 1
#
    #    return symmetrized_examples

    #def increase_symbols_by_one(self, eq, mode):
    #    auto_plus_one = {}
    #    if mode == 'all' or mode == 'variables':
    #        auto_plus_one.update({x: self.args.VARIABLES[i+1] for i, x in enumerate(self.args.VARIABLES[:-1])})
    #        auto_plus_one[self.args.VARIABLES[-1]] = self.args.VARIABLES[0]
    #        if mode == 'variables':
    #            auto_plus_one.update({x: x for x in self.args.ALPHABET})
    #    if mode == 'all' or mode == 'letters':
    #        auto_plus_one.update({x: self.args.ALPHABET[i+1] for i, x in enumerate(self.args.ALPHABET[:-1])})
    #        auto_plus_one[self.args.ALPHABET[-1]] = self.args.ALPHABET[0]
    #        if mode == 'letters':
    #            auto_plus_one.update({x: x for x in self.args.VARIABLES})
#
    #    eq = self.we.transformations.apply_automorphism(eq, auto_plus_one)
    #    return eq
#
    #def get_new_actions(self, prob_actions, mode):
    #    num_vars = len(self.args.VARIABLES)
    #    num_letters = len(self.args.ALPHABET)
    #    new_probs = np.zeros(len(prob_actions))
    #    if mode == 'all' or mode == 'variables':
    #        new_probs[1:num_vars] = prob_actions[0:num_vars-1]
#
    #    if mode == 'all' or mode == 'letters':
    #        new_probs[num_vars + num_letters + 1: num_vars + num_letters*num_letters] = \
    #            prob_actions[num_vars: num_vars + num_letters*num_letters - num_letters - 1]
#
    #    if mode == 'all':
    #        new_probs[num_vars + num_letters*num_letters + 2*num_vars + 2: len(prob_actions)] = \
    #            prob_actions[num_vars + num_letters*num_letters: len(prob_actions) - 2*num_vars - 2]
#
    #    if mode == 'letters':
    #        new_probs[num_vars + num_letters*num_letters + 2*num_vars: len(prob_actions)] = \
    #            prob_actions[num_vars + num_letters*num_letters: len(prob_actions) - 2*num_vars]
#
    #    if mode == 'variables':
    #        new_probs[num_vars + num_letters*num_letters + 2: len(prob_actions)] = \
    #            prob_actions[num_vars + num_letters*num_letters: len(prob_actions) - 2]
#
    #    return new_probs


    #ef augment_examples(self, examples):
    #   new_examples = []
    #   for ex in examples:
    #       num_augment = 1 if ex[2] > 0 else 3
    #       chunk = []
    #       for _ in range(num_augment):
    #           if _ > 0:
    #               e0 = deepcopy(ex[0])
    #               e1 = 0.85*np.array(ex[1].copy()) + 0.15*np.random.dirichlet(self.args.num_actions*[self.args.noise_param])
    #               e1 = list(e1)
    #               if ex[2] > 0:
    #                   e2 = min(0.8 + 0.2*np.random.normal(0,0.7), 1)
    #               else:
    #                   e2 = max(-0.8 + 0.2*np.random.normal(0,0.7), -1)

    #               chunk += [[e0, e1, e2]]
    #           else:
    #               chunk += [ex]

    #       new_examples += chunk
    #   return new_examples



    #def allowed_to_play(self, i, mode):
    #    return True
    #    if self.args.test_mode:
    #        return True
    #    if self.args.truncate:
    #        if mode == 'train':
    #            return True
    #        elif mode == 'test':
    #            if self.get_score() >= self.args.min_eqs_for_successful_test_player:
    #                return False
    #            if i - self.get_score() > self.args.num_equations_test - self.args.min_eqs_for_successful_test_player:
    #                return False
    #        return True
    #    else:
    #        return True

    # def transform_failed_pool(self):
    #    return [WordEquation(self.args, x) for x in self.failed_pool]

    #def find_num_mcts_and_timeout(self, level, mode):
    #    if mode == 'train':
    #        num_mcts = self.args.num_mcts_function(level)
    #        timeout = self.args.timeout_time
    #    if mode == 'test':
    #        num_mcts = self.args.num_mcts_function(level)
    #        timeout = self.args.timeout_time
    #    if mode == 'BENCHMARK':
    #        num_mcts = self.args.num_mcts_simulations_benchmark
    #        timeout = self.args.timeout_time_benchmark
    #    return num_mcts, timeout


    # def get_random_symmetrizations(self, eq, pi):
    #    examples = [(self.we.utils.format_state(eq), pi, None)]
    #    for _ in range(self.args.num_random_symmetrizations):
    #        alphabet = [x for x in self.args.ALPHABET]
    #        random.shuffle(alphabet)
    #        alphabet_auto = {self.args.ALPHABET[i]: alphabet[i] for i in range(len(alphabet))}
    #
    #        variables = [x for x in self.args.VARIABLES]
    #        random.shuffle(variables)
    #        variables_auto = {self.args.VARIABLES[i]: variables[i] for i in range(len(variables))}
    #
    #        auto = alphabet_auto
    #        auto.update({x: i for x,i in variables_auto.items()})
    #
    #        new_eq = self.we.transformations.apply_automorphism(eq.deepcopy(), auto)
    #        if not self.args.quadratic_mode:
    #            new_pi = self.apply_automorphism_to_actions(deepcopy(pi), auto)
    #        else:
    #            new_pi = deepcopy(pi)
    #
    #        examples.append((self.we.utils.format_state(new_eq), new_pi, None))
    #        #if self.args.learnable_smt:
    #        #    eqs_z3.append(eq.z3out)
    #    return examples
    #
    # def apply_automorphism_to_actions(self, pi, auto):
    #
    #    fast_dict = self.we.moves.fast_dict
    #    new_pi = [0 for _ in range(len(pi))]
    #    for x in self.args.VARIABLES:
    #        original_action_index = fast_dict['delete'][x]
    #        new_action_index = fast_dict['delete'][auto[x]]
    #        new_pi[original_action_index] = pi[new_action_index]
    #
    #    for x in self.args.ALPHABET:
    #        for y in self.args.ALPHABET:
    #            original_action_index = fast_dict['compress'][x][y]
    #            new_action_index = fast_dict['compress'][auto[x]][auto[y]]
    #            new_pi[original_action_index] = pi[new_action_index]
    #
    #    for x in self.args.ALPHABET:
    #        for s in ['left', 'right']:
    #            for y in self.args.VARIABLES:
    #                original_action_index = fast_dict['pop'][x][s][y]
    #                new_action_index = fast_dict['pop'][auto[x]][s][auto[y]]
    #                new_pi[original_action_index] = pi[new_action_index]
    #    return new_pi
