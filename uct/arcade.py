# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:28:18 2019

@author: garre
"""

import random
import numpy as np
import time
from multiprocessing import Process, Queue
from .player import Player
from .neural_net_wrapper import NNetWrapper
import logging
from .utils import Utils, seed_everything
import os
import pandas as pd
import copy
import matplotlib.pyplot as plt
from .neural_net_models.uniform_model import UniformModel
plt.interactive(True)


class Arcade(object):

    def __init__(self, args, load, name=''):


        mode = args.mode
        self.init_log(args.folder_name, mode)
        logging.error('\n\n\nNEW ARCADE ---- load: {} ---- {}'.format(load, args.folder_name))
        logging.error(f'Test mode: {args.test_mode}')
        self.args = args


        seed_everything(self.args.seed_class-1)

        self.utils = Utils(args)

        if self.args.load_model:
            self.args = self.utils.load_object('arguments')
            self.args.load_model = True
            self.train_examples_history =[[]]
            self.args.search_times_train_this_level_log = []
            self.args.search_times_test_this_level_log = []

        else:
            # TODO: Better way to organize logs: dictionary plus list of log
            #  names (allows to unifiy several functions later on)
            self.train_examples_history = [[]]

        self.previous_process_play_time = time.time()


        logging.error(vars(self.args))
        logging.error(f'NUM CPUs:{self.args.num_cpus}')

        self.active_players = self.args.num_cpus*[False]
        self.received_play_results = self.args.num_cpus*[False]
        self.active_training = False
        self.received_train_results = False
        self.benchmark_due = False # if not self.args.load_model else True
        self.ongoing_benchmark = False
        self.test_due = False # if not self.args.load_model else True
        self.ongoing_test = False

        if not self.args.load_model:
            def initiate_container(value):
                if type(value) == list:
                    return [value.copy() for _ in range(self.args.level)]
                else:
                    return [value for _ in range(self.args.level)]
            self.args.sat_times_train_mean = initiate_container([])
            self.args.sat_times_train_std = initiate_container([])
            self.args.sat_times_train_max = initiate_container([])
            self.args.sat_steps_train_mean = initiate_container([])
            self.args.sat_steps_train_std = initiate_container([])
            self.args.sat_steps_train_max = initiate_container([])
            self.args.precentage_of_actions_train = initiate_container([])
            self.args.num_solved_train = initiate_container([])
            self.args.num_failed_train = initiate_container([])
            self.args.num_timeouts_train = initiate_container([])
            self.args.search_times_train_mean = initiate_container([])
            self.args.search_times_train_max = initiate_container([])

            self.args.sat_times_test_mean = initiate_container([])
            self.args.sat_times_test_std = initiate_container([])
            self.args.sat_times_test_max = initiate_container([])
            self.args.sat_steps_test_mean = initiate_container([])
            self.args.sat_steps_test_std = initiate_container([])
            self.args.sat_steps_test_max = initiate_container([])
            self.args.precentage_of_actions_test = initiate_container([])
            self.args.num_solved_test = initiate_container([])
            self.args.num_failed_test = initiate_container([])
            self.args.num_timeouts_test = initiate_container([])
            self.args.search_times_test_mean = initiate_container([])
            self.args.search_times_test_max = initiate_container([])
            self.args.score_test_this_level = initiate_container([])
            self.args.timeouts_test_this_level = initiate_container([])
            self.args.fails_test_this_level = initiate_container([])

            self.args.current_level_sat_times_train = []  # self.args.max_level*[[]]
            self.args.current_level_sat_times_train = []  # self.args.max_level*[[]]
            self.args.current_level_sat_times_train = []  # self.args.max_level*[[]]
            self.args.sat_steps_taken_in_level_train = []  # self.args.max_level*[[]]
            self.args.sat_steps_taken_in_level_train = []  # self.args.max_level*[[]]
            self.args.sat_steps_taken_in_level_train = []  # self.args.max_level*[[]]
            self.args.percentage_of_actions_this_level_train = []  # self.args.max_level*[[]]
            # selargs.s.success_fail_term_log_this_level_train = []  # self.args.max_level*[[]]
            self.args.search_times_train_this_level_log = []  # self.args.max_level*[[]]
            self.args.score_train_this_level = []  # initiate_container([])
            self.args.timeouts_train_this_level = []  # initiate_container([])
            self.args.fails_train_this_level = []  # initiate_container([])

            self.args.current_level_sat_times_test = []  # self.args.max_level * [[]]
            self.args.current_level_sat_times_test = []  # self.args.max_level * [[]]
            self.args.current_level_sat_times_test = []  # self.args.max_level * [[]]
            self.args.sat_steps_taken_in_level_test = []  # self.args.max_level * [[]]
            self.args.sat_steps_taken_in_level_test = []  # self.args.max_level * [[]]
            self.args.sat_steps_taken_in_level_test = []  # self.args.max_level * [[]]
            self.args.percentage_of_actions_this_level_test = []  # self.args.max_level * [[]]
            # selargs.s.success_fail_term_log_this_level_test = [] # self.args.max_level * [[]]
            self.args.search_times_test_this_level_log = []  # self.args.max_level*[[]]
            self.args.score_test_this_level = []  # initiate_container([])
            self.args.timeouts_test_this_level = []  # initiate_container([])
            self.args.fails_test_this_level = []  # initiate_container([])

            self.args.percentage_timeouts_failed_mean =initiate_container([])
            self.args.percentage_timeouts_failed_this_level = []

            self.args.percentage_timeouts_solved_mean = initiate_container([])
            self.args.percentage_timeouts_solved_this_level = []

            self.args.z3score_level_train = []
            self.args.z3score_train = initiate_container([])
            self.args.z3score_level_test = []
            self.args.z3score_test = initiate_container([])

            self.args.eq_history = [[]]

            self.args.z3mctstime_sat_train = []
            self.args.z3mctstime_unsat_train = []
            self.args.z3mctstime_sat_test = []
            self.args.z3mctstime_unsat_test = []
            self.args.z3mctstime_sat_test_final =[]
            self.args.z3mctstime_unsat_test_final = []
            self.args.z3mctstime_sat_train_final = []
            self.args.z3mctstime_unsat_train_final = []

            self.args.action_indices_level = []

            self.args.evolution_test = [0]
            self.args.iterations_train = [0]
            self.args.iterations_test =[0]
            self.args.evolution_train = [0]
            self.args.evolution_test_eq = [0]
            self.args.iterations_train_eq = [0]
            self.args.model_names = ['']

            self.args.iterations_test_eq = [0]
            self.args.evolution_train_eq = [0]
        self.name = name

    def init_log(self, folder_name, mode='train'):
        import logging
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # TODO: What is this for?
        #logging.getLogger("tensorflow").setLevel(logging.WARNING)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=folder_name + f'/log_{mode}.log')
        console = logging.StreamHandler()
        from PIL import PngImagePlugin
        logger = logging.getLogger(PngImagePlugin.__name__)
        logger.setLevel(logging.INFO)  # tame the "STREAM" debug messages

        console.setLevel(logging.ERROR)
        logging.getLogger('').addHandler(console)


    def init_dict_of_process_and_queus(self):
        pipes_play = {_: Queue() for _ in range(self.args.num_cpus)}
        m = 'train' if not self.args.active_tester else 'test'
        modes = self.args.num_cpus * [m]  # + 1 * ['test']
        if self.test_due:
            modes[0] = 'test'

        processes_play ={}
        for _ in range(self.args.num_cpus):

            self.args.num_init_players += 1
            seed = self.args.num_init_players + 1000*self.args.seed_class
            processes_play.update({_: Process(
                target=self.individual_player_session,
                args=([self.args,
                       self.model_play,
                       modes[_],
                       pipes_play[_], _,
                       seed],))})
            self.active_players[_] = True
            processes_play[_].start()
            self.args.total_plays += 1
        player_levels = {_: self.args.level for _ in range(self.args.num_cpus)}

        return processes_play, pipes_play, modes, player_levels

    def run_play_mode(self, pools=None):
        self.args.initial_level_time = time.time()

        if self.args.load_model:
            self.args.modification_when_loaded()
            self.benchmark_due = False
            self.initiate_loop = not self.benchmark_due
            self.ongoing_benchmark = False
            self.test_due = False # True
            self.args.skip_first_benchmark = False
            self.train_examples_history =[[]]
        else:
            self.initiate_loop = True
        #processes_play, pipes_play, modes, player_levels = self.init_dict_of_process_and_queus()

        self.args.total_plays = 0
        iteration = 0

        seed = self.args.num_init_players + 10000 * self.args.seed_class

        parent_conn_train = Queue()
        train_mode = 'normal'#'initialize'
        p_train = Process(target=self.arcade_train,
                          args=[self.args,
                                self.model_play,
                                self.train_examples_history,
                                parent_conn_train,
                                train_mode,
                                None,
                                seed])
        p_train.start()
        active_time = time.time()
        self.quarters = 0 + int(self.args.initial_time/3600)
        self.args.checkpoint_num_plays = 0
        while (self.args.checkpoint_num_plays < self.args.max_num_plays and
               not self.args.active_tester) or \
                (self.args.active_tester and self.args.checkpoint_num_plays < self.args.max_num_plays and
                 self.args.new_play_examples_available < len([x for x in self.args.pools if type(x) != str])) :
            #print(time.time() -active_time)
            folder_name = self.args.folder_name
            #folder_name = os.path.join('\\', 'we 0.1', 'test_files', folder_name)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)

            if ((time.time() + self.args.initial_time-active_time)/600 - self.quarters)> 0:
                if not self.args.active_tester:
                    self.model_play.save_checkpoint(folder=self.args.folder_name,
                                                    filename=f'model_min_{10*self.quarters}.pth.tar')

                self.quarters += 1
                self.args.evolution_test.append(0)
                self.args.evolution_train.append(0)
                self.args.iterations_test.append(0)
                self.args.iterations_train.append(0)
                self.args.model_names.append('')

            if self.args.total_plays - self.args.checkpoint_num_plays > 0:

                if not self.args.active_tester:
                    self.model_play.save_checkpoint(folder=self.args.folder_name,
                                                    filename=f'model_plays_{self.args.checkpoint_num_plays}.pth.tar')

                    self.args.model_names[-1] = f'model_plays_{self.args.checkpoint_num_plays}.pth.tar'

                #if self.args.num_init_players % 5 == 0 and self.args.num_init_players > 0 and self.args.num_init_players > prior_num_init_players:
                #    self.args.checkpoint_num_plays = 5 + self.args.num_init_players
                # prior_num_init_players = self.args.num_init_players
                print(f'ev/it train eq: {self.args.evolution_test_eq}, {self.args.iterations_test_eq}')
                print(f'total plays {self.args.total_plays}, checkpoint num plays: {self.args.checkpoint_num_plays}')
                print(self.args.model_names)

                if False:#len(self.args.evolution_test_eq) > 3:
                    last_score = self.args.evolution_test_eq[-1] / max(1., self.args.iterations_test_eq[-1])
                    scores___ = [self.args.evolution_test_eq[i] / max(1., self.args.iterations_test_eq[i]) for i
                                 in range(3,len(self.args.evolution_test_eq))]
                    current_best_score_idx = np.argmax(scores___[::-1])+1#retorns the rightmost argmax
                    current_best_score = scores___[-current_best_score_idx]
                    print(f'last score: {last_score}, best score: {current_best_score}')

                    if last_score < current_best_score and not self.args.active_tester:
                        print(f'loading file model_plays_{(3+ current_best_score_idx) * 5}.pth.tar')

                        self.model_play = self.utils.load_nnet(device='cpu',  # TODO: what does this argument do?
                                                               training=True,
                                                               load=True,
                                                               folder=self.args.folder_name,
                                                               filename=self.args.model_names[-current_best_score_idx]#f'model_plays_{( current_best_score_idx) * 5}.pth.tar'
                                                               )  # TODO: do I need to initialize the model like this?load_model()

                self.args.checkpoint_num_plays += 5
                self.args.evolution_test_eq.append(0)
                self.args.evolution_train_eq.append(0)
                self.args.iterations_test_eq.append(0)
                self.args.iterations_train_eq.append(0)
                self.args.model_names.append('')

            iteration += 1
#
            if self.initiate_loop:
                self.initiate_loop = False
                processes_play, pipes_play, modes, player_levels = self.init_dict_of_process_and_queus()


            if self.args.active_tester or \
                    (not self.args.test_mode):
                for _ in range(self.args.num_cpus):
                    try:
                        if not pipes_play[_].empty():  # or (iteration==1 and self.args.load_model):

                            print(f'Hola player {_}')
                            self.active_players[_] = False
                            result_play = pipes_play[_].get(block=True)
                            self.args.num_finished_play_session_per_player[_] += 1
                            self.process_play_result(**result_play)

                            pipes_play[_].close()
                            pipes_play[_].join_thread()
                            processes_play[_].join()
                            processes_play[_].close()

                            print(self.args.test_mode, self.args.total_plays, self.args.save_model)
                            self.save_data()
                            self.args.new_play_examples_available += 1

                            if not self.active_players[_]:
                                if (self.args.active_tester and self.args.num_init_players < len(self.args.pools)) or not self.args.active_tester:
                                    if (self.test_due and (not self.ongoing_test)) or self.args.active_tester:
                                        mode = 'test'
                                        self.test_due = False
                                        self.ongoing_test = True
                                    else:
                                        mode = 'train'  # modes[_]

                                    processes_play, pipes_play, player_levels = self.init_player(processes_play, pipes_play, mode,
                                                                                                 player_levels, _)
                                    self.active_players[_] = True
                                    processes_play[_].start()
                                    self.args.total_plays += 1

                            time.sleep(2.)
                    except:
                        #print(f'could not check emptiness of  pipe {_}')
                        #print(self.args.total_plays, self.args.checkpoint_num_plays, self.args.num_init_players, self.args.new_play_examples_available)
                        #self.args.new_play_examples_available += 1
                        #self.args.total_plays += 1
                        pass

                if not self.args.test_mode:
                    if self.args.new_play_examples_available >= self.args.num_play_iterations_before_test_iteration:  # or total_plays == 0:
                        if (not parent_conn_train.empty()) or (iteration == 1 and self.args.load_model):

                            print('Getting train results')
                            result_train = parent_conn_train.get(block=True)

                            self.process_train_result(**result_train)
                            p_train.join()
                            p_train.close()
                            parent_conn_train.close()
                            parent_conn_train.join_thread()

                            parent_conn_train = Queue()

                            seed = self.args.num_init_players + 10000 * self.args.seed_class

                            p_train = Process(target=self.arcade_train,
                                              args=[self.args,
                                                    self.model_play,
                                                    self.train_examples_history,
                                                    parent_conn_train,
                                                    'normal',
                                                    self.args.eq_history, seed])

                            self.args.new_play_examples_available = 0


                            if iteration == 0 and self.args.load_model:
                                pass
                            else:
                                self.test_due = True #True
                                p_train.start()

        print('Waiting  before start closing procesess')
        time.sleep(10)


        for _ in range(self.args.num_cpus):
            try:
                if processes_play[_].is_alive():
                    print('Finalizing player {}'.format(_))
                    processes_play[_].terminate()
                    time.sleep(3)
                    processes_play[_].join()
                    processes_play[_].close()
            except:
                pass

        if p_train.is_alive():
            print('Finalizing nn training process')
            p_train.terminate()
            time.sleep(5)
            p_train.join()
            p_train.close()

        for handler in logging.getLogger('').handlers:
            handler.close()
            logging.getLogger('').removeHandler(handler)
        return None

    @staticmethod
    def individual_player_session(arguments):
        args = arguments[0]
        model = arguments[1]
        mode = arguments[2]
        pipe = arguments[3]
        player_num = arguments[4]
        seed = arguments[5]

        if mode == 'BENCHMARK':
            print('benchmark')

        model.training = False
        model.model.eval()

        for param in model.model.parameters():
            param.requires_grad_(False)

        results = dict({'level': args.level})

        if not args.active_tester:
            l = 41 #if not args.hanoi else args.hanoi_max_lvl
            if True:#not args.hanoi:
                level_list = [random.choice(range(3,l)) for _ in range(args.num_equations_train)]
            else:
                level_list = [max(4,min(args.hanoi_max_lvl/2, np.random.poisson(lam=(4+args.hanoi_max_lvl
                                                                                  *(args.checkpoint_num_plays)
                                                                                  /(args.max_num_plays)/2))))
                              for _ in range(args.num_equations_train)]

        else:
            level_list = None
        #print(level_list,args.hanoi_max_lvl,(4 + args.checkpoint_num_plays),(4 + args.total_plays))
        player = Player(args, model,
                        mode=mode, name=f'player_{player_num}_{mode}',
                        pool=[],  # args.failed_pools[player_num] if (mode != 'test' or player_num <= 3) else [],
                        previous_attempts_pool=[], seed = seed)
        if args.active_tester:
            print(f'pool num {args.num_init_players}')
            if args.num_init_players < len(args.pools):
                player.pool = args.pools[args.num_init_players]
            else:
                player.pool = random.choice(args.pools)
            print(args.pools)
#
        player.play(level_list)

        results['examples'] = player.train_examples
        results['score'] = player.score
        results['sat_times'] = player.execution_times_sat
        results['sat_steps'] = player.sat_steps_taken
        results['pool_generation_time'] = player.pool_generation_time
        results['truncate'] = player.truncate
        results['player_num'] = player_num
        results['mode'] = player.mode
        results['failed_pool'] = player.failed_pool
        results['eqs_solved'] = player.eqs_solved
        results['eqs_solved_Z3'] = player.eqs_solved_z3
        results['previous_attempts_pool'] = player.previous_attempts_pool
        results['search_times'] = player.search_times
        results['num_timeouts'] = player.num_timeouts
        results['num_fails'] = player.num_fails
        results['type_of_benchmark'] = player.type_of_benchmark
        results['percentage_timeouts_solved'] = player.percentage_timeouts_solved
        results['percentage_timeouts_failed'] = player.percentage_timeouts_failed
        results['z3score']= player.z3score
        if args.learnable_smt:
            results['eqs'] = player.eqs
        results['z3mctstime_sat'] = player.z3mctstimes_sat
        results['z3mctstime_unsat'] = player.z3mctstimes_unsat
        results['action_indices'] = player.action_indices

        if mode == 'BENCHMARK' or not args.test_mode or args.active_tester:
            pipe.put(results)
            if args.num_mcts_simulations > 2:
                time.sleep(2)
            else:
                time.sleep(10)
            while not pipe.empty():
                #print(f'waiting for quue {player_num} to empty')
                if args.num_mcts_simulations > 2:
                    time.sleep(2)
                #else:
                #    time.sleep(10)
                pass
            del model
        else:
            return results



    @staticmethod
    def arcade_train(args, model_original, train_examples_history, pipe, train_mode='normal', eq_history = None, seed=None):

        results = {}
        try:
            model_original.model.to(args.train_device)
        except:
            model_original.model.to(args.train_device)

        model_original.set_parameter_device(args.train_device)
        model = NNetWrapper(args, args.train_device, training=True, seed = seed)
        if type(model_original.model) != UniformModel:
            model.model.load_state_dict(model_original.model.state_dict())
            model.optimizer.load_state_dict(model_original.optimizer.state_dict())
        model.set_optimizer_device(args.train_device)

        train_examples = []
        for examples_in_level in train_examples_history:
            for e in examples_in_level:
                train_examples.extend(e)

        print(f'len train examples: {len(train_examples)}')



        ex_eqs = train_examples

        smt_values = None

        model.model.train()

        if len(train_examples) >= args.batch_size and train_mode != 'initialize' and args.train_model: #and args.new_play_examples_available >= args.num_play_iterations_before_test_iteration:
            model.train(train_examples, None)

        model.set_parameter_device('cpu')
        model.set_optimizer_device('cpu')
        model_original.set_parameter_device('cpu')
        model_original.set_optimizer_device('cpu')

        model.model.to('cpu')
        model_original.model.to('cpu')

        results['state_dict'] = model.model.state_dict() #if len(train_examples) >= args.batch_size else 0
        results['optimizer_state_dict'] = model.optimizer.state_dict() #if len(train_examples) >= args.batch_size else 0
        results['pi_losses'] = model.pi_losses #if len(train_examples) >= args.batch_size else 0
        results['v_losses'] = model.v_losses #if len(train_examples) >= args.batch_size else 0

        pipe.put(results)
        time.sleep(2)
        del model
        while pipe.qsize() > 0:
            #
            time.sleep(2)
            pass

    def run(self):
        self.run_play_mode()
        return None

    def run_test_mode(self, level_list):
        pass

    def init_player(self, processes_play, pipes_play, mode, player_levels, player_idx):
        self.args.num_init_players +=1
        seed = self.args.num_init_players  + 1000*self.args.seed_class
        pipes_play[player_idx] = Queue()
        processes_play[player_idx] = Process(target=self.individual_player_session, args=(
            [self.args,
             self.model_play,
             mode,
             pipes_play[player_idx],
             player_idx, seed],))
        player_levels[player_idx] = self.args.level
        return processes_play, pipes_play, player_levels

    def get_score(self, score):
        return sum([sum(x) for x in score])

    def process_play_result(self, examples, sat_times, pool_generation_time, sat_steps
                    , score, level, mode, truncate, player_num, eqs_solved, eqs_solved_Z3,
                            failed_pool, previous_attempts_pool, search_times,
                            num_timeouts, num_fails, type_of_benchmark, percentage_timeouts_solved,
                            percentage_timeouts_failed, z3score, z3mctstime_sat, z3mctstime_unsat, action_indices,  eqs = None):

        total_score = self.get_score(score)
        if self.args.active_tester:
            self.plotter.eqs_solved.append(eqs_solved)
            self.plotter.eqs_solved_Z3.append(eqs_solved_Z3)
            print('\n!!!!!appending eqs_solved Z3\n!!!!!appending eqs_solved Z3\n!!!!!appending eqs_solved Z3\n!!!!!appending eqs_solved Z3\n!!!!!appending eqs_solved Z3\n!!!!!appending eqs_solved Z3\n!!!!!appending eqs_solved Z3\n!!!!!appending eqs_solved Z3', eqs_solved_Z3)

            #self.plotter.times_eqs_solved
        if truncate:
            self.args.num_truncated_pools_per_player[player_num] += 1

        if mode == 'train':
            self.args.evolution_train[self.quarters] += total_score
            self.args.iterations_train[self.quarters] += 1
            self.args.evolution_train_eq[int(self.args.checkpoint_num_plays/5)] += total_score
            self.args.iterations_train_eq[int(self.args.checkpoint_num_plays/5)] += 1
            self.train_examples_history[-1].append(examples)
            if self.args.learnable_smt:
                self.args.eq_history[-1].append(eqs)
            if len(self.train_examples_history[-1]) > self.args.num_iters_for_level_train_examples_history:
                logging.error(f"len(train_examples_history) in last level = "
                                f"{len( self.train_examples_history[-1] )} => remove the oldest trainExamples")
                logging.error(f'There are {len( self.train_examples_history[-1] )} episode data in current level')
                self.train_examples_history[-1].pop(0)
                if self.args.learnable_smt:
                    self.args.eq_history[-1].pop(0)

            self.update_logs(sat_times, sat_steps, search_times, score,
                             num_fails, num_timeouts, pool_generation_time, level, percentage_timeouts_solved,
                             percentage_timeouts_failed, z3score, z3mctstime_sat, z3mctstime_unsat, action_indices, mode = 'train')

            #if len(self.args.max_sat_time_level_train) > 0:
            if type(self.args.sat_times_train_max[- 1][1]) != str:
                max_time = max([x for z in self.args.current_level_sat_times_train for y in z for x in y])
                self.args.timeout_time = self.args.timeout_function(self.args.timeout_time,
                                                                     max_time, level = level)
                logging.error(f'New timeout update: {self.args.timeout_time}')

        if mode == 'test':
            self.args.evolution_test[self.quarters] += total_score
            self.args.iterations_test[self.quarters] += 1
            self.args.evolution_test_eq[int(self.args.checkpoint_num_plays/5)] += total_score
            self.args.iterations_test_eq[int(self.args.checkpoint_num_plays/5)] += 1

            self.update_logs(sat_times, sat_steps, search_times, score,
                             num_fails, num_timeouts, pool_generation_time, level,  percentage_timeouts_solved,
                             percentage_timeouts_failed, z3score, z3mctstime_sat, z3mctstime_unsat, action_indices, mode = 'test')

            if level == self.args.level:
                self.args.num_plays_at_current_level += 1
                self.check_level_up_conditions(total_score, level)

            self.ongoing_test = False

            if type(self.args.sat_times_test_max[- 1][1]) != str:
                max_time = max([x for z in self.args.current_level_sat_times_test for y in z for x in y])
                self.args.timeout_time = self.args.timeout_function(self.args.timeout_time, max_time, level)
                logging.error(f'New timeout update: {self.args.timeout_time}')

            if self.args.active_tester:
                # self.plotter = Plotter(self.args, filename=f'timeout_{self.args.timeout_time}')
                self.plotter.args = self.args
                #self.plotter.plot(self.args.model_name)

        if mode != 'BENCHMARK':
            self.print_logged_statistics()

        self.args.current_level_spent_time += round(time.time() - self.previous_process_play_time, 2)
        self.previous_process_play_time = time.time()

    @staticmethod
    def normalized_benchmark_score(score, sat_times_mean):
        return round(score/(np.log(15+sat_times_mean)),3)

    def update_logs(self, sat_times, sat_steps, search_times, score,
                    num_fails, num_timeouts, pool_generation_time, level, percentage_timeouts_solved,
                    percentage_timeouts_failed, z3score, z3mctstime_sat, z3mctstime_unsat, action_indices, mode='train'):
        self.args.action_indices_level.append(action_indices)

        if mode == 'train':
            self.args.search_times_train_this_level_log.append(search_times)
            self.args.current_level_sat_times_train.append(sat_times)
            self.args.sat_steps_taken_in_level_train.append(sat_steps)
            self.args.score_train_this_level.append(score)
            self.args.timeouts_train_this_level.append(num_timeouts)
            self.args.fails_train_this_level.append(num_fails)
            self.args.percentage_timeouts_solved_this_level.append(percentage_timeouts_solved)
            self.args.percentage_timeouts_failed_this_level.append(percentage_timeouts_failed)
            self.args.z3score_level_train.append(z3score)
            self.args.z3mctstime_sat_train.append(z3mctstime_sat)
            self.args.z3mctstime_unsat_train.append(z3mctstime_unsat)

            self.produce_final_logs('train', level)

        if mode == 'test':
            self.args.search_times_test_this_level_log.append(search_times)
            self.args.current_level_sat_times_test.append(sat_times)
            self.args.sat_steps_taken_in_level_test.append(sat_steps)
            self.args.score_test_this_level.append(score)
            self.args.timeouts_test_this_level.append(num_timeouts)
            self.args.fails_test_this_level.append(num_fails)
            self.args.z3score_level_test.append(z3score)
            self.args.z3mctstime_sat_test.append(z3mctstime_sat)
            self.args.z3mctstime_unsat_test.append(z3mctstime_unsat)
            self.produce_final_logs('test', level)

            self.args.pool_generation_times_this_level.append(pool_generation_time)
            self.args.timeout_times_this_level.append(self.args.timeout_time)

    def produce_final_logs(self, mode, level):
        def get_lvl_stat(listing, lvl, mode, accuracy = 2):
            combined_array = np.array([y for x in listing if len(x) >= lvl+1 for y in x[lvl]])
            if len(combined_array) > 0:
                if mode == 'mean':
                    return [lvl+1, round(combined_array.mean(), accuracy)]
                elif mode == 'std':
                    return [lvl+1, round(combined_array.std(), accuracy)]
                elif mode == 'max':
                    return [lvl+1, round(combined_array.max(), accuracy)]
                elif mode == 'sum':
                    return [lvl+1, round(combined_array.sum(), accuracy)]
            else:
                return [lvl+1, 'nan']

        def get_action_indices(lvl):
            combined_array = np.array([y for x in self.args.action_indices_level if len(x) >= lvl+1 for y in x[lvl]])
            if len(combined_array) > 0:
                return [lvl+1, list((pd.Series(combined_array).value_counts()/len(combined_array)).items())]
            else:
                return [lvl+1, 'nan']

        def initiate_container(value):
            if type(value) == list:
                return [value.copy() for _ in range(level)]
            else:
                return [value for _ in range(level)]

        self.args.action_indices = initiate_container([])


        if mode == 'train':
            self.args.sat_times_train_mean = initiate_container([])
            self.args.sat_times_train_std =  initiate_container([])
            self.args.sat_times_train_max =  initiate_container([])
            self.args.sat_steps_train_mean = initiate_container([])
            self.args.sat_steps_train_std =  initiate_container([])
            self.args.sat_steps_train_max =  initiate_container([])
            self.args.num_solved_train =  initiate_container([])
            self.args.num_failed_train =  initiate_container([])
            self.args.num_timeouts_train = initiate_container([])
            self.args.search_times_train_max =  initiate_container([])
            self.args.search_times_train_mean = initiate_container([])
            self.args.percentage_timeouts_failed_mean = initiate_container([])
            self.args.percentage_timeouts_solved_mean = initiate_container([])
            self.args.z3score_train = initiate_container([])
            self.args.z3mctstime_sat_train_final = initiate_container([])
            self.args.z3mctstime_unsat_train_final = initiate_container([])

            for lvl in range(level):
                if self.args.use_solver:
                    self.args.z3score_train[lvl] = get_lvl_stat(self.args.z3score_level_train, lvl, 'mean')
                self.args.sat_times_train_mean[lvl] = get_lvl_stat(self.args.current_level_sat_times_train, lvl, 'mean')
                self.args.sat_times_train_std[lvl] = get_lvl_stat(self.args.current_level_sat_times_train, lvl, 'std')
                self.args.sat_times_train_max[lvl] = get_lvl_stat(self.args.current_level_sat_times_train, lvl, 'max')
                self.args.sat_steps_train_mean[lvl] = get_lvl_stat(self.args.sat_steps_taken_in_level_train, lvl, 'mean')
                self.args.sat_steps_train_std[lvl] = get_lvl_stat(self.args.sat_steps_taken_in_level_train, lvl, 'std')
                self.args.sat_steps_train_max[lvl] = get_lvl_stat(self.args.sat_steps_taken_in_level_train, lvl, 'max')
                self.args.search_times_train_mean[lvl] = get_lvl_stat(self.args.search_times_train_this_level_log, lvl, 'mean', accuracy = 4)
                self.args.search_times_train_max[lvl] = get_lvl_stat(self.args.search_times_train_this_level_log, lvl, 'max', accuracy = 4)
                self.args.num_solved_train[lvl] = get_lvl_stat(self.args.score_train_this_level, lvl,
                                                              'mean')
                self.args.percentage_timeouts_failed_mean[lvl] = get_lvl_stat(self.args.percentage_timeouts_failed_this_level, lvl, 'mean')
                self.args.percentage_timeouts_solved_mean[lvl] = get_lvl_stat(self.args.percentage_timeouts_solved_this_level, lvl, 'mean')
                self.args.z3mctstime_sat_train_final[lvl] = get_lvl_stat(self.args.z3mctstime_sat_train, lvl, 'mean')
                self.args.z3mctstime_unsat_train_final[lvl] = get_lvl_stat(self.args.z3mctstime_unsat_train, lvl, 'mean')

                self.args.action_indices[lvl] = get_action_indices(lvl)

        if mode == 'test':

            self.args.sat_times_test_mean = initiate_container([])
            self.args.sat_times_test_std = initiate_container([])
            self.args.sat_times_test_max = initiate_container([])
            self.args.sat_steps_test_mean = initiate_container([])
            self.args.sat_steps_test_std = initiate_container([])
            self.args.sat_steps_test_max = initiate_container([])
            self.args.num_solved_test = initiate_container([])
            self.args.num_failed_test = initiate_container([])
            self.args.num_timeouts_test = initiate_container([])
            self.args.search_times_test = initiate_container([])
            self.args.search_times_test_max = initiate_container([])
            self.args.search_times_test_mean = initiate_container([])
            self.args.z3score_test = initiate_container([])
            self.args.z3mctstime_sat_test_final = initiate_container([])
            self.args.z3mctstime_unsat_test_final = initiate_container([])

            for lvl in range(level):
                if self.args.use_solver:
                    self.args.z3score_test[lvl] = get_lvl_stat(self.args.z3score_level_test, lvl, 'mean')
                self.args.sat_times_test_mean[lvl] = get_lvl_stat(self.args.current_level_sat_times_test, lvl, 'mean')
                self.args.sat_times_test_std[lvl] = get_lvl_stat(self.args.current_level_sat_times_test, lvl, 'std')
                self.args.sat_times_test_max[lvl] = get_lvl_stat(self.args.current_level_sat_times_test, lvl, 'max')
                self.args.sat_steps_test_mean[lvl] = get_lvl_stat(self.args.sat_steps_taken_in_level_test, lvl, 'mean')
                self.args.sat_steps_test_std[lvl] = get_lvl_stat(self.args.sat_steps_taken_in_level_test, lvl, 'std')
                self.args.sat_steps_test_max[lvl] = get_lvl_stat(self.args.sat_steps_taken_in_level_test, lvl, 'max')
                self.args.search_times_test_mean[lvl] = get_lvl_stat(self.args.search_times_test_this_level_log, lvl, 'mean', accuracy = 4)
                self.args.search_times_test_max[lvl] = get_lvl_stat(self.args.search_times_test_this_level_log, lvl, 'max', accuracy = 4)
                self.args.num_solved_test[lvl] = get_lvl_stat(self.args.score_test_this_level, lvl, 'mean') #sum([x[lvl] for x in self.args.score_test_this_level if len(x) >= lvl+1])

                self.args.z3mctstime_sat_test_final[lvl] = get_lvl_stat(self.args.z3mctstime_sat_test, lvl, 'mean')
                self.args.z3mctstime_unsat_test_final[lvl] = get_lvl_stat(self.args.z3mctstime_unsat_test, lvl, 'mean')

                self.args.action_indices[lvl] = get_action_indices(lvl)

    def process_train_result(self, state_dict, optimizer_state_dict, pi_losses, v_losses):

        if type(state_dict) != int:
            self.model_play.model.load_state_dict(state_dict)
            self.model_play.optimizer.load_state_dict(optimizer_state_dict)
            del state_dict, optimizer_state_dict
            if len(pi_losses) > 0:
                self.args.loss_log.append([self.args.level, round(pi_losses[-1],3), round(v_losses[-1],3)])
        else:
            return None

    def check_level_up_conditions(self, score, level):
        if score >= self.args.min_eqs_for_successful_test_player or self.args.test_mode:
            self.args.num_successful_plays_at_level += 1
            logging.error(f'SUCCESSFUL POOL\n')
        else:
            self.args.num_previous_test_fails += 1
            logging.error(f'FAILED POOL\n')

        if self.args.num_successful_plays_at_level >= self.args.min_successful_players_to_level_up:
            if not self.args.active_tester:
                self.level_up()
            self.args.total_time += self.args.current_level_spent_time

            level_time = self.args.current_level_spent_time
            self.args.time_log.append([self.args.level, level_time, self.args.total_time])
            self.args.initial_level_time = time.time()
            self.args.current_level_spent_time = 0.
            self.previous_process_time = time.time()

    def update_timeout_log(self):
        if len(self.args.timeout_times_this_level) > 0:
            average_timeout_time_this_level = round(np.array(self.args.timeout_times_this_level).mean(), 2)
            self.args.timeout_times_log.append([self.args.level, average_timeout_time_this_level])

    def update_pool_generation_time_log(self):
        average_pool_generation_times = round(np.array(self.args.pool_generation_times_this_level).mean(), 2)
        self.args.pool_generation_times_log.append([self.args.level, average_pool_generation_times])

    def level_up(self):
        self.update_timeout_log()
        self.update_pool_generation_time_log()

        logging.error('********\nLEVEL_UP\nLEVEL COMPLETED: {}'.format(self.args.level))
        self.args.history_successful_vs_total_plays.append([self.args.level, self.args.num_successful_plays_at_level,
                                                            self.args.num_plays_at_current_level,
                                                            self.args.num_successful_plays_at_level /
                                                            self.args.num_plays_at_current_level])
        self.update_parameters()
        self.args.level += 1

        if len(self.train_examples_history) > self.args.num_levels_in_examples_history:
            self.train_examples_history.pop(0)
        self.train_examples_history.append([])
        self.args.eq_history.append([])


    def update_parameters(self):
        self.args.update_parameters()
        self.args.timeout_steps *= 1.2
        self.args.solver_timeout += self.args.solver_timeout_increase_per_level

        current_timeout = self.args.timeout_time
        if type(self.args.sat_times_test_max[-1][1]) != str:
            self.args.timeout_time = self.args.timeout_function(self.args.timeout_time,
                                                                self.args.sat_times_test_max[-1][1], self.args.level)
            if type(self.args.sat_times_train_max[-1][1]) != str:
                self.args.timeout_time = max(self.args.timeout_function(self.args.timeout_time,
                                                                        self.args.sat_times_train_max[ - 1][1], self.args.level),
                                             self.args.timeout_time)
            logging.error(f'New timeout: {self.args.timeout_time}')
        else:
            logging.error(f'No maximum sat time found. Timeout was not updated. Something is wrong')

        self.args.timeout_time = max(current_timeout+self.args.timeout_forced_increment, self.args.timeout_time)
        self.args.num_successful_plays_at_level = 0
        self.args.num_perfect_plays = 0
        self.args.num_plays_at_current_level = 0
        self.args.timeout_times_this_level = []
        self.args.pool_generation_times_this_level = []
        self.args.num_previous_test_fails = 0

        #self.args.num_init_players = 200*self.args.level


    def print_logged_statistics(self):
        self.args.min_level -= 1
        logging.error(
            f'\n\nSUCCESSFUL/TOTAL PLAYS AT LEVEL {self.args.level}: '
            f'{self.args.num_successful_plays_at_level}/{self.args.num_plays_at_current_level}/{self.args.num_previous_test_fails}\n'
            f'SUCCESSFUL/TOTAL PLAYS HISTORY: { self.args.history_successful_vs_total_plays}\n'
            f'AVERAGE STEPS TAKEN IN SUCCESSFUL EQUATIONS TRAIN: {self.args.sat_steps_train_mean[self.args.min_level:]}\n'
            f'AVERAGE STEPS TAKEN IN SUCCESSFUL EQUATIONS TEST: {self.args.sat_steps_test_mean[self.args.min_level:]}\n'
            f'TIME LOG: {self.args.time_log }\n'
            f'Average timeout times: {self.args.timeout_times_log}\n'
            f'Average pool generation times: {self.args.pool_generation_times_log}\n'
            f'Max, std of number steps in SAT eqs train: {self.args.sat_steps_train_max[self.args.min_level:]}\n {self.args.sat_steps_train_std[self.args.min_level:]}\n'
            f'Max, std of number steps in SAT eqs test: {self.args.sat_steps_test_max[self.args.min_level:]}\n {self.args.sat_steps_test_std[self.args.min_level:]}\n'
            f'Mean, std, max time spent in SAT eqs train: {self.args.sat_times_train_mean[self.args.min_level:]}\n{self.args.sat_times_train_std[self.args.min_level:]}\n' 
            f'{self.args.sat_times_train_max[self.args.min_level:]}\n'
            f'Mean, std, max time spent in SAT eqs test: {self.args.sat_times_test_mean[self.args.min_level:]}\n{self.args.sat_times_test_std[self.args.min_level:]}\n' 
            f'{self.args.sat_times_test_max[self.args.min_level:]}\n'
            f'Search times avg (train/test): {self.args.search_times_train_mean[self.args.min_level:]} / {self.args.search_times_test_mean[self.args.min_level:]}\n'
            f'Search times max (train/test): {self.args.search_times_train_max[self.args.min_level:]} / {self.args.search_times_test_max[self.args.min_level:]}\n'
            f'Loss log: {self.args.loss_log}\n'
            f'Benchmark tests (level, score, mean sat time, max sat time, mean sat steps, max sat steps, timeouts):\n '
            f'{self.args.benchmark_test_log}\n'
            f'Action percentages: {self.args.action_indices}\n'
            f'z3 mcts search times (train: sat/unsat): {self.args.z3mctstime_sat_train_final}/{self.args.z3mctstime_unsat_train_final}\n'
            f'z3 mcts search times (test: sat/unsat):{self.args.z3mctstime_sat_test_final}/{self.args.z3mctstime_unsat_test_final}\n'
            f'New play examples available: {self.args.new_play_examples_available}\n'
            f'Num levels without benchmark: {self.args.num_levels_without_benchmark}\n'
            f'Percentage timeouts in smt searches (fail/solv): {self.args.percentage_timeouts_failed_mean[self.args.min_level:]}/{self.args.percentage_timeouts_solved_mean[self.args.min_level:]}\n'
            f'Z3scores (test/train): {self.args.z3score_test[self.args.min_level:]}/{self.args.z3score_train[self.args.min_level:]}\n'
            f'WNscores (test/train): {self.args.num_solved_test[self.args.min_level:]}/{self.args.num_solved_train[self.args.min_level:]}\n'
            f'{[round(self.args.evolution_train[i] / max(self.args.iterations_train[i], 0.001), 2) for i in range(len(self.args.evolution_train))]},{[round(self.args.evolution_test[i] / max(self.args.iterations_test[i], 0.001), 2) for i in range(len(self.args.evolution_test))]}\n'
            f'{[round(self.args.evolution_train_eq[i] / max(self.args.iterations_train_eq[i], 0.001), 2) for i in range(len(self.args.evolution_train_eq))]},{[round(self.args.evolution_test_eq[i] / max(self.args.iterations_test_eq[i], 0.001), 2) for i in range(len(self.args.evolution_test_eq))]}\n'
            f'State :: test_due {self.test_due} - ongoing-test {self.ongoing_test} - benchmark_due {self.benchmark_due} - ongoing_benchmark {self.ongoing_benchmark}\n\n')
        self.args.min_level+=1


    def save_data(self, save_model = True, args_name = 'arguments'):
        folder_name = self.args.folder_name
        #folder_name = os.path.join('\\', 'we 0.1', 'test_files', folder_name)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        if True:
            logging.error('======= SAVING DATA =========')
            if not self.args.active_tester:
                self.model_play.save_checkpoint(folder=folder_name,
                                                filename='model.pth.tar')
                self.utils.save_object('examples', self.train_examples_history, folder = folder_name)
            self.utils.save_object(args_name, self.args, folder =folder_name)
        if self.args.level % 1 == 0 and save_model:
            if not self.args.active_tester:
                self.model_play.save_checkpoint(folder=folder_name,
                                                filename=f'model_level_{self.args.level}.pth.tar')
