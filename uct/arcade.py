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
        logging.info('\n\n\nNEW ARCADE ---- load: {} ---- {}'.format(load, args.folder_name))
        logging.info(f'Test mode: {args.test_mode}')
        self.args = args
        self.num_train = 0

        seed_everything(self.args.seed_class - 1)

        self.utils = Utils(args)

        if self.args.load_model:
            self.args = self.utils.load_object('arguments')
            self.args.load_model = True
            self.train_examples_history = [[]]

        else:
            self.train_examples_history = [[]]

        logging.info(vars(self.args))
        logging.info(f'NUM CPUs:{self.args.num_cpus}')

        self.active_players = self.args.num_cpus * [False]
        self.received_play_results = self.args.num_cpus * [False]
        self.active_training = False
        self.received_train_results = False
        self.test_due = True  # if not self.args.load_model else True
        self.ongoing_test = False

        if not self.args.load_model:
            self.args.iterations_test_eq = [0]
            self.args.evolution_train_eq = []
            self.args.evolution_train_eq_full = []
            self.args.training_performance_log = []
            self.args.checkpoint_num_plays_being_tested = 0
            self.args.evolution_scores = []

            self.args.evolution_scores_full = []
            self.args.checkpoint_train_intervals = []

        self.active_solved_test = 0
        self.active_total_eqs = 0
        self.best_model = 'model_plays_0.pth.tar'
        self.best_score = 0
        self.active_model_being_tested = 'model_plays_0.pth.tar'
        self.name = name

        self.utils.load_nnet(device='cpu', training=True, load=False, folder=self.args.folder_name,
                             filename=f'model.pth.tar')

    def init_log(self, folder_name, mode='train'):
        import logging
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # TODO: What is this for?
        # logging.getLogger("tensorflow").setLevel(logging.WARNING)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=folder_name + f'/log_{mode}.log')
        console = logging.StreamHandler()
        from PIL import PngImagePlugin
        logger = logging.getLogger(PngImagePlugin.__name__)
        logger.setLevel(logging.INFO)  # tame the "STREAM" debug messages

        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    def init_dict_of_process_and_queus(self):
        pipes_play = {_: Queue() for _ in range(self.args.num_cpus)}
        m = 'train' if not self.args.active_tester else 'test'
        modes = self.args.num_cpus * [m]  # + 1 * ['test']
        modes[-1] = 'test'

        processes_play = {}
        for _ in range(self.args.num_cpus):
            self.args.num_init_players += 1
            seed = self.args.num_init_players + 1000 * self.args.seed_class
            processes_play.update({_: Process(
                target=self.individual_player_session,
                args=([self.args, self.model_play, modes[_], pipes_play[_], _, seed],))})
            self.active_players[_] = True
            processes_play[_].start()
        player_levels = {_: self.args.level for _ in range(self.args.num_cpus)}
        if self.args.active_tester:
            self.args.total_plays += 1
        return processes_play, pipes_play, modes, player_levels

    def run_play_mode(self, pools=None):

        if self.args.load_model:
            self.args.modification_when_loaded()
            self.benchmark_due = False
            self.initiate_loop = not self.benchmark_due
            self.ongoing_benchmark = False
            self.test_due = False  # True
            self.args.skip_first_benchmark = False
            self.train_examples_history = [[]]
        else:
            self.initiate_loop = True

        iteration = 0
        num_evol_model = 0
        #self.model_play = self.load
        seed = self.args.num_init_players + 10000 * self.args.seed_class

        parent_conn_train = Queue()
        train_mode = 'normal'
        p_train = Process(target=self.arcade_train,
                          args=(self.args, self.model_play, self.train_examples_history, parent_conn_train,
                                train_mode, seed))
        p_train.start()
        nnet_train_done = False
        while (self.args.checkpoint_num_plays < self.args.max_num_plays and
               not self.args.active_tester) or \
                (self.args.active_tester and self.args.checkpoint_num_plays < self.args.max_num_plays and
                 self.args.new_play_examples_available < len([x for x in self.args.pools if type(x) != str])):

            #if self.active_total_eqs >= self.args.num_iters_for_level_train_examples_history and not self.args.active_tester:
            #    current_score = 100 * self.active_solved_test / self.active_total_eqs
            #    self.args.evolution_scores.append(current_score)
            #    self.args.evolution_scores_full.append(current_score)
##
            #    self.active_total_eqs = 0
            #    self.active_solved_test = 0

            folder_name = self.args.folder_name
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)

            iteration += 1
            #
            if self.initiate_loop:
                self.initiate_loop = False
                processes_play, pipes_play, modes, player_levels = self.init_dict_of_process_and_queus()

            if self.args.active_tester or (not self.args.test_mode):
                for _ in range(self.args.num_cpus):

                    if not pipes_play[_].empty():

                        print(f'Hola player {_}')
                        self.active_players[_] = False
                        print(f'Hola again player {_}')

                        result_play = pipes_play[_].get(block=True)
                        self.args.num_finished_play_session_per_player[_] += 1
                        self.process_play_result(**result_play)

                        pipes_play[_].close()
                        pipes_play[_].join_thread()
                        processes_play[_].join()
                        processes_play[_].close()

                        if not self.active_players[_]:
                            if not self.args.active_tester:
                                if (self.test_due and (
                                not self.ongoing_test) and nnet_train_done) or self.args.active_tester:
                                    mode = 'test'
                                    self.test_due = False
                                    self.ongoing_test = True
                                    nnet_train_done = False
                                else:
                                    mode = 'train'  # modes[_]

                                processes_play, pipes_play, player_levels = self.init_player(processes_play, pipes_play,
                                                                                             mode,
                                                                                             player_levels, _)
                                self.active_players[_] = True
                                processes_play[_].start()
                                self.args.total_plays += 1
                                self.args.checkpoint_num_plays += 1
                                self.test_due = True
                                self.save_data()
                                print('new_play_examples_available', self.args.new_play_examples_available)
                                self.args.new_play_examples_available += 1

                            else:
                                self.args.total_plays += 1
                                self.args.checkpoint_num_plays += 1
                                self.save_data()
                                print('new_play_examples_available', self.args.new_play_examples_available)
                                self.args.new_play_examples_available += 1

                        time.sleep(2.)

                if not self.args.test_mode:
                    if self.args.new_play_examples_available > self.args.num_play_iterations_before_test_iteration:  # or total_plays == 0:
                        if (not parent_conn_train.empty()) or (iteration == 1 and self.args.load_model):
                            print('Getting train results')
                            result_train = parent_conn_train.get(block=True)

                            self.process_train_result(**result_train)
                            p_train.join()
                            p_train.close()
                            parent_conn_train.close()
                            parent_conn_train.join_thread()

                            nnet_train_done = True
                            self.num_train += 1

                            self.model_play.save_checkpoint(folder=self.args.folder_name,
                                                            filename=f'model_train_{self.num_train}.pth.tar')

                            parent_conn_train = Queue()

                            seed = self.args.num_init_players + 10000 * self.args.seed_class

                            p_train = Process(target=self.arcade_train,
                                              args=(self.args, self.model_play, self.train_examples_history,
                                                    parent_conn_train, 'normal', seed))

                            self.args.new_play_examples_available = 0
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
        model.training = False
        model.model.eval()

        for param in model.model.parameters():
            param.requires_grad_(False)

        results = dict()

        if not args.active_tester:
            level_list = [random.choice(range(3, 41)) for _ in range(args.num_equations_train)]
        else:
            level_list = None
        player = Player(args, model,
                        mode=mode, name=f'player_{player_num}_{mode}',
                        pool=None,  # args.failed_pools[player_num] if (mode != 'test' or player_num <= 3) else [],
                        seed=seed)
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
        results['mode'] = player.mode


        pipe.put(results)
        time.sleep(2)
        while pipe.qsize() > 0:
            time.sleep(2)

    @staticmethod
    def arcade_train(args, model_original, train_examples_history, pipe, train_mode='normal', seed=None):
        results = {}
        try:
            model_original.model.to(args.train_device)
        except:
            model_original.model.to(args.train_device)

        model_original.set_parameter_device(args.train_device)
        model = NNetWrapper(args, args.train_device, training=True, seed=seed)
        if type(model_original.model) != UniformModel:
            model.model.load_state_dict(model_original.model.state_dict())
            model.optimizer.load_state_dict(model_original.optimizer.state_dict())
        model.set_optimizer_device(args.train_device)

        train_examples = []
        for examples_in_level in train_examples_history:
            for e in examples_in_level:
                train_examples.extend(e)

        print(f'len train examples: {len(train_examples)}')

        model.model.train()

        if len(
                train_examples) >= args.batch_size and train_mode != 'initialize' and args.train_model:  # and args.new_play_examples_available >= args.num_play_iterations_before_test_iteration:
            model.train(train_examples)

        model.set_parameter_device('cpu')
        model.set_optimizer_device('cpu')
        model_original.set_parameter_device('cpu')
        model_original.set_optimizer_device('cpu')

        model.model.to('cpu')
        model_original.model.to('cpu')

        results['state_dict'] = model.model.state_dict()  # if len(train_examples) >= args.batch_size else 0
        results[
            'optimizer_state_dict'] = model.optimizer.state_dict()  # if len(train_examples) >= args.batch_size else 0
        results['v_losses'] = model.v_losses  # if len(train_examples) >= args.batch_size else 0

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

    def init_player(self, processes_play, pipes_play, mode, player_levels, player_idx):
        self.args.num_init_players += 1
        seed = self.args.num_init_players + 1000 * self.args.seed_class
        pipes_play[player_idx] = Queue()
        processes_play[player_idx] = Process(target=self.individual_player_session, args=(
        [self.args, self.model_play, mode, pipes_play[player_idx], player_idx, seed],))
        player_levels[player_idx] = self.args.level
        return processes_play, pipes_play, player_levels

    def get_score(self, score):
        return sum([sum(x) for x in score])

    def process_play_result(self, examples, score, mode):

        total_score = self.get_score(score)
        if mode == 'train':
            self.args.evolution_train_eq.append(total_score)
            self.train_examples_history[-1].append(examples)
            if len(self.train_examples_history[-1]) > self.args.num_iters_for_level_train_examples_history:
                logging.info(f"len(train_examples_history) in last level = "
                             f"{len(self.train_examples_history[-1])} => remove the oldest trainExamples")
                logging.info(f'There are {len(self.train_examples_history[-1])} episode data in current level')
                self.train_examples_history[-1].pop(0)
                if self.args.learnable_smt:
                    self.args.eq_history[-1].pop(0)

        if mode == 'test':
            #self.active_solved_test += total_score
            #self.active_total_eqs += self.args.num_iters_for_level_train_examples_history
            self.args.evolution_scores.append(total_score)
            self.args.iterations_test_eq.append(1)
            self.ongoing_test = False

        self.print_logged_statistics()

    def process_train_result(self, state_dict, optimizer_state_dict, v_losses):
        self.model_play.model.load_state_dict(state_dict)
        self.model_play.optimizer.load_state_dict(optimizer_state_dict)
        #del state_dict, optimizer_state_dict
        if len(v_losses)>0: #TODO: how could == 0 happen?
            self.args.loss_log.append( round(v_losses[-1], 6))
        self.args.checkpoint_train_intervals.append(self.args.checkpoint_num_plays)
        train_score =np.array(self.args.evolution_train_eq).mean()
        self.args.training_performance_log.append(train_score)
        self.args.evolution_train_eq = []

    def print_logged_statistics(self):
        self.args.min_level -= 1
        logging.info(
            f'Loss log: {self.args.loss_log}\n'
            f'New play examples available: {self.args.new_play_examples_available}\n'
            f'test performance log {self.args.evolution_scores}\n'
            f'training performance log: {self.args.training_performance_log}\n'
            f'State :: test_due {self.test_due} - ongoing-test {self.ongoing_test}\n\n')
        self.args.min_level += 1

    def save_data(self):
        folder_name = self.args.folder_name
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        logging.info('======= SAVING DATA =========')
        if not self.args.active_tester:
            self.model_play.save_checkpoint(folder=folder_name, filename='model.pth.tar')
            self.utils.save_object('examples', self.train_examples_history, folder=folder_name)
        self.utils.save_object('arguments', self.args, folder=folder_name)

