import random
import numpy as np
import time
from multiprocessing import Process, Queue
from .player import Player
from .neural_net_wrapper import NNetWrapper
from .utils import Utils, seed_everything
import os
import matplotlib.pyplot as plt
from .neural_net_models.uniform_model import UniformModel
from PIL import PngImagePlugin
import logging

plt.interactive(True)


class Arcade(object):
    """
    This class implements the training process for the MCTS_nn algorithm as described in the paper.
    The main function is "run workers". This creates a certain number of workers which will, in parallel, successively
    attempt to solve randomly generated equations. The data collected while doing so is used to train (in parallel)
    the neural network. THe number of workers for us is given by the argument args.num_cpus
    """
    def __init__(self, args, load, name=''):
        self.init_log(args.folder_name)
        logging.info('\n\n\nNEW ARCADE ---- load: {} ---- {}'.format(load, args.folder_name))
        self.args = args

        seed_everything(self.args.seed_class - 1)
        self.utils = Utils(args)

        if self.args.load_model:
            self.args = self.utils.load_object('arguments')
            self.args.load_model = True
            self.train_examples_history = [[]]  # Data collected and used to train the network
        else:
            self.train_examples_history = [[]]
            self.args.num_train = 0

        logging.info(vars(self.args))
        logging.info(f'NUM CPUs:{self.args.num_cpus}')

        self.active_players = self.args.num_cpus * [False]  # keep track of which workers are active
        self.test_due = True  # from time to time we want one of the workers to perform a test on a fixed dataset
        self.ongoing_test = False

        if not self.args.load_model:  # Initialize some lists for keeping track of the training evolution
            self.args.train_scores_current_iteration = []
            self.args.train_scores_per_iteration = []
            self.args.test_scores = []
            self.args.checkpoint_train_intervals = []

        self.utils.load_nnet(device='cpu', training=True, load=self.args.load_model, folder=self.args.folder_name,
                             filename=f'model.pth.tar')

    def init_log(self, folder_name, mode='train'):
        """
        Auxiliary method for logging
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=folder_name + f'/log_{mode}.log')
        console = logging.StreamHandler()
        logger = logging.getLogger(PngImagePlugin.__name__)
        logger.setLevel(logging.INFO)  # tame the "STREAM" debug messages

        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    def init_dict_of_process_and_queues(self):
        """
        This creates a dictionary of processes and queues (from python's multiprocessing library), one for worker.
        Each worker has a mode, either train (standard worker) or test.
        """
        queues = {worker_num: Queue() for worker_num in range(self.args.num_cpus)}
        mode = 'train' if not self.args.active_tester else 'test'

        modes = self.args.num_cpus * [mode]
        modes[-1] = 'test'

        processes = {}
        for worker_num in range(self.args.num_cpus):
            self.args.num_init_players += 1
            seed = self.args.num_init_players + 1000 * self.args.seed_class
            processes.update({worker_num: Process(
                target=self.individual_player_session,
                args=([self.args, self.model_play, modes[worker_num], queues[worker_num], worker_num, seed],))})
            self.active_players[worker_num] = True
            processes[worker_num].start()
        return processes, queues, modes

    def run_workers(self):

        if self.args.load_model:
            self.args.modification_when_loaded()
            self.test_due = False  # True
            self.train_examples_history = [[]]
        else:
            self.initiate_loop = True

        iteration = 0
        seed = self.args.num_init_players + 10000 * self.args.seed_class

        parent_conn_train = Queue()
        train_mode = 'normal'
        p_train = Process(target=self.train_model,
                          args=(self.args, self.model_play, self.train_examples_history, parent_conn_train,
                                train_mode, seed))
        p_train.start()
        nnet_train_done = False
        
        while self.args.num_pools_processed < self.args.max_num_pools:

            if self.args.active_tester and self.args.num_pools_since_last_nn_train < \
                len([x for x in self.args.pools if type(x) != str]):
                break

            folder_name = self.args.folder_name
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)

            iteration += 1

            if self.initiate_loop:
                self.initiate_loop = False
                processes, queues, modes, player_levels = self.init_dict_of_process_and_queues()

            if self.args.active_tester or not self.args.test_mode:
                for _ in range(self.args.num_cpus):

                    if not queues[_].empty():

                        print(f'Hola player {_}')
                        self.active_players[_] = False
                        print(f'Hola again player {_}')

                        result_play = queues[_].get(block=True)
                        self.process_play_result(**result_play)

                        queues[_].close()
                        queues[_].join()
                        processes[_].join()
                        processes[_].close()

                        if not self.active_players[_]:
                            if not self.args.active_tester:
                                if (self.test_due and 
                                not self.ongoing_test and nnet_train_done) or self.args.active_tester:
                                    mode = 'test'
                                    self.test_due = False
                                    self.ongoing_test = True
                                    nnet_train_done = False
                                else:
                                    mode = 'train'  # modes[_]

                                processes, queues, player_levels = self.init_player(processes, queues,
                                                                                             mode,
                                                                                             player_levels, _)
                                self.active_players[_] = True
                                processes[_].start()
                                self.args.num_pools_processed += 1
                                self.test_due = True
                                self.save_data()
                                print('num_pools_since_last_nn_train', self.args.num_pools_since_last_nn_train)
                                self.args.num_pools_since_last_nn_train += 1

                            else:
                                self.args.num_pools_processed += 1
                                self.save_data()
                                print('num_pools_since_last_nn_train', self.args.num_pools_since_last_nn_train)
                                self.args.num_pools_since_last_nn_train += 1

                        time.sleep(2.)

                if not self.args.test_mode:
                    if self.args.num_pools_since_last_nn_train > self.args.num_pools_for_nn_train:
                        if (not parent_conn_train.empty()) or (iteration == 1 and self.args.load_model):
                            print('Getting train data')
                            result_train = parent_conn_train.get(block=True)

                            self.process_train_result(**result_train)
                            p_train.join()
                            p_train.close()
                            parent_conn_train.close()
                            parent_conn_train.join_thread()

                            nnet_train_done = True
                            self.args.num_train += 1

                            self.model_play.save_checkpoint(folder=self.args.folder_name,
                                                            filename=f'model_train_{self.args.num_train}.pth.tar')

                            parent_conn_train = Queue()

                            seed = self.args.num_init_players + 10000 * self.args.seed_class

                            p_train = Process(target=self.train_model,
                                              args=(self.args, self.model_play, self.train_examples_history,
                                                    parent_conn_train, 'normal', seed))

                            self.args.num_pools_since_last_nn_train = 0
                            p_train.start()

        print('Waiting  before start closing procesess')
        time.sleep(10)

        for _ in range(self.args.num_cpus):
            try:
                if processes[_].is_alive():
                    print('Finalizing player {}'.format(_))
                    processes[_].terminate()
                    time.sleep(3)
                    processes[_].join()
                    processes[_].close()
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

        player = Player(args, model, mode=mode,
                        name=f'player_{player_num}_{mode}',
                        pool=None, seed=seed)
        if args.active_tester:
            print(f'pool num {args.num_init_players}')
            #if args.num_init_players < len(args.pools):
            player.pool = args.pools[args.num_init_players]
            #else:
            #    player.pool = random.choice(args.pools)
            #print(args.pools)
        #
        player.play()

        results['examples'] = player.train_examples
        results['score'] = player.score
        results['mode'] = player.mode

        pipe.put(results)
        time.sleep(2)
        while pipe.qsize() > 0:
            time.sleep(2)

    @staticmethod
    def train_model(args, model_original, train_examples_history, pipe, train_mode='normal', seed=None):
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
                train_examples) >= args.batch_size and train_mode != 'initialize' and args.train_model:  # and args.num_pools_since_last_nn_train >= args.num_pools_for_nn_train:
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


    def init_player(self, processes, queues, mode, player_levels, player_idx):
        self.args.num_init_players += 1
        seed = self.args.num_init_players + 1000 * self.args.seed_class
        queues[player_idx] = Queue()
        processes[player_idx] = Process(target=self.individual_player_session, args=(
        [self.args, self.model_play, mode, queues[player_idx], player_idx, seed],))
        player_levels[player_idx] = self.args.level
        return processes, queues, player_levels

    def get_score(self, score):
        return sum([sum(x) for x in score])

    def process_play_result(self, examples, score, mode):

        total_score = self.get_score(score)
        if mode == 'train':
            self.args.train_scores_current_iteration.append(total_score)
            self.train_examples_history[-1].append(examples)
            if len(self.train_examples_history[-1]) > self.args.num_iters_for_level_train_examples_history:
                logging.info(f"len(train_examples_history) in last level = "
                             f"{len(self.train_examples_history[-1])} => remove the oldest trainExamples")
                logging.info(f'There are {len(self.train_examples_history[-1])} episode data in current level')
                self.train_examples_history[-1].pop(0)

        if mode == 'test':
            self.args.test_scores.append(total_score)

            self.ongoing_test = False

        self.print_logged_statistics()

    def process_train_result(self, state_dict, optimizer_state_dict, v_losses):
        self.model_play.model.load_state_dict(state_dict)
        self.model_play.optimizer.load_state_dict(optimizer_state_dict)
        if len(v_losses)>0:
            self.args.loss_log.append( round(v_losses[-1], 6))
        self.args.checkpoint_train_intervals.append(self.args.num_pools_processed)
        train_score =np.array(self.args.train_scores_current_iteration).mean()
        self.args.train_scores_per_iteration.append(train_score)
        self.args.train_scores_current_iteration = []

    def print_logged_statistics(self):
        logging.info(
            f'Loss log: {self.args.loss_log}\n'
            f'New play examples available: {self.args.num_pools_since_last_nn_train}\n'
            f'test performance log {self.args.test_scores}\n'
            f'training performance log: {self.args.train_scores_per_iteration}\n'
            f'State :: test_due {self.test_due} - ongoing-test {self.ongoing_test}\n\n')

    def save_data(self):
        folder_name = self.args.folder_name
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        logging.info('======= SAVING DATA =========')
        if not self.args.active_tester:
            self.model_play.save_checkpoint(folder=folder_name, filename='model.pth.tar')
            self.utils.save_object('examples', self.train_examples_history, folder=folder_name)
        self.utils.save_object('arguments', self.args, folder=folder_name)

