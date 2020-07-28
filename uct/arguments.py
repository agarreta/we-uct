from string import ascii_lowercase, ascii_uppercase, punctuation

import numpy as np
from math import ceil, floor

# from keras.preprocessing.text import Tokenizer
#  from we.word_equation.we import WE

variables = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

numeric_vars = [chr(i) for i in
                range(0, 40)]  # [str(i)+str(j)+str(k) for i in range(10) for j in range(10) for k in range(10)]
numeric_vars = [x for x in numeric_vars if x not in list(punctuation) + ['.', '=']]
print(len(numeric_vars))

numeric_vars_alph = [chr(i) for i in range(40, 80)]
numeric_vars_alph = [x for x in numeric_vars_alph if x not in list(punctuation) + ['.', '=']]


class Arguments(object):

    def modification_when_loaded(self):
        self.checkpoint_num_plays = self.checkpoint_train_intervals[-1]
        self.max_num_plays = 100050
        self.new_play_examples_available = 0
        self.local_execution_times_train_history = []
        self.local_execution_times_test_history = []
        self.sat_steps_taken_train_history = []
        self.sat_steps_taken_test_history = []
        # self.train_level_slots = [range(floor(3 +1.5*i), ceil(3+1.5*(i+1))) for i in range(0,9)] # [range(5 +2*i, 5+2*(i+1)) for i in range(0,9)]
        self.num_cpus = 8

    def quadratic_setting(self):

        self.SIDE_MAX_LEN = 150  # 24 #24
        num_vars = 14
        num_alph = 10

        self.level = 40 if not self.oracle else 41
        self.test_mode_pool_filename = 'benchmarks/pool_lvl_6_30_size_100_quadratic-oriented_tiny.pth.tar' if not self.test_mode else None  # 'benchmarks/pool_lvl_6_30_size_100_regular-ordered_tiny.pth.tar'

        self.ALPHABET = [x for x in ascii_lowercase][0:num_alph] if not self.large else numeric_vars_alph
        self.VARIABLES = [x for x in ascii_uppercase[::]] + [x for x in '0123456789'] + list(ascii_lowercase[6:])
        self.ALPHABET = self.ALPHABET[:num_alph] if not self.large else numeric_vars_alph
        self.VARIABLES = self.VARIABLES[:num_vars] if not self.large else numeric_vars
        self.LEN_CORPUS = len(self.VARIABLES) + len(self.ALPHABET)
        self.update_symbol_index_dictionary()
        self.generation_mode = 'constant_side'  # ''quadratic-oriented'
        self.generation_mode = 'quadratic-oriented'
        # self.generation_mode = 'quadratic-oriented-linear'

        self.side_maxlen_pool = self.SIDE_MAX_LEN
        self.pool_max_initial_length = self.SIDE_MAX_LEN - len(self.VARIABLES)
        if self.large:
            self.ALPHABET = numeric_vars_alph

    def __init__(self, folder_name=''):
        self.nnet_type = 'newresnet'
        self.test_mode = not True
        self.oracle = not True
        self.use_length_constraints = not True
        self.num_cpus = 8
        self.test_solver = False
        self.use_oracle = False
        self.mcts_type = 'alpha0np'
        self.mcts_value_type = 'normal'  # 'Qvalues'
        self.max_num_plays = 200050
        self.learning_rate = 1e-3
        self.forbid_repetitions = False
        self.noise_param = 0.1
        self.num_mcts_simulations = 50
        self.linear_hidden_size = 64
        self.num_resnet_blocks = 2
        self.num_channels = 64
        self.discount = 0.9 if self.mcts_value_type == 'Qvalues' else 0.9
        self.mcts_smt_time_max = 800
        self.checkpoint_num_plays = 0
        self.total_plays = 0
        # self.pool_name_load = 'benchmarks/30_10_5_v2.pth.tar' #'benchmarks/pool_30_10_5_v2.pth.tar'
        self.pool_name_load = 'benchmarks/pool_150_14_10.pth.tar'  # 'benchmarks/pool_150_15_10.pth.tar'  #'benchmarks/pool_30_10_5_v2.pth.tar'
        self.num_levels_in_examples_history = 1
        self.num_iters_for_level_train_examples_history = 100
        self.num_play_iterations_before_test_iteration = self.num_iters_for_level_train_examples_history
        self.test_pool_length = 100  # self.num_iters_for_level_train_examples_history
        self.batch_size = 64
        self.log_comments = ''
        # self.smt_evaluation = True
        self.smt_solver = 'Z3'  # if 'CVC4' in folder_name else 'Z3'
        self.train_device = 'cuda:0'
        self.evaluation_after_exceeded_steps_or_time = -1
        self.use_leafs = True
        self.quadratic_mode = True
        self.equation_sizes = 'small'
        self.sat_value = 1
        self.unknown_value = 0
        self.unsat_value = -1
        self.epochs = 1
        self.load_model = False
        self.active_tester_time = 30
        self.test_time = 30
        self.train_time = 30
        self.using_attention = False
        self.bound = True
        self.train_level_slots = [range(floor(3 + 1. * i), ceil(3 + 1. * (i + 1))) for i in range(0, 9)]  #
        self.test_level_slots = [range(floor(3 + 1. * i), ceil(3 + 1. * (i + 1))) for i in range(0,
                                                                                                 9)]  ##[range(ceil(5 +1.7*i), ceil(5+1.7*(i+1))) for i in range(0,9)] # [range(5 +2*i, 5+2*(i+1)) for i in range(0,9)]
        self.format_mode = 'cuts'
        self.NNET_SIDE_MAX_LEN = 24

        self.num_train = 0
        self.num_collisions_test = 0
        self.few_channels = False
        self.use_random_symmetry = False
        self.skip_first_benchmark = False
        self.generate_z3_unsolvable = False
        self.z3_is_final = False
        if self.z3_is_final:
            self.generate_z3_unsolvable = True

        self.quit = False
        self.initial_time = 0

        self.cube = False
        self.hanoi = False

        self.nobound = False

        self.values01 = False

        self.rec_mcts = False
        self.mcts_classic = False
        self.mcgs_type = 2

        self.recurrent_blocks = False

        self.maxpool_input = False
        self.chunk_h = 8
        self.chunk_w = 30

        self.values_01 = False
        self.max_aggregation = False
        self.affine_transf = True

        self.chunk_size = 10
        self.large = False if not self.oracle else True

        self.augment_examples = False
        self.automatic_compress = False
        self.medium = False
        self.small = True
        self.tiny = True
        self.very_tiny = True
        self.size_type = 'tiny'
        self.timeout_time = 50 if not self.large else 50  # 200
        self.timeout_forced_increment = 10
        self.constant_side = False if self.large else False
        self.cnf_benchmark = False
        self.generation_mode = 'standard'

        self.episode_timeout_method = 'time'
        self.use_seed = True
        self.learnable_smt = False
        self.use_noise_in_test = False

        if self.test_mode:
            self.num_cpus = 1

        self.value_mode = 'value-classic'  # solver, entropy, classic, value-entropy, value-classic
        self.use_solver = False
        if self.value_mode in ['solver', 'entropy', 'value-entropy', 'value-classic']:
            self.use_value_nn_head = True
            self.episode_timeout_method = 'time'
            self.allowance_factor_play = 1.6
            self.value_timed_out_simulation = 0 if not self.values01 else 0.5
        else:
            self.use_value_nn_head = True

        self.timeout_value_smt = 0.
        self.solver_timeout = 100
        self.solver_timeout_increase_per_level = 30
        self.solver_proportion = 0.5
        self.check_z3_conversion = False
        if not self.use_value_nn_head:
            assert self.solver_proportion == 1.

        self.side_maxlen_pool = self.SIDE_MAX_LEN
        # self.pool_max_initial_length = self.pool_max_initial_length if self.nnet_type != 'supernet' else 10 # not self.large else 100
        self.pool_max_initial_constants = 1

        self.min_eqs_for_successful_test_player = 9

        self.frequency_of_benchmark_test = 100
        self.mode = 'train'
        self.type_of_benchmark = None

        self.dynamic_timeout = False
        self.perform_benchmark = True
        self.save_model = True
        self.train_model = True
        self.load_level = True
        self.use_dynamic_negative_reward = False
        self.use_test_player = False
        self.symmetrize_examples = False
        self.use_true_symmetrization = False
        self.num_random_symmetrizations = 10
        self.check_LP = True if not self.oracle else False
        self.use_steps_for_timeout = True
        self.truncate = True
        self.use_clears = False

        self.VARIABLES = self.VARIABLES[:num_vars] if not self.large else numeric_vars
        if '=' in self.VARIABLES:
            print('= in variables remove')
            # self.VARIABLES = [x for x in self.VARIABLES if x not in '=']
        print(len(self.VARIABLES), self.VARIABLES)
        # assert len(self.VARIABLES) == num_vars
        self.ALPHABET = [x for x in ascii_lowercase][0:num_alph] if not self.large else numeric_vars_alph

        self.empty_symbol = '.'
        self.SPECIAL_CHARS = ['=', self.empty_symbol]
        self.LEN_CORPUS = len(self.ALPHABET) + len(self.VARIABLES)

        self.play_device = 'cpu'
        # self.batch_size = 1 if self.nnet_type in ['attention', 'graphwenet'] else self.batch_size

        #

        self.num_equations_train = 10
        self.num_equations_test = 10

        self.num_init_players = 0

        self.maxlenOfQueue = 200000
        self.cpuct = 1
        self.pb_c_base = 20000
        self.pb_c_init = 1.25
        self.temp = 1
        self.test_mode_level_list = range(3, 21) if not self.wordgame else range(2, 10)

        self.folder_name = 'evaluations'
        self.examples_file_name = 'examples.pth.tar'
        self.test_log_file_name = 'test_log.pth.tar'
        self.model_file_name = 'model.pth.tar'
        self.time_log_file_name = 'time_log.pth.tar'

        self.timeout_steps = 10
        self.timeout_steps_small = 7
        # not uised?
        # self.score_to_level_up = 0.875
        self.level_up_threshold = 0.79
        self.allowance_factor = 1.1
        self.test_max_steps = 25
        self.num_linear_lc_blocks = 5
        self.lc_output_size = 256
        self.num_linear_value_blocks = 2
        self.num_linear_pi_blocks = 2
        self.linear_output_size = 256
        self.num_lstm_hidden_layers = 3
        self.lstm_hidden_size = 512
        self.dropout = 0.2

        self.num_total_plays = 0

        self.min_successful_players_to_level_up = 1

        self.failed_num_mcts_multiplier = 1
        self.num_mcts_multiplier = 300

        self.timeout_mcts = 0.1

        self.update_symbol_index_dictionary()
        self.update_parameters()
        self.init_parameters()

        # we = WE(self,seed=1)
        self.num_actions = 8  # we.moves.get_action_size() if not self.sat else 1

        if not self.check_LP:
            print('WARNING: Not checking LP')

        if self.test_mode:
            self.modify_parameters_for_benchmark_test()
        self.min_level = self.level

        self.modes = None
        self.pools = None
        self.active_tester = False
        self.timeout_time = 180 if not self.cube and not self.wordgame and not self.hanoi else 300
        if self.maze or self.sokoban or self.wordgame:
            self.timeout_time = 40

        if self.large:
            self.ALPHABET = numeric_vars_alph

    def init_parameters(self):
        self.time_log = []
        self.avg_sat_steps_taken_per_level_test = []
        self.avg_sat_steps_taken_per_level_train = []

        self.history_successful_vs_total_plays = []
        self.pool_generation_times_this_level = []
        self.pool_generation_times_log = []
        self.timeout_times_this_level = []
        self.timeout_times_log = []

        self.max_mcts_time_log = []
        self.loss_log = []
        self.num_finished_play_session_per_player = self.num_cpus * [0]
        self.num_failed_play_session_per_player = self.num_cpus * [0]
        self.num_truncated_pools_per_player = self.num_cpus * [0]
        self.num_successful_plays_at_level = 0
        self.num_plays_at_current_level = 0

        self.total_successful_episodes_log = [[self.level, 0, 0]]
        self.ous_session_total_time = 0
        self.current_level_spent_time = 0
        self.total_time = 0
        self.benchmark_test_log = []
        self.failed_pools = self.num_cpus * [[]]
        self.previous_attempts_pool = self.num_cpus * [[]]
        self.test_failed_pool = []
        self.test_previous_attempts_pool = []

        self.num_previous_test_fails = 0

        self.new_play_examples_available = 0
        self.num_levels_without_benchmark = 0

    def update_symbol_index_dictionary(self):
        if self.nnet_type in ['resnet', 'recnewnet', 'newresnet', 'GIN', 'pgnn', 'resnet1d', 'resnet_double',
                              'satnet', 'graphwenet', 'attention', 'lstmpe', 'hanoinet', 'wordgamenet']:
            self.LEN_CORPUS += 1
            word_index = {}
            word_index.update({x: i for i, x in enumerate(self.VARIABLES)})
            word_index.update({x: i + len(self.VARIABLES) for i, x in enumerate(self.ALPHABET)})
            word_index.update({'.': self.LEN_CORPUS - 1})
        symbol_indices = word_index
        symbol_indices_reverse = {i: x for i, x in enumerate(symbol_indices.keys())}
        self.symbol_indices = symbol_indices
        self.symbol_indices_reverse = symbol_indices_reverse
        self.alphabet_indices = {x: i for i, x in enumerate(self.ALPHABET)}
        self.variable_indices = {x: i for i, x in enumerate(self.VARIABLES)}

    def change_parameters_by_folder_name(self, folder_name):
        folder_name = folder_name.split('\\')[-1]
        self.folder_name = folder_name
        self.use_normal_forms = True if not self.oracle else False
        self.train_model = True
        self.episode_timeout_method = 'time'
        self.quadratic_mode = True
        self.quadratic_setting()
        self.seed_class = int(folder_name.split('seed')[1])
        print('SEED', self.seed_class)
        self.num_actions = 8
        self.value_timed_out_simulation = -1
        self.use_normal_forms = False if self.use_oracle else True

        print(f'PARAMETER CONFIGURATION: \n'
              f'nnet type: {self.nnet_type}\n',
              f'folder_name: {folder_name}\n'
              f'use_random_symmetry: {self.use_random_symmetry}\n'
              f'solver_proportion: {self.solver_proportion}\n'
              f'train model: {self.train_model}\n'
              f'value_timed_out_episode: {self.evaluation_after_exceeded_steps_or_time}\n'
              f'use_normal_forms: {self.use_normal_forms}\n'
              f'check_LP: {self.check_LP}\n'
              f'generate z3 unsolvable: {self.generate_z3_unsolvable}\n'
              f'z3 is final: {self.z3_is_final}\n'
              f'seconds per step: {self.seconds_per_step}\n'
              f'discount: {self.discount}')

