from string import ascii_lowercase, ascii_uppercase, punctuation

import numpy as np

# from keras.preprocessing.text import Tokenizer
#  from we.word_equation.we import WE

variables = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

numeric_vars = [chr(i) for i in range(0, 40)] #[str(i)+str(j)+str(k) for i in range(10) for j in range(10) for k in range(10)]
numeric_vars = [x for x in numeric_vars if x not in list(punctuation) + ['.', '=']]
print(len(numeric_vars))

numeric_vars_alph = [chr(i) for i in range(40, 80)]
numeric_vars_alph = [x for x in numeric_vars_alph if x not in list(punctuation) + ['.', '=']]


class Arguments(object):


    def modify_parameters_for_benchmark_test(self):
        self.num_equations_test = 20*10 if self.hanoi else 30*10
        self.num_equations_train = 20*10 if self.hanoi else 30*10
        self.max_level_benchmark = 21
        self.min_level_benchmark = 5

        self.num_cpus = 1

        self.test_mode_device = 'cpu'
        self.play_device = self.test_mode_device
        self.min_successful_players_to_level_up = 8
        self.num_mcts_multiplier = 11

        self.num_mcts_simulations_benchmark = self.num_mcts_simulations
        self.timeout_time_benchmark = 300           #self.timeout_function(level=self.max_level_benchmark,
                                                              # num_mcts=self.num_mcts_simulations_benchmark)
        self.timeout_mcts_benchmark = 1.5*self.timeout_mcts

        self.timeout_time_multiplyer = 2
        if not self.medium and not self.large:
            if self.tiny:
                self.test_mode_pool_filename = 'benchmarks/pool_lvl_6_30_size_100_standard_tiny.pth.tar' if not self.test_mode else 'benchmarks/pool_lvl_6_30_size_100_regular-ordered_tiny.pth.tar'
            elif self.small:
                self.test_mode_pool_filename = 'benchmarks/pool_lvl_6_30_size_100_standard.pth.tar' if not self.test_mode else 'benchmarks/pool_lvl_6_30_size_100_regular-ordered.pth.tar'

    def modification_when_loaded(self):
        self.checkpoint_num_plays = 96
        self.max_num_plays = 305

        return None

    def cnf_benchmark_modify_params(self):
        self.num_mcts_simulations = 200
        self.timeout_time = 4000
        self.level = 50
        self.save_model = False
        self.num_cpus = 2

    def quadratic_setting(self):

        if self.equation_sizes == 'small':
            self.SIDE_MAX_LEN = 30  # 24 #24
            num_vars = 10
            num_alph = 5
        elif self.equation_sizes == 'medium':
            self.SIDE_MAX_LEN =300
            num_vars = 26
            num_alph = 26
        self.num_play_iterations_before_test_iteration = 5
        self.timeout_forced_increment = 5

        self.level = 40 if not self.oracle else 41
        self.test_mode_pool_filename = 'benchmarks/pool_lvl_6_30_size_100_quadratic-oriented_tiny.pth.tar' if not self.test_mode else None  # 'benchmarks/pool_lvl_6_30_size_100_regular-ordered_tiny.pth.tar'

        self.ALPHABET = [x for x in ascii_lowercase][0:num_alph] if not self.large else numeric_vars_alph
        self.VARIABLES = [x for x in ascii_uppercase[::-1]] + [x for x in '0123456789']
        self.ALPHABET = self.ALPHABET[:num_alph] if not self.large else numeric_vars_alph
        self.VARIABLES = self.VARIABLES[:num_vars]  if not self.large else numeric_vars
        self.LEN_CORPUS = len(self.VARIABLES) + len(self.ALPHABET)
        self.update_symbol_index_dictionary()
        self.generation_mode = 'constant_side'#''quadratic-oriented'
        self.generation_mode = 'quadratic-oriented'
        #self.generation_mode = 'quadratic-oriented-linear'


        self.epochs = 1
        self.side_maxlen_pool = self.SIDE_MAX_LEN
        self.pool_max_initial_length = self.SIDE_MAX_LEN -  len(self.VARIABLES)
        if self.large:
            self.ALPHABET = numeric_vars_alph

    @staticmethod
    def max_steps(lvl):
        return lvl + 2*np.sqrt(lvl)



    def __init__(self, folder_name='', size='medium'):
        self.nnet_type = 'newresnet'
        self.test_mode = not True
        self.oracle = not True
        self.use_length_constraints = not True
        self.num_cpus =4
        self.test_solver = True
        self.use_oracle = True
        self.mcts_type = 'alpha0np'
        self.max_num_plays =  305
        self.learning_rate = 2e-4
        self.forbid_repetitions = False
        self.noise_param = 0.1
        self.num_mcts_simulations = 10  # 10
        self.linear_hidden_size=64
        self.num_resnet_blocks=8
        self.num_channels=128
        self.discount=0.9
        self.equation_sizes = size
        self.mcts_smt_time_max = 800
        self.log_comments = 'allowing repetitions'


        # self.smt_evaluation = True
        self.smt_solver = 'seq' #if 'CVC4' in folder_name else 'Z3'
        self.train_device = 'cpu'
        self.evaluation_after_exceeded_steps_or_time =-1
        self.use_leafs = False
        self.quadratic_mode = True
        self.large = False


        self.load_model = False
        self.few_channels = False
        self.use_random_symmetry = False
        self.skip_first_benchmark = False
        self.generate_z3_unsolvable = False
        self.z3_is_final = False
        if self.z3_is_final:
            self.generate_z3_unsolvable = True


        self.quit = False
        self.initial_time =0

        self.sokoban = False
        self.sokoban_room_size = 7
        self.maze = False
        self.maze_size = 5
        self.wordgame  = False
        self.SOLUTION = 'aaabbb'
        self.wordgame_del = True
        self.wordgame_tree = False
        self.wordgame_size = len(self.SOLUTION) + 4
        self.wordgame_alphabet = list(set(self.SOLUTION).union({'c', 'd'}))
        if self.wordgame_tree:
            self.wordgame_alphabet = list(set(self.SOLUTION).union({'b'}))

        self.wordgame_alphabet.sort()
        self.cube = False
        self.hanoi = False
        self.num_discs = 5
        self.hanoi_max_lvl =25 #2**self.num_discs+1#int(2**self.num_discs/2) -1

        self.sat = False
        self.sat_min_n = 4
        self.sat_max_n = 100
        self.sat_max_alpha = 10  # alpha = m/n (m = num clauses)
        self.sat_min_alpha = 2
        self.sat_max_k = 4  # k = length of clause
        self.sat_min_k = 4
        self.alpha_resolutions = 10
        self.sat_num_layers = 2
        self.sat_hidden_dim =  32

        self.nobound = False

        self.values01=False

        self.rec_mcts = False
        self.mcts_classic = False
        self.mcgs_persistence = 0.999# 0.99975
        self.mcgs_type = 2

        self.recurrent_blocks1 = False
        self.recurrent_blocks2 = False
        self.recurrent_blocks = False

        self.maxpool_input = False
        self.chunk_h = 8
        self.chunk_w = 30

        self.values_01 = False
        self.max_aggregation = False
        self.affine_transf= True

        self.chunk_size = 10

        self.seconds_per_step = 18 if not self.sokoban else 15
        if self.wordgame:
            self.seconds_per_step = 9
        if self.sat:
            self.seconds_per_step = 25
        if self.large:
            self.seconds_per_step = 20

        self.augment_examples = False
        self.automatic_compress = False
        self.medium = False
        self.small = True
        self.tiny = True
        self.very_tiny = True
        self.size_type = 'tiny'
        self.timeout_time = 50 if not self.large else 50 #200
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


        if not self.large:
            self.VARIABLES = [x for x in ascii_uppercase[::-1]] + [x for x in '0123456789']
        else:
            self.VARIABLES = numeric_vars #[x for x in ascii_uppercase[::-1]] + [x for x in '0123']

        if self.small:
            self.SIDE_MAX_LEN = 30
            num_vars = 8 if not self.oracle else 24
            num_alph = 5 if not self.oracle else 24
            self.pool_max_initial_length = 10
            self.test_mode_pool_filename = 'benchmarks/pool_lvl_6_30_size_100_standard.pth.tar' if not self.test_mode else None # 'benchmarks/pool_lvl_6_30_size_100_regular-ordered.pth.tar'
            self.num_play_iterations_before_test_iteration = 5
            if self.tiny:
                self.SIDE_MAX_LEN = 30
                num_vars = 8 if not self.oracle else 50
                num_alph = 5 if not self.oracle else 50
                self.pool_max_initial_length = 4
                self.test_mode_pool_filename = 'benchmarks/pool_lvl_6_30_size_100_standard_tiny.pth.tar' if not self.test_mode else None # 'benchmarks/pool_lvl_6_30_size_100_regular-ordered_tiny.pth.tar'
                self.num_play_iterations_before_test_iteration = 5
                if self.very_tiny:
                    self.timeout_forced_increment = 3
                    self.level = 4 if not self.oracle else 21





        self.side_maxlen_pool = self.SIDE_MAX_LEN
        # self.pool_max_initial_length = self.pool_max_initial_length if self.nnet_type != 'supernet' else 10 # not self.large else 100
        self.pool_max_initial_constants = 1

        self.min_eqs_for_successful_test_player = 9
        self.num_levels_in_examples_history = 1
        self.num_iters_for_level_train_examples_history = 5
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
            print('= in variables remove' )
            #self.VARIABLES = [x for x in self.VARIABLES if x not in '=']
        print(len(self.VARIABLES), self.VARIABLES)
        #assert len(self.VARIABLES) == num_vars
        self.ALPHABET = [x for x in ascii_lowercase][0:num_alph] if not self.large else numeric_vars_alph

        self.pgnn_feature_dim = 7
        self.pgnn_input_dim = self.pgnn_feature_dim
        self.pgnn_output_dim = self.pgnn_input_dim
        self.pgnn_hidden_dim = self.pgnn_input_dim
        self.pgnn_num_layers = 2
        self.pgnn_len_side = self.SIDE_MAX_LEN
        self.pgnn_max_num_nodes =  1+2*self.SIDE_MAX_LEN + len(self.VARIABLES) + len(self.ALPHABET) + 1 +1
        self.pgnn_true_output_dim = self.pgnn_max_num_nodes * int(int(np.log2(self.pgnn_max_num_nodes))**2)
        self.pgnn_approximate = 1

        self.timeout_steps_std_multiplyer = 4
        self.timeout_time_multiplyer = 1.2
        self.timeout_steps_small_std_multiplyer = 2

        self.empty_symbol = '.'
        self.SPECIAL_CHARS = ['=', self.empty_symbol]
        self.LEN_CORPUS = len(self.ALPHABET) + len(self.VARIABLES)

        self.play_device = 'cpu'
        self.batch_size = 1
        self.batch_size = self.batch_size if not self.recurrent_blocks else self.batch_size
        self.batch_size = 1 if self.sat else self.batch_size
        #self.batch_size = 1 if self.nnet_type in ['attention', 'graphwenet'] else self.batch_size

        self.epochs = 1
        #

        self.num_equations_train = 10
        self.num_equations_test = 10

        self.num_init_players = 0

        self.maxlenOfQueue = 200000
        self.cpuct = 1
        self.pb_c_base = 20000
        self.pb_c_init = 1.25
        self.temp = 1
        self.test_mode_level_list = range(3, 21) if not self.wordgame else range(2,10)

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

        #we = WE(self,seed=1)
        self.num_actions =8# we.moves.get_action_size() if not self.sat else 1


        if not self.check_LP:
            print('WARNING: Not checking LP')

        if self.test_mode:
            self.modify_parameters_for_benchmark_test()
        self.min_level = self.level

        self.modes = None
        self.pools = None
        self.active_tester = False
        #self.forced_settings()
        self.timeout_time =  180 if not self.cube and not self.wordgame and not self.hanoi else 300
        if self.maze or self.sokoban or self.wordgame:
            self.timeout_time=40

        if self.large:
            self.ALPHABET = numeric_vars_alph

        self.change_parameters_by_folder_name('we_alpha0_disc90_smaller_seed2')

    def timeout_function(self, current_time, max_time, level = None):
        # max_time = round(2*mean_time)
        return 180 if not self.cube and not self.wordgame and not self.hanoi else 300


    def timeout_function2(self, time, max_time, level = None):
        return 40#180 if not self.cube and not self.wordgame and not self.hanoi else 300


        #(level/10)*level*1.1*num_mcts #(np.log(level)/80) * (np.log(level)*3) * num_mcts  # (level/12) * (level*1.2) * num_mcts / 10

    def timeout_time_test_function(self, level, num_mcts):
        return self.timeout_function(level, num_mcts)

    def timeout_time_benchmark_function(self, level, num_mcts):
        return self.timeout_function(level, num_mcts)

    def num_mcts_function(self, level):
        return self.num_mcts_simulations # int(round(self.num_mcts_multiplier*np.log(level)))
            #int(round(10*np.log(level)))

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
        self.failed_pools = self.num_cpus*[[]]
        self.previous_attempts_pool = self.num_cpus * [[]]
        self.test_failed_pool = []
        self.test_previous_attempts_pool = []

        self.num_previous_test_fails = 0

        self.new_play_examples_available = 0
        self.num_levels_without_benchmark = 0

    def update_parameters(self):
        self.update_num_mcts_simulations()
        self.update_num_constants_in_pool()

    def update_num_mcts_simulations(self):
        self.num_mcts_simulations = self.num_mcts_function(self.level)
            # int(round(self.num_mcts_multiplier * self.level))

    def update_symbol_index_dictionary(self):
        if self.nnet_type in ['resnet', 'recnewnet', 'newresnet', 'GIN', 'pgnn', 'resnet1d', 'resnet_double',
                              'satnet', 'graphwenet', 'attention', 'lstmpe','hanoinet', 'wordgamenet']:
            self.LEN_CORPUS += 1
            word_index = {}
            word_index.update({x: i for i,x in enumerate(self.VARIABLES)})
            word_index.update({x: i+len(self.VARIABLES) for i,x in enumerate(self.ALPHABET)})
            word_index.update({'.': self.LEN_CORPUS-1})
        if self.nnet_type in ['lstmpe']:
            word_index.update({'=': self.LEN_CORPUS})
        symbol_indices = word_index
        #for x in symbol_indices.keys():
        #    symbol_indices[x] -= 1
        symbol_indices_reverse = {i: x for i, x in enumerate(symbol_indices.keys())}
        self.symbol_indices = symbol_indices
        self.symbol_indices_reverse = symbol_indices_reverse
        self.alphabet_indices ={x: i for i,x in enumerate(self.ALPHABET)}
        self.variable_indices ={x: i for i,x in enumerate(self.VARIABLES)}


    def update_num_constants_in_pool(self):
        if len(self.ALPHABET) <= self.pool_max_initial_constants:
            self.pool_max_initial_constants = len(self.ALPHABET)-1

    def change_parameters_by_folder_name(self, folder_name):

        def get_bool(param):
            split = self.folder_name.split(param)
            return bool(int(split[0][-1]))

        def get_int(param):
            split = self.folder_name.split(param)
            x = split[0][-1]
            if x == 'F':
                return 10
            elif x == 'L':
                self.learnable_smt = True
                return 10
            else:
                return float(split[0][-1])

        #if os.path.exists(folder_name):
        #    self.load_model = True
        #else:
        #    self.load_model = False
        folder_name = folder_name.split('\\')[-1]

        self.folder_name = folder_name
        self.use_random_symmetry = False
        self.use_normal_forms = True if not self.oracle else False
        self.solver_proportion = 0
        self.train_model = True
        self.episode_timeout_method = 'time'
        self.use_noise_in_test = True

        if 'recblock1' in folder_name:
            self.recurrent_blocks1 = True
            self.recurrent_blocks2 = False
            self.recurrent_blocks = True

        if 'recblock2' in folder_name:
            print('with recblock')
            self.recurrent_blocks1 = False
            self.recurrent_blocks2 = True
            self.recurrent_blocks = True

                #self.use_random_symmetry = True if get_bool('symmetry') else False
        #self.use_normal_forms = True if get_bool('normal_form') else False
        #self.solver_proportion = get_int('solver')/10
        if 'z3isfinal' in folder_name:
            self.z3_is_final = get_bool('z3isfinal')
        if 'unsolvz3gen' in folder_name:
            self.generate_z3_unsolvable = get_bool('unsolvz3gen')
        if 'fewchannels' in folder_name:
            self.few_channels = get_bool('fewchannels')

        self.quadratic_mode = True
        if self.quadratic_mode:
            self.quadratic_setting()

        if 'values01' in folder_name:
            self.values_01 = get_bool('values01')
        if 'maxaggregation' in folder_name:
            self.max_aggregation = get_bool('maxaggregation')

        if 'LP' in folder_name:
            self.check_LP = get_bool('LP')
        if 'normalforms' in folder_name:
            self.use_normal_forms = get_bool('normalforms')


        self.seed_class =  int(folder_name.split('seed')[1])



        #we = WE(self, seed=1)
        self.num_actions = 8 # we.moves.get_action_size() if  not self.sat else 1
        #self.timeout_time = 15 + self.seconds_per_step * self.level
        if self.maze or self.sokoban or self.wordgame:
            self.timeout = 60
        if 'affine' in folder_name:
            self.affine_transf = get_bool('affine')

        if 'norepetitions' in folder_name:
            self.forbid_repetitions = get_bool('norepetitions')


        self.seconds_per_step = self.seconds_per_step if 'reckblock' not in folder_name else 9
        self.seconds_per_step = self.seconds_per_step if 'recnew' not in folder_name else 9

        self.batch_size = 16
        #self.batch_size = 1 if self.nnet_type in ['attention', 'graphwenet'] else self.batch_size

        if 'maxpool' in folder_name:
            self.maxpool_input = True
        else:
            self.maxpool_input = False

        if 'pers' in folder_name:
            self.mcgs_persistence = str(folder_name.split('pers')[1])
            self.mcgs_persistence  = int(self.mcgs_persistence)/(10**len(self.mcgs_persistence))
            print(f'mcgs_persistence: {self.mcgs_persistence}')

        if 'load' in folder_name:
            self.load_model = get_bool('load')


        self.value_timed_out_simulation=-1

        if 'simoval' in folder_name:
            self.value_timed_out_simulation = -int(folder_name.split('simoval')[0][-1])
            print(f'Value timed out simulation: {self.value_timed_out_simulation}')

        #if self.nnet_type == 'recnewnet':


        if 'sqrt' in folder_name:
            self.mcgs_fun = np.sqrt
        elif 'log' in folder_name:
            self.mcgs_fun = np.log1p
        else:
            self.mcgs_fun = 'id'

        self.usePE = False if 'noPE' in folder_name else True
        self.oracle = False
        self.use_normal_forms = True if not  self.oracle else False


        print(f'PARAMETER CONFIGURATION: \n'
              f'nnet type: {self.nnet_type}\n',
              f'folder_name: {folder_name}\n'
              f'use_random_symmetry: {self.use_random_symmetry}\n'
              f'solver_proportion: {self.solver_proportion}\n'
              f'train model: {self.train_model}\n'
              f'value_timed_out_episode: {self.evaluation_after_exceeded_steps_or_time}\n'
              # f'episode_timeout_method: {self.episode_timeout_method}\n'
              f'use_normal_forms: {self.use_normal_forms}\n'
              f'check_LP: {self.check_LP}\n'
              f'generate z3 unsolvable: {self.generate_z3_unsolvable}\n'
              f'z3 is final: {self.z3_is_final}\n'
              f'seconds per step: {self.seconds_per_step}\n'
              f'discount: {self.discount}')
        return None

