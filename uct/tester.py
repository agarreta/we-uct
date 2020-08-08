
from .neural_net_wrapper import NNetWrapper
from .utils import Utils
from .arcade import Arcade
from .arguments import Arguments
import os
import seaborn as sbn
from we.plotter import Plotter
from pickle import Pickler, Unpickler
from .neural_net_models.uniform_model import UniformModel
from .utils import seed_everything


seed_everything(10)
sbn.set(style="white", palette='muted')


class Tester(object):
    def __init__(self,
                 test_name='',
                 model_folder=None,
                 algorithm_name=None,
                 use_oracle=False,
                 test_solver=False,
                 benchmark_filename=None,
                 num_mcts_sims = None,
                 timeout = None,
                 use_constant_model=False,
                 model = None):

        #self.model = None  !!!!

        self.plotter = Plotter(Arguments(), folder_name='evaluation_' + test_name,
                                      filename= test_name)  # TODO: plotter is currently only used to store results. Can be simplified

        args = Arguments()
        args.test_mode = False  # TODO: deprecate
        args.change_parameters_by_folder_name(model_folder)
        args.num_mcts_simulations = num_mcts_sims
        args.timeout_time = timeout

        args.level = 30  # TODO: why this=?
        args.active_tester = True
        args.modes = args.num_cpus * ['test']
        args.test_mode = True
        args.num_init_players = 0  # TODO: already by default?
        args.save_model = True

        pool = self.load_pool(benchmark_filename)
        pools = self.split_pool(pool)
        args.pools = pools

        args.test_solver = test_solver
        args.use_oracle = use_oracle

        args.folder_name = model_folder + '/evaluation/' + algorithm_name
        args.timeout = timeout
        args.num_mcts_simulations = num_mcts_sims
        self.arcade = Arcade(args, load=False)

        self.arcade.plotter = self.plotter
        self.utils = Utils(args)

        self.arcade.args.model_name = algorithm_name
        self.arcade.args.num_init_players = 0
        if not use_constant_model:
            self.load_model(model_filename=model_folder, model_name=model)
        else:
            print('using untrained model')
            self.arcade.model_play = NNetWrapper(args, 'cpu', training=False, seed=args.seed_class)
            self.arcade.model_play.model = UniformModel(args, args.num_actions)

    def wn_test(self):
        self.arcade.run_play_mode()
        self.plotter = self.arcade.plotter
        self.save_plotter()

    def split_pool(self, pool):
        #print([eq.w for eq in self.pool])
        num_chunks = 10 #int(len(pool) / 10)
        pools = []
        for i in range(num_chunks):
            pools.append(pool[10 * i: 10 * (i + 1)])
        print(f'NUMBER OF EQUATION POOLS FOR CURRENT BENCHMARK: {len(pools)}')
        return pools

    # TODO: Import the next two  functions from utils
    def load_pool(self, pool_filename):
        with open(pool_filename, "rb") as f:
            return Unpickler(f).load()

    def save_plotter(self):
        filename = os.path.join(self.plotter.folder_name, 'plotter.pth.tar')
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.plotter)

    def load_model(self, model_filename=None, model_name=None):
        if model_filename is not None:
            print('loading', model_filename)
            model_name = 'model.pth.tar' if model_name is None else model_name
            self.arcade.model_play = self.utils.load_nnet(device = 'cpu', training = False, load = True, folder = model_filename,
                                                          filename=model_name)


    #def z3_pool_evaluation(self, timeout, verbose=False):
    #    for i, eq in enumerate(self.pool):
    #        self.z3converter = WeToSmtZ3(self.args.VARIABLES, self.args.ALPHABET, timeout=timeout)
    #        eval = self.z3converter.eval(eq.w)
    #        self.z3evaluations=\
    #            self.z3evaluations.append(pd.DataFrame([[eq.level, timeout, float(eval)]], columns= self.z3evaluations.columns))#, ignore_index=True)
    #        if verbose: print(i, eq.level, eval, eq.w)
#
    #def z3_print(self):
    #    plt.title('Z3 evaluations')
    #    sbn.lineplot(data=self.z3evaluations, x='level', y='result', hue='timeout')  # , hue = 'type')
    #    plt.savefig('Z3 evaluation')
    #    plt.close()
#
    #def z3_test(self):
    #    for timeout in [1000, 5000, 10000, 30000]:
    #        self.z3_pool_evaluation(timeout, verbose=True)
    #        print(self.z3evaluations)
    #        self.z3_print()

    #def a(self):
    #    processes_play, queues_play, _ = self.utils.init_dict_of_process_and_queues(target_fun=self.arcade.individual_player_session(),
    #                                                                                model=self.model,
    #                                                                                modes=8*['test'])

