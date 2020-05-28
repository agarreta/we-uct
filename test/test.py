# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:06:03 2019

@author: garre
"""

from we import *
from we.arguments import Arguments
from we.tester import Tester
import os
import gc
import logging
import numpy as np
import torch
import random
from pickle import Unpickler
from we.arcade_single import solve_pool


# TODO: import from utils
def init_log(folder_name, mode='train'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=folder_name + f'/log_{mode}.log')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


def train(folder,
          use_oracle=False,
          num_mcts_simulations=80,
          timeout=30):

    # TODO: import from utils
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    random.seed(0)

    args = Arguments()
    args.use_oracle = use_oracle
    args.num_mcts_simulations = num_mcts_simulations
    args.timeout_time = timeout

    args.change_parameters_by_folder_name(folder)
    init_log(folder, 'train')

    arcade = Arcade(args, load=args.load_model)
    model = arcade.utils.load_nnet(device='cpu',  # TODO: what does this argument do?
                                   training=True,
                                   load=args.load_model,
                                   folder=folder)  # TODO: do I need to initialize the model like this?
    arcade.model_play = model
    arcade.model_train = model
    arcade.run()

    # TODO: needed?
    for handler in logging.getLogger('').handlers:
        handler.close()
        logging.getLogger('').removeHandler(handler)
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(folder,
             name,
             use_oracle=False,
             test_solver=False,
             ran=None,
             test_set=None,
             num_mcts_simulations=80,
             timeout=30,
             use_constant_model=False):
    """
    use_oracle:  TODO: write doc
    """

    algorithm_names = [f'{10 * i}_{name}' for i in ran]
    folders = [folder for _ in ran]
    print(folders)
    print(algorithm_names)
    benchmarks = len(ran) * [test_set]
    mcts = [num_mcts_simulations for _ in ran]
    timeouts = [timeout for _ in ran]
    models = [f'model_plays_{10 * j}.pth.tar' for j in ran]

    print(models)
    for i in range(len(folders)):
        tester = Tester(test_name=algorithm_names[i],
                        model_folder=folders[i],
                        algorithm_name=algorithm_names[i],
                        use_oracle=use_oracle,
                        test_solver=test_solver,
                        benchmark_filename=benchmarks[i],
                        num_mcts_sims=mcts[i],
                        timeout=timeouts[i],
                        model=models[i],
                        use_constant_model=use_constant_model)
        tester.wn_test()


def simple_evaluate(pool_filepath, model_folder, model_filename, solver, seed=14, size='small', smt_max_time=800):

    def uniformize_levels(pool):
        max_level=25
        max_size_pool=200
        npool = []
        num_eqs_per_lvl_npool = [0 for _ in range(4,max_level+1)]
        num_eqs_per_lvl_oldpool = [0 for _ in range(4,max_level+1)]
        for e in pool:
            num_eqs_per_lvl_oldpool[e.level - 4] += 1
        for l in range(4,max_level+1):
            for e in pool:
                if len(npool) > max_size_pool:
                    break
                if e.level == l and num_eqs_per_lvl_npool[l - 4] < 4 + min(1, l % 4):
                    npool.append(e)
                    num_eqs_per_lvl_npool[l - 4] += 1
            if len(npool) > max_size_pool:
                break
        print(num_eqs_per_lvl_oldpool)
        print(num_eqs_per_lvl_npool)
        assert len(npool) >= max_size_pool
        pool = npool
        return pool

    with open(pool_filepath, 'rb+') as f:
        pool = Unpickler(f).load()
    if 'pool' in pool_filepath:
        pool = uniformize_levels(pool)
    if '02_track' in pool_filepath:
        new_pool = []
        for x in pool:
            if x.w not in [y.w for y in new_pool]:
                new_pool.append(x)
        pool = new_pool
    print([eq.w for eq in pool])
    args = Arguments(size)
    args.folder_name = model_folder
    args.smt_solver = solver
    args.seed_class = seed
    args.mcts_smt_time_max = smt_max_time
    if solver is None:
        args.use_oracle = False
        args.test_solver = False
    args.pool_name = pool_filepath
    # assert  args.equation_sizes != 'medium' or 'track' in pool_filepath
    solve_pool(args, pool, model_folder, model_filename, mode='test', seed=seed, num_cpus=args.num_cpus)


if __name__ == "__main__":

    seeds = range(14, 17) #[2,3,1]
    algorithm_names = [f'we_uct_oracleZ3_disc90_track{1}_seed{seed}' for seed in seeds]
    folders = [
        f'we_alpha0_disc90_smaller_seed{seed}' for seed in seeds
    ]  # TODO: set the args in the name in the arguments file

    print(algorithm_names)
    #
    if False:
        for seed in range(14,17):
            for model_name in [f'model_plays_{50*i}.pth.tar' for i in range(7)]:
                simple_evaluate(f'benchmarks/pool_30_10_5.pth.tar', folders[seed-14], model_name,
                                solver=None, seed=seed)

    #train(folders[i], use_oracle=False)

    #simple_evaluate(f'benchmarks/pool_30_10_5.pth.tar', folders[0], 'uniform', 'sloth')
    #for i in
    for solver in ['Z3', 'seq', 'CVC4']:
        for track in [1,2,3]:
            simple_evaluate(f'benchmarks/0{track}_track/transformed', folders[0], 'uniform', solver=solver)

    # for solver in ['Z3', 'seq',  'woorpje', 'CVC4']:
    #     simple_evaluate(f'benchmarks/pool_50_20_10.pth.tar', folders[0], 'uniform', solver=solver)

    assert False
    for seed in [14,15,16]:
        for model_name in [f'model_plays_{i}.pth.tar' for i in [0, 300, 600]]:
            simple_evaluate(f'benchmarks/pool_30_10_5.pth.tar', folders[seed-14], model_name, solver=None, seed=seed)
