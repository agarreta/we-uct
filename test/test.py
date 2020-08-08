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
import math


def seed_everything(seed):
    # https://www.kaggle.com/hmendonca/fold1h4r3-arcenetb4-2-256px-rcic-lb-0-9759 cells 45-50
    #print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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


def simple_evaluate(pool_filepath, model_folder, model_filename, solver, seed=14, size='small', smt_max_time=None):
    seed_everything(seed)
    def uniformize_levels(pool):

        npool = []
        level_slots = [range(math.floor(10 + 1.6 * i), math.floor(10 + 1.6 * (i + 1))) for i in range(0, 9)] if '20_5_3' in pool_filepath else [range(math.floor(3 + 1. * i), math.floor(3 + 1. * (i + 1))) for i in range(0, 9)]
        num_slots = [0 for _ in range(0, 9)]
        while len(npool) < 200:
            a = int(np.argmin([x for x in num_slots]))
            l = level_slots[a]
            for x in [y for y in pool if y not in npool]:
                if x.level in l:
                    npool.append(x)
                    num_slots[a] += 1
                    break
        assert len(pool) >= 200
        pool = npool[:200]
        print('TEST LEVELS: ', num_slots)
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
    if '00' in pool_filepath:
        new_pool=[]
        for x in pool:
            ar = Arguments()
            e = WordEquation(ar)
            e.w = x
            new_pool.append(e)
        pool=new_pool

    print([eq.w for eq in pool])
    args = Arguments(size)
    if '05' in pool_filepath:
        new_pool=[]
        for x in pool:
            x.ell = [0] + x.ell
            new_pool.append(x)
        pool=new_pool
        args.use_length_constraints=True
    args.folder_name = model_folder
    args.smt_solver = solver
    args.seed_class = seed
    if smt_max_time is not None:
        args.mcts_smt_time_max = smt_max_time
    if solver is None:
        args.use_oracle = False
        args.test_solver = False
        args.oracle = False
        args.use_normal_forms = True
    else:
        args.use_oracle = True
        args.oracle = True
        args.use_normal_forms = False
        args.check_LP = False
        args.test_solver=True
    if '05_track' in pool_filepath:
        args.use_length_constraints = True
    args.pool_name = pool_filepath
    args.pool_name_load = pool_filepath
    # assert  args.equation_sizes != 'medium' or 'track' in pool_filepath
    solve_pool(args, pool, model_folder, model_filename, mode='test', seed=seed, num_cpus=args.num_cpus)


if __name__ == "__main__":

    seeds = range(16, 19) #[2,3,1]
    algorithm_names = [f'we_uct_oracleZ3_disc90_track{1}_seed{seed}' for seed in seeds]
    folders = [
        f'we_alpha0_disc90_smaller_seed{seed}' for seed in seeds
    ]  # TODO: set the args in <<<<<<<<<<<the name in the arguments file

    if True:
        for t in [0]:
            print(algorithm_names)

            for solver in ['TRAU','CVC4']:
                simple_evaluate(f'benchmarks/0{5}_track/transformed',
                                'v56', 'uniform', solver=solver, smt_max_time=800, seed=20)

        assert False
        for t in [1,2,3]:
            print(algorithm_names)
            for j in [77,52,0]:
                simple_evaluate(f'benchmarks/0{t}_track/transformed',
                                'v57', f'model_train_{j}.pth.tarf', solver=None, smt_max_time=800, seed=20)

            if False:
                simple_evaluate(f'benchmarks/pool_20_5_3.pth.tar',
                                    'v57',  f'model_train_{t}.pth.tar', solver=None, smt_max_time=800, seed=20)
                simple_evaluate(f'benchmarks/pool_150_14_10.pth.tar',
                                'v57', f'model_train_{t}.pth.tar', solver=None, smt_max_time=800, seed=20)

    assert False

    simple_evaluate(f'benchmarks/pool_20_5_3.pth.tar',
                    'v56', 'model_train_130.pth.tar', solver=None, smt_max_time=800, seed=20)
    assert False
    solvers = ['Z3', 'seq', 'CVC4', 'TRAU']
    for solver in ['dumb']:
        for seed in [14]:
            for track in [1,2,3]:
                simple_evaluate(f'benchmarks/0{track}_track/transformed',
                                f'we_alpha0_disc90_smaller_seed{seed}',
                                'uniform', solver=solver, smt_max_time=800, seed=seed)
            simple_evaluate(f'benchmarks/pool_20_5_3.pth.tar', f'we_alpha0_disc90_smaller_seed{seed}',
                            'uniform', solver=solver, smt_max_time=800, seed=seed)
    assert False
    if True:
        #for i in
        seed = 14
        for tr in [0]:
            for track in [3]:
                simple_evaluate(f'benchmarks/0{track}_track/transformed',
                                'v55', f'model_train_{tr}.pth.tar', solver=None, smt_max_time=800, seed=seed)
            simple_evaluate(f'benchmarks/pool_20_5_3.pth.tar',
                            'v55',f'model_train_{tr}.pth.tar', solver=None, smt_max_time=800, seed=seed)




    #seed=14
    #simple_evaluate(f'benchmarks/pool_20_5_3.pth.tar', f'we_alpha0_disc90_smaller_seed{seed}',
    #                'uniform', solver='CVC4', smt_max_time=800, seed=seed)
    if False:
        for seed in [14,15,16]:

            for model_name in [f'model_train_{i}.pth.tar' for i in range(1, 14)[::3]]:
                simple_evaluate(f'benchmarks/pool_20_5_3.pth.tar', 'v55', model_name, solver=None, seed=seed)

            for model_name in [f'model_train_{i}.pth.tar' for i in range(1, 24)[::3]]:
                pass


