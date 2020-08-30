
from uct.arguments import Arguments
import os
import gc
import logging
import numpy as np
import torch
import random
from pickle import Unpickler
from uct.arcade_test import solve_pool
from uct.arcade_train import Arcade
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


def train(folder):

    args = Arguments()
    args.change_parameters_by_folder_name(folder)
    init_log(folder, 'train')

    arcade = Arcade(args, load=args.load_model)
    model = arcade.utils.load_nnet(device='cpu',
                                   training=True,
                                   load=args.load_model,
                                   folder=folder)
    arcade.model_play = model
    arcade.model_train = model
    arcade.run()





def simple_evaluate(pool_filepath, model_folder, model_filename, solver, seed=14, size='small', smt_max_time=None):
    seed_everything(seed)
    def uniformize_levels(pool):

        npool = []
        level_slots = [range(math.floor(10 + 1.6 * i), math.floor(10 + 1.6 * (i + 1))) for i in range(0, 9)]
        num_slots = [0 for _ in range(0, 9)]
        while len(npool) < 200:
            a = int(np.argmin([x for x in num_slots]))
            l = level_slots[a]
            for x in [y for y in pool if y not in npool]:
                if x.level in l:
                    npool.append(x)
                    num_slots[a] += 1
                    break
        assert len(pool) == 200
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
    print([eq.w for eq in pool])
    args = Arguments(size)
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

    args.pool_name = pool_filepath
    args.pool_name_load = pool_filepath
    solve_pool(args, pool, model_folder, model_filename, mode='test', seed=seed, num_cpus=args.num_cpus)


if __name__ == "__main__":

    seeds = range(17, 17) #[2,3,1]
    algorithm_names = [f'we_uct_oracleZ3_disc90_track{1}_seed{seed}' for seed in seeds]
    folders = [
        f'we_alpha0_disc90_smaller_seed{seed}' for seed in seeds
    ]  # TODO: set the args in the name in the arguments file

    simple_evaluate(f'benchmarks/pool_150_14_10.pth.tar',
                     'v57', 'model_train_0.pth.tar', solver=None, smt_max_time=800, seed=20)
    simple_evaluate(f'benchmarks/pool_20_5_3.pth.tar',
                    'v56', 'model_train_130.pth.tar', solver=None, smt_max_time=800, seed=20)
