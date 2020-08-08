# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:28:18 2019

@author: garre
"""
from string import ascii_uppercase, ascii_lowercase

from .player import Player
from .utils import Utils, seed_everything
import matplotlib.pyplot as plt
from we.neural_net_wrapper import NNetWrapper
from we.player import Player
from we.neural_net_models.uniform_model import  UniformModel
import os
from multiprocessing import Pool
from math import ceil
import datetime
plt.interactive(True)
import numpy as np


t=800
timeouts={
    'Z3': {
        '01_track':t,
        '02_track': t,
        '03_track':t,
        'pool_20_5_3.pth.tar':t
    },
    'seq': {
        '01_track': t,
        '02_track': t,
        '03_track': t,
        'pool_20_5_3.pth.tar': t
    },
    'CVC4': {
        '01_track':t,
        '02_track':t,
        '03_track':t,
        'pool_20_5_3.pth.tar':t
    },
    'dumb': {
        '01_track': t,
        '02_track': t,
        '03_track': t,
        'pool_20_5_3.pth.tar': t
    },
    'TRAU': {
        '01_track':3500,
        '02_track': 3500,
        '03_track': 3500,
        'pool_20_5_3.pth.tar': 3500
    },
    'woorpje': {
        '01_track': 3000,
        '02_track': 3000,
        '03_track': 3000,
        'pool_20_5_3.pth.tar': 3000
    }
}
def solve_pool(args, pool, model_folder, model_filename, mode, seed, num_cpus=1):
    seed_everything(seed)
    if num_cpus == 1:
        results = individual_player_session((args, pool, model_folder, model_filename, mode, seed))
        results['sat_steps'] = results['sat_steps_taken']
    else:
        p = Pool(num_cpus)
        chunk = ceil(len(pool)/num_cpus)
        eq_sub_pools = [pool[i*chunk:(i+1)*chunk] for i in range(num_cpus) if len(pool[i*chunk:(i+1)*chunk])>0]
        play_args = []
        sub_results = p.map(t, [(args, sub_pool, model_folder, model_filename, mode, seed) for sub_pool in eq_sub_pools])
        results = {}
        for i, sr in enumerate(sub_results):
            if i == 0:
                results['eqs_solved'] = sr['eqs_solved']
                results['sat_times'] = sr['sat_times']
                results['eqs_solved_Z3'] = sr['eqs_solved_Z3']
                results['sat_steps'] = sr['sat_steps_taken']
            else:
                results['eqs_solved'] += sr['eqs_solved']
                results['sat_times'] += sr['sat_times']
                results['eqs_solved_Z3'] += sr['eqs_solved_Z3']
                results['sat_steps'] += sr['sat_steps_taken']
    eqs_solved = set([x[1] for x in results['eqs_solved']])
    sc = np.array([x[1] for x in results['eqs_solved']])
    tm = np.array([x[-1] for x in results['eqs_solved']])
    score = len(sc)
    time_avg = np.mean(tm)
    time_std = np.std(tm)
    l=len(results['sat_steps'])
    steps = np.array(results['sat_steps'])
    steps_avg = np.mean(steps)
    steps_std = np.std(steps)
    if args.test_solver:
        solver_eqs_solved = set([x[1] for x in results['eqs_solved_Z3']])
        solver_score = len([x[1] for x in results['eqs_solved_Z3']])
        solver_tm = np.array([x[-1] for x in results['eqs_solved_Z3']])
        solver_time_avg = np.mean(solver_tm)
        solver_time_std = np.std(solver_tm)

        intersection = eqs_solved & solver_eqs_solved
        intersection_solved = np.array([x[-1] for x in results['eqs_solved'] if x[1] in intersection])
        intersection_solver_solved = np.array([x[-1] for x in results['eqs_solved_Z3'] if x[1] in intersection])

        intersection_times = np.mean(intersection_solved)
        intersection_times_std = np.std(intersection_solved)
        intersection_times_solver =np.mean(intersection_solver_solved)
        intersection_times_solver_std =np.std(intersection_solver_solved)
    else:
        solver_time_avg = -1
        solver_score = -1
        intersection_times = -1
        intersection_times_std = -1
        solver_time_std=-1
        intersection_times_solver = -1
        intersection_times_solver_std=-1
        solver_eqs_solved=0
        intersection=-1
        solver_tm=-1
        intersection_solved=-1
        intersection_solver_solved=-1

    log = f'\nPool {args.pool_name}, Seed: {args.seed_class}\n' \
          f'Model: {os.path.join(model_folder, model_filename)}\n' \
          f'Score: {score}, Time avg: {time_avg} ({time_std}), Intersection time avg: {intersection_times} ({intersection_times_std}), ' \
          f'Num steps avg: {steps_avg} ({steps_std}) ({l})\n' \
          f'Solver score: {solver_score}, Solver time avg: {solver_time_avg} ({solver_time_std}), ' \
          f'Intersection time avg: {intersection_times_solver} ({intersection_times_solver_std}),\n' \
          f'Steps: {steps}\n'\
          f'SIDE_MAX_LEN: {args.SIDE_MAX_LEN}, num_vars: {len(args.VARIABLES)}, num_letters: {len(args.ALPHABET)}\n' \
          f'num_channels: {args.num_channels}, num_residual_blocks: {args.num_resnet_blocks}\n' \
          f'num mcts simulations: {args.num_mcts_simulations}\n' \
          f'Max solver time in mcts: {args.mcts_smt_time_max}\n' \
          f'Comments: {args.log_comments}\n' \
          f'Date: {datetime.datetime.today()}\n'

    with open(f'log_{args.smt_solver}.txt', 'a+') as f:
        f.write(log)

    log_full = f'\nPool {args.pool_name}, Seed: {args.seed_class}\n' \
          f'Model: {os.path.join(model_folder, model_filename)}\n' \
          f'Score: {score}, Time avg: {time_avg} ({time_std}), Intersection time avg: {intersection_times} ({intersection_times_std}), Num steps avg: {steps_avg} ({steps_std}) ({l})\n' \
          f'Solver score: {solver_score}, Solver time avg: {solver_time_avg} ({solver_time_std}), Intersection time avg: {intersection_times_solver} ({intersection_times_solver_std}),\n' \
          f'Solved: {eqs_solved}\n' \
          f'Solver solved: {solver_eqs_solved}\n' \
          f'Intersection: {intersection}\n'\
          f'Times: {tm}\n' \
          f'Solver times: {solver_tm}\n' \
          f'Intersection times: {intersection_solved}\n' \
          f'Intersection solver times: {intersection_solver_solved}\n' \
          f'SIDE_MAX_LEN: {args.SIDE_MAX_LEN}, num_vars: {len(args.VARIABLES)}, num_letters: {len(args.ALPHABET)}\n' \
          f'num_channels: {args.num_channels}, num_residual_blocks: {args.num_resnet_blocks}\n' \
          f'num mcts simulations: {args.num_mcts_simulations}\n' \
          f'Max solver time in mcts: {args.mcts_smt_time_max}\n' \
          f'Comments: {args.log_comments}\n' \
          f'Date: {datetime.datetime.today()}\n'

    with open(f'log_full_{args.smt_solver}.txt', 'a+') as f:
        f.write(log_full)

def individual_player_session(play_args):
    args, pool, model_folder, model_filename, mode, seed = play_args
    pname = args.pool_name_load.split('/')[1]
    if args.smt_solver is not None:
        try:
            args.mcts_smt_time_max = timeouts[args.smt_solver][pname]
        except:
            args.mcts_smt_time_max = 800
    if  model_folder in ['v55','v57','v61']:
        num_alph=10
        num_vars=14
        args.SIDE_MAX_LEN = 150
        args.ALPHABET = list(ascii_lowercase)
        args.VARIABLES = list(ascii_uppercase)
        #if 'track'   in args.pool_name:
        #    args.ALPHABET = [x for x in ascii_lowercase][0:num_alph]
        #    args.VARIABLES = [x for x in ascii_uppercase]
        #    args.ALPHABET = args.ALPHABET[:num_alph]
        #    args.VARIABLES = args.VARIABLES[:num_vars]
        #else:
        args.VARIABLES = [x for x in ascii_uppercase]
        args.VARIABLES = args.VARIABLES[:num_vars]
        args.ALPHABET = args.ALPHABET[:num_alph]
        args.LEN_CORPUS = len(args.VARIABLES)+len(args.ALPHABET)
        args.num_resnet_blocks=2
    elif model_folder in ['v56']:
        num_alph=3
        num_vars=5
        args.SIDE_MAX_LEN = 20
        args.num_mcts_simulations = 10
        args.ALPHABET = list(ascii_lowercase)
        args.VARIABLES = list(ascii_uppercase)
        args.num_resnet_blocks=2
        args.ALPHABET = [x for x in ascii_lowercase][0:num_alph]
        if 'track' not in args.pool_name:
            args.VARIABLES = [x for x in ascii_uppercase[::-1]]
        args.ALPHABET = args.ALPHABET[:num_alph]
        args.VARIABLES = args.VARIABLES[:num_vars]
        args.LEN_CORPUS = len(args.VARIABLES)+len(args.ALPHABET)

    elif  model_folder in ['v58']:
        num_alph=10
        num_vars=14
        args.SIDE_MAX_LEN = 150
        args.ALPHABET = list(ascii_lowercase)
        args.VARIABLES = list(ascii_uppercase)
        #if 'track'   in args.pool_name:
        #    args.ALPHABET = [x for x in ascii_lowercase][0:num_alph]
        #    args.VARIABLES = [x for x in ascii_uppercase]
        #    args.ALPHABET = args.ALPHABET[:num_alph]
        #    args.VARIABLES = args.VARIABLES[:num_vars]
        #else:
        args.VARIABLES = [x for x in ascii_uppercase]
        args.VARIABLES = args.VARIABLES[:num_vars]
        args.ALPHABET = args.ALPHABET[:num_alph]
        args.LEN_CORPUS = len(args.VARIABLES)+len(args.ALPHABET)
        args.num_resnet_blocks=2


    args.update_symbol_index_dictionary()
    seed_everything(seed)
    results = dict({'level': args.level})

    nnet = NNetWrapper(args, device='cpu', training=False, seed=seed)
    if model_filename != 'uniform':
        if model_filename != 'model_train_0.pth.tar':
            nnet.load_checkpoint(folder=model_folder, filename=model_filename)
    else:
        nnet.model = UniformModel(args, args.num_actions)

    nnet.training = False
    nnet.model.eval()
    for param in nnet.model.parameters():
        param.requires_grad_(False)

    args.active_tester = True


    player = Player(args, nnet,
                    mode=mode, name=f'player_0',
                    pool=pool,  # args.failed_pools[player_num] if (mode != 'test' or player_num <= 3) else [],
                    previous_attempts_pool=[], seed = seed)

    player.play(level_list=None)

    results['sat_times'] = player.execution_times_sat
    results['eqs_solved'] = player.eqs_solved
    results['eqs_solved_Z3'] = player.eqs_solved_z3
    results['sat_steps_taken'] = player.sat_steps_taken

    return results

