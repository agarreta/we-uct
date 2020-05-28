# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:28:18 2019

@author: garre
"""


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


def solve_pool(args, pool, model_folder, model_filename, mode, seed, num_cpus=1):
    seed_everything(seed)
    if num_cpus == 1:
        results = individual_player_session((args, pool, model_folder, model_filename, mode, seed))
    else:
        p = Pool(num_cpus)
        chunk = ceil(len(pool)/num_cpus)
        eq_sub_pools = [pool[i*chunk:(i+1)*chunk] for i in range(num_cpus) if len(pool[i*chunk:(i+1)*chunk])>0]
        play_args = []
        sub_results = p.map(individual_player_session, [(args, sub_pool, model_folder, model_filename, mode, seed) for sub_pool in eq_sub_pools])
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
    num_solved = sum([len(x) for x in results['eqs_solved']])
    eqs_solved = set([x[1] for x in results['eqs_solved']])
    score = len([x[1] for x in results['eqs_solved']])
    time_avg = sum([x[-1] for x in results['eqs_solved']]) / max(1, score)
    l=len(results['sat_steps'])
    steps_avg = sum(results['sat_steps'])/max(1,l)
    if args.test_solver:
        solver_eqs_solved = set([x[1] for x in results['eqs_solved_Z3']])
        solver_score = len([x[1] for x in results['eqs_solved_Z3']])
        solver_time_avg = sum([x[-1] for x in results['eqs_solved_Z3']])/max(1,solver_score)

        intersection = eqs_solved & solver_eqs_solved
        intersection_solved = [x for x in results['eqs_solved'] if x[1] in intersection]
        intersection_solver_solved = [x for x in results['eqs_solved_Z3'] if x[1] in intersection]

        intersection_times = sum([x[-1] for x in intersection_solved])/max(1,len(intersection_solved))
        intersection_times_solver = sum([x[-1] for x in intersection_solver_solved])/max(1,len(intersection_solver_solved))
    else:
        solver_time_avg = -1
        solver_score = -1
        intersection_times = -1
        intersection_times_solver = -1

    log = f'\nPool {args.pool_name}, Seed: {args.seed_class}\n' \
          f'Model: {os.path.join(model_folder, model_filename)}\n' \
          f'Score: {score}, Time avg: {time_avg}, Intersection time avg: {intersection_times}, Num steps avg: {steps_avg} ({l})\n' \
          f'Solver score: {solver_score}, Solver time avg: {solver_time_avg}, Intersection time avg: {intersection_times_solver},\n' \
          f'SIDE_MAX_LEN: {args.SIDE_MAX_LEN}, num_vars: {len(args.VARIABLES)}, num_letters: {len(args.ALPHABET)}\n' \
          f'num_channels: {args.num_channels}, num_residual_blocks: {args.num_resnet_blocks}\n' \
          f'num mcts simulations: {args.num_mcts_simulations}\n' \
          f'Max solver time in mcts: {args.mcts_smt_time_max}\n' \
          f'Comments: {args.log_comments}\n' \
          f'Date: {datetime.datetime.today()}\n'

    with open(f'log_{args.smt_solver}.txt', 'a+') as f:
        f.write(log)

def individual_player_session(play_args):
    args, pool, model_folder, model_filename, mode, seed = play_args
    results = dict({'level': args.level})

    nnet = NNetWrapper(args, device='cpu', training=False, seed=seed)
    if model_filename!= 'uniform':
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


