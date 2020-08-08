import numpy as np
import math
import re
import random
import networkx as nx
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
import random
from copy import deepcopy
import torch
import re
import numpy as np
import os
from pickle import Pickler, Unpickler
import logging
#gym.logger.set_level(logging.INFO)


class Hanoi(object):
    def __init__(self, dim=3, puzzle=None, args=None):

        self.args = args
        self.num_discs = args.num_discs if args is not None else 4
        self.puzzle = puzzle if puzzle is not None else  self.get_final_state()
        self.utils = HanoiUtils(self.args)
        self.level = 1
        self.w = self.get_string_form(self.puzzle)
        self.sat = self.check_sat(self)
        self.ctx = None

        # The next are only used in equation generator

        self.attempted_wrong_move = False

        self.candidate_sol = dict({x : '' for x in range(10)})
        self.not_normalized = ''
        self.id = self.w  # self.get_string_form() + str(round(random.random(), 5))[1:]

    def __repr__(self):
        a=self.puzzle.view(-1).cpu().detach().numpy()
        s=''
        for i,x in enumerate(a):
            s += str(int(x))
            if (i+1)%(self.num_discs+1)==0 and i > 0:
                s+='|'
        return s #str(self.puzzle.cpu().detach().numpy())

    def get_final_state(self):
        t= torch.zeros(3, self.num_discs+1, dtype=torch.float) #  the first column is a dummy column for implementation issues (see act() function)
        t[-1,1:] = torch.ones(self.num_discs, dtype=torch.float)
        return t

    def get_string_form(self, puzzle=None):
        return self.__repr__()

    def get_string_form_for_print(self):
        return self.__repr__()

    def simplify_candidate_sol(self):
        return self.__repr__()

    def update(self, puzzle):
        self.puzzle = puzzle
        self.w = self.get_string_form(puzzle)
        self.sat = self.check_sat(self)
        self.id = str(random.random())# self.w  #self.get_string_form() + str(round(random.random(), 5))[1:]

    def check_sat(self, puzzle):
        return self.utils.check_satisfiability(puzzle)

    def deepcopy(self, copy_id=None):
        return deepcopy(self)

class HanoiTransformations(object):
    def __init__(self, args=None):
        pass

    def normal_form(self, puzzle, minimize=True, mode='play'):
        return puzzle

class HanoiUtils(object):
    def __init__(self, args=None, seed=0):
        self.args = args
        self.num_discs = args.num_discs if args is not None else 4
        self.final_state = self.get_final_state()
        self.final_state_w = 1

    def format_state(self, puzzle, device='cpu'):
        return puzzle.puzzle

    def check_satisfiability(self, puzzle, time=500):
        p = puzzle.puzzle

        # condition = torch.mean((p - self.final_cube)**2).item() < 1e-8
        condition = (p[2,1:].mean()-1)**2 < 0.0001  # self.are_equal(puzzle, self.final_state_w)
        if condition:
            puzzle.sat = 1
        else:
            puzzle.sat = 0
        return puzzle

    def are_equal(self, puzz1, puzz2):
        return (str(puzz1) == str(puzz2))  # .all().item() # False  # all actions lead to a different state   # puzz1.w == puzz2.w

    def get_final_state(self):
        t= torch.zeros(3, self.num_discs+1, dtype=torch.float)
        t[-1,1:] = torch.ones(self.num_discs, dtype=torch.float)
        return t

class HanoiMoves(object):
    def __init__(self, args=None, seed=0):
        self.action_info={
            0: ['peg1_1_peg2_2', [0,1]],
            1: ['peg1_1_peg2_3', [0,2]],
            2: ['peg1_2_peg2_1', [1,0]],
            3: ['peg1_2_peg2_3', [1,2]],
            4: ['peg1_3_peg2_1', [2,0]],
            5: ['peg1_3_peg2_2', [2,1]],
        }
        self.actions=[{'description':self.action_info[_]} for _ in range(self.get_action_size())]
        self.num_discs = args.num_discs if args is not None else 4

    def act(self, puzzle, action_num, verbose=0, mode='play'):
        puzzle_copy = puzzle.deepcopy()

        pegs = self.action_info[action_num][-1]

        hanoi = puzzle_copy.puzzle
        hanoi_cs = hanoi.cumsum(-1)
        min_disks = hanoi_cs.argmin(-1)
        if min_disks[pegs[0]] < min(min_disks[pegs[1]], self.num_discs-1) and hanoi_cs[pegs[0], :].sum() > 0.0001: # disks are the largest on the right
            hanoi[pegs[0], min_disks[pegs[0]]+1] = 0.
            hanoi[pegs[1], min_disks[pegs[0]]+1] = 1.
            #puzzle_copy.puzzle = hanoi
            puzzle_copy.attempted_wrong_move = False
            puzzle_copy.update(hanoi)

            return puzzle_copy
        else:
            puzzle_copy.attempted_wrong_move = True
            #if mode =='play':
            #    if hanoi_cs[pegs[0], :].sum() > 0.0001:#  and min_disks[pegs[0]] +1==hanoi.shape[1]-1:
            #        puzzle_copy.attempted_wrong_move = False
            #        #print(hanoi)
            #        hanoi[pegs[0], min_disks[pegs[0]] + 1] = 0.
            #        hanoi[pegs[1], min_disks[pegs[0]] + 1] = 1.
            #        hanoi[pegs[1], :min_disks[pegs[0]] + 1]=0.
            #        puzzle_copy.update(hanoi)
            #    else:
            #        puzzle_copy.attempted_wrong_move = True
#
            #else:
            #    puzzle_copy.attempted_wrong_move = True
#
        return puzzle_copy

    def get_action_size(self):
        return len(self.action_info)

    def get_afterstates(self, puzzle):
        return [self.act(puzzle, action) for action in range(self.get_action_size())]

    def get_valid_actions(self, eq):
        afterstates = self.get_afterstates(eq)
        valid_actions = torch.tensor([float(not afterstates[i].attempted_wrong_move) for i in range(6)]) #torch.ones(self.get_action_size(), dtype=torch.float, requires_grad=False, device='cpu')
        return valid_actions, afterstates



class HanoiGenerator(object):
    def __init__(self, args=None, seed=0):
        self.args=args
        self.moves = HanoiMoves(args=self.args, seed=0)

    def generate_pool(self, size, level_list, forbidden_list=None):
        self.pool = []
        visited =set({})
        if forbidden_list is not None:
            visited = set([x.w for x in forbidden_list])
        num=0
        while len(self.pool) < size:
            puzzle = self.get_puzzle(level_list[num])
            if True:  ## puzzle.w not in visited:
                self.pool.append(puzzle)
                visited.update({puzzle.w})
                num+=1
                if self.args.test_mode:
                    print(len(self.pool),puzzle.level, puzzle.w)
        self.save_pool([10,10], 100)
        self.pool_generation_time=1
        print('pool generated')

    def get_puzzle(self, level):
        #t= torch.zeros(3, self.args.num_discs+1, dtype=torch.float)
        #t[0,1:] = torch.ones(self.args.num_discs, dtype=torch.float)
        #return Hanoi(args=self.args, puzzle=t)
        dist = [2, 1, 2, 1, 3, 3]
        acts = [a for a in range(6) for _ in range(dist[a])]
        #print(acts)
        hanoi = Hanoi(puzzle=None, dim=3, args=self.args)
        visited =set({})
        while hanoi.level < level:# _ in range(level -hanoi.level):
            act = random.choice(acts)
            hanoi = self.moves.act(hanoi, act, mode = 'generation')
            if not hanoi.attempted_wrong_move:
                if hanoi.w not in visited:
                    hanoi.level +=1
                    #print(hanoi.level, hanoi)
                    visited.update({hanoi.w})

        return hanoi

    def save_pool(self, level_list, size):
        folder = self.args.folder_name + '/pools'
        if not os.path.exists(folder):
            os.makedirs(folder)
        pool_names = os.listdir(folder)

        if not self.args.test_mode:
            filename = os.path.join(folder, f'hanoi_pool{len(pool_names)}_lvl_{level_list[0]}_{level_list[-1]}_size_{size}.pth.tar')
        else:
            filename = os.path.join('benchmarks', f'pool_lvl_{level_list[0]}_{level_list[-1]}_size_{size}_{self.args.generation_mode}_{self.args.size_type}_{self.args.SOLUTION}.pth.tar')

        with open(filename, "wb+") as f:
            Pickler(f).dump(self.pool)
        f.close()



class Args():
    def __init__(self):
        self.num_discs=12

args = Args()

Moves = HanoiMoves(args)


def multinomial(n, k, num):
    return math.factorial(n) / (math.factorial(k) ** num)


def moves_wordgame():
    mo = [(a + b, b + a) for a in alphabet for b in alphabet]
    return mo


def moves_hanoi():
    pass


def generate_graph(word, nx_graph=False):
    # moves = [(a + b, b + a) for a in alphabet for b in alphabet]
    # moves = [(a + b, b + a) for i,a in enumerate(alphabet)
    #         for j, b in enumerate(alphabet) if i > j]
    moves = range(Moves.get_action_size())

    states = set({word.w})
    edges = set({})
    words_to_check = set({word})
    if nx_graph:
        G = nx.Graph()
        G.add_node(word)
    else:
        G = None

    def process(word):
        # words_to_check.remove(word)
        for move in moves:
            new_word = Moves.act(word, move)

            if new_word.w != word.w:
                edges.update({(word.w, move, new_word.w)})
                if nx_graph:
                    G.add_edge(word.w, new_word.w)
                    G.add_edge(new_word.w, word.w)

            if new_word.w not in states:
                states.update({new_word.w})
                if nx_graph:
                    G.add_node(new_word.w)
                words_to_check.update({new_word})
                # generate_states(new_word, states)

    while len(words_to_check) > 0:
        process(words_to_check.pop())
        if len(states) % 1000 == 0:
            print(f'{len(states)} states created')
    return states, edges, G


init_word = 'aabbccdd'  # 2500, 64%
init_word = 'aaaaabbbbbccccc'  # 620.000, 63%
init_word = 'aaaabbbbcccc'  # 30000, 61%
init_word = 'aaaaaaaabbbbbbbbc'  # 176.000, 48%

init_word = 'aaaaaaabbbbbbbcc'  # 366.000, 67%
init_word = 'aaaaaabbbbbbbcc'  # 383.000, 57%
init_word = 'abcdefghi'  # 362.000, 75%

init_words = [
    'aaaaaabbbbbbc',  # 10.000, 47%
    'aaaabbbbccc',  # 10.000, 59%
    'aaabbbcccd',  # 16.000, 66%
    'aaabbccdd',  # 7.500, 66%
    'abcdefgha',  # 181K, 74%
]

for init_word in init_words:
    print(init_word)
    alphabet = list(set(init_word))

    states, edges, G = generate_graph(Hanoi(args=args), nx_graph=True)
    # print(states)
    # print(edges)
    print('number of states: ', len(states))

    print('number of edges: ', len(edges) / 2)
    print(f'number of clique edges: {0.5 * len(states) * (len(states) - 1)}')
    print(f'cyclomatic number: {(0.5 * len(edges) - len(states) + 1) / (0.5 * len(edges))}')
    # nx.draw(G, with_labels=True)
    # plt.show()
    print('\n')