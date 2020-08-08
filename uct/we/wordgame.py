import gym
import gym_sokoban
from copy import deepcopy
import torch
import re
import numpy as np
import os
from pickle import Pickler, Unpickler
import random
from .word_equation_utils import seed_everything


class WordGame(object):
    def __init__(self, solution, word= None):
        self.state = word
        self.solution = solution


class WordGameToWE(object):
    def __init__(self, wordgame, args=None, seed =None):

        self.args = args

        self.ctx = None
        self.wordgame = wordgame
        # The next are only used in equation generator

        self.attempted_wrong_move = False
        self.candidate_sol = dict({x : '' for x in range(10)})
        self.not_normalized = ''
        self.level = args.level

        self.w = self.get_string_form()
        self.id = self.w  # self.get_string_form() + str(round(random.random(), 5))[1:]
        self.sat = 0 if self.w != wordgame.solution else 1


    def get_string_form(self):
        return self.wordgame.state

    def get_string_form_for_print(self):
        return self.get_string_form()

    def update(self, wordgame):
        self.w = self.get_string_form()
        self.sat = 0 if self.w != wordgame.solution else 1
        self.id =str(random.random()) #self.get_string_form() + str(round(random.random(), 5))[1:]

    def deepcopy(self, copy_id=None):
        return deepcopy(self)

class WordGameTransformations(object):
    def __init__(self, args=None):
        pass

    def normal_form(self, puzzle, minimize=True, mode='play'):
        return puzzle

class WordGameUtils(object):
    def __init__(self, args=None, seed=0):
        self.channel={l:i for i,l in enumerate(args.wordgame_alphabet)}
        self.args = args
    def format_state(self, puzzle, device='cpu'):
        w=puzzle.wordgame.state
        t = torch.zeros(len(self.args.wordgame_alphabet)*self.args.wordgame_size, dtype =torch.float, device=device)
        for i,x in  enumerate(w):
            t[len(self.args.wordgame_alphabet)*self.channel[x] + i] = 1.
        return t

    def check_satisfiability(self, puzzle, time=500):
        if puzzle.wordgame.state == puzzle.wordgame.solution:
            puzzle.sat = 1
        else:
            puzzle.sat = 0
        if self.args.wordgame_tree:
            alph = list(set(self.args.wordgame_alphabet) -set({'0'}))
            if not any([x in puzzle.wordgame.state for x in [y+z for y in alph for z in alph if y!= z]]):
                puzzle.sat = -1

        return puzzle


class WordGameGenerator(object):
    def __init__(self, args=None, seed=0):
        self.args=args
        seed_everything(seed)
        self.seed = seed
    def swap(self, letter1, letter2, word):
        return re.sub(letter1 + letter2, letter2 + letter1, word)

    def get_random_initial_word(self):
        word = self.args.SOLUTION
        level = 0
        choices = 13*['swap'] +['del']
        current_alph = set(list(word))
        total_alph = self.args.wordgame_alphabet

        while  level <   41:#(self.args.level-self.args.min_level+1):
            if len(set(total_alph) - current_alph) ==0:
                choices = ['swap']
            mode = random.choice(choices)
            if mode == 'swap':
                l = list(set(self.args.wordgame_alphabet) - set({'0'}))

                letter1 = random.choice(l)
                l.remove(letter1)
                letter2 = random.choice(l)
                if self.args.wordgame_tree:
                    idx = word.find('0')
                    if idx >= 0 and len(word) > idx+1 and word[idx] == '0' and letter2+letter1 in word:
                        word = word[:idx] + letter1 + word[idx+1:] #[idx+1] = letter1
                new_word = self.swap(letter2, letter1, word)
            else:
                idx1 = random.choice(list(range(len(word[:-2]))))
                idx2 = idx1 + random.choice(list(range(len(word[idx1:]))))
                l = random.choice(list(set(total_alph) - set(current_alph)))
                new_word = word[:idx1]  + l + word[idx1:idx2] + l + word[idx2:]
                current_alph.update({l})

            if new_word != word:
                level += 1
                word= new_word
            #print(word)
        return word

    def generate_pool(self, *args, **kwargs):
        size = self.args.wordgame_size #if not self.args.test_mode else 100
        if not self.args.test_mode:
            self.pool = [WordGameToWE(WordGame(self.args.SOLUTION, self.get_random_initial_word()), self.args)
                         for _ in range(10)]
        else:
            self.pool = []
            for lvl in range(0,10):
                self.args.level = lvl
                self.pool += [WordGameToWE(WordGame(self.args.SOLUTION, self.get_random_initial_word()),  self.args, self.seed) for _ in range(size) for _ in range(20)]

        self.pool_generation_time=1
        if self.args.test_mode:
            print(len(self.pool))
            print('test pool generated')
        self.save_pool([10,10], 100)

    def save_pool(self, level_list, size):
        folder = self.args.folder_name + '/pools'
        if not os.path.exists(folder):
            os.makedirs(folder)
        pool_names = os.listdir(folder)

        if not self.args.test_mode:
            filename = os.path.join(folder, f'pool{len(pool_names)}_lvl_{level_list[0]}_{level_list[-1]}_size_{size}.pth.tar')
        else:
            filename = os.path.join('benchmarks', f'pool_lvl_{level_list[0]}_{level_list[-1]}_size_{size}_'
            f'standard_{self.args.size_type}_{self.args.SOLUTION}.pth.tar')

        with open(filename, "wb+") as f:
            Pickler(f).dump(self.pool)
        f.close()


class WordGameMoves(object):
    def __init__(self, args=None, seed=0):
        alph = list(set(args.wordgame_alphabet) - set({'0'}))
        pairs = [(x, y) for x in alph for y in alph if y!= x]
        self.action_dict = {i : pair for i, pair in
                            enumerate(pairs)}
        if  args.wordgame_del:
            self.action_dict.update({len(pairs)+i: a for i,a in enumerate(set({'c','d','e'}))})
        self.args = args

    def swap(self, letter1, letter2, word):
        return re.sub(letter1 + letter2, letter2 + letter1, word)

    def act(self, puzzle, action_num, verbose=0):
        if action_num==-1:
            return puzzle
        puzzle_copy = puzzle.deepcopy()

        a = self.action_dict[action_num]
        if type(a) == tuple:
            letter1, letter2 = a
            new_state = self.swap(letter1, letter2, puzzle.wordgame.state)
            if self.args.wordgame_tree:
                if new_state != puzzle.wordgame.state:
                    idx = new_state.find(letter1)
                    new_state = new_state[:idx] + '0' + new_state[idx+1:]
        else:
            new_state = re.sub(a, '', puzzle.wordgame.state)
        puzzle_copy.wordgame.state = new_state
        puzzle_copy.update(puzzle_copy.wordgame)
        return puzzle_copy

    def get_action_size(self):
        return len(self.action_dict)

    def get_afterstates(self, puzzle):
        return [self.act(puzzle, action) for action in range(self.get_action_size())]

    def get_valid_actions(self, eq):
        afterstates = self.get_afterstates(eq)
        valid_actions = torch.ones(self.get_action_size(), dtype=torch.float, requires_grad=False, device='cpu')
        for i,x in enumerate(afterstates):
            if x.w == eq.w:
                valid_actions[i]=0.
        return valid_actions, afterstates
