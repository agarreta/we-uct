
import re
import torch
from uct.we.word_equation import WordEquationUtils, seed_everything
from uct.we.word_equation import WordEquationTransformations

class WordEquationMovesQuadratic(object):

    def __init__(self, args, seed=0):
        self.args = args
        self.utils = WordEquationUtils(args)
        self.transformations = WordEquationTransformations(args)
        self.create_action_dict()
        self.create_fast_action_dict()
        if seed is not None:
            seed_everything(seed)

    def delete_var(self, eq, eq_side, word_side):
        eq_split = eq.w.split('=')
        #print(eq_split, eq_side, word_side)
        var = eq_split[eq_side][word_side]
        #let = eq_split[1 - side][0]
        if var not in self.args.VARIABLES:
            eq.attempted_wrong_move = True
            return eq
        else:
            new_w = re.sub(var, '', eq.w)
            eq.w = new_w
            eq.attempted_wrong_move = False
            if self.args.use_length_constraints:
                for o,x in enumerate(self.args.coefficients_variables_lc):
                    for _ in range(len( x)):
                        x[_][var] = 0
            return eq

    def move(self, eq, eq_side, word_side):
        # side = 0 if side == 'left' else 1
        eq_split = eq.w.split('=')
        word_side_aux = 0 if word_side == 0 else -1
        var = eq_split[eq_side][word_side_aux]
        let = eq_split[1-eq_side][word_side_aux]
        if var not in self.args.VARIABLES:
            eq.attempted_wrong_move = True
            return eq
        else:
            new_w = re.sub(var, ((1-word_side)*let)+var+((word_side)*let), eq.w)
            eq.w =new_w
            eq.attempted_wrong_move = False
            if self.args.use_length_constraints:
                for o,x in enumerate(eq.coefficients_variables_lc):
                    for _ in range(len( x)):
                        if let in self.args.VARIABLES:
                            x[_][let] += x[_][var]
                        elif let in self.args.ALPHABET:
                            eq.ell[o][_] -= x[_][var]
            return eq

    def act(self, eq, action_num, verbose=0):
        eq.attempted_wrong_move = False
        new_eq = eq.deepcopy()
        action = self.actions[action_num]
        type_of_action = action['type']

        if type_of_action == 'delete':
            eq_side = action['eq_side']
            word_side = action['word_side']
            new_eq = self.delete_var(new_eq, eq_side, word_side)

        if type_of_action in ['move_0', 'move_1']:
            eq_side = action['eq_side']
            word_side = action['word_side']
            new_eq = self.move(new_eq, eq_side, word_side)

        if new_eq.attempted_wrong_move:
            return new_eq

        new_eq.not_normalized = new_eq.get_string_form()
        new_eq = self.transformations.normal_form(new_eq)

        if self.utils.are_equal(new_eq, eq):
            new_eq.attempted_wrong_move = True
            return new_eq
        else:
            eq.attempted_wrong_move = False
            new_eq.get_id()
            return new_eq

    def get_afterstates(self, eq):
        afterstates = []
        for action_idx in self.actions.keys():
            new_eq = self.act(eq, action_idx)
            if new_eq.attempted_wrong_move:
                afterstates.append(0)
            else:
                afterstates.append(new_eq)
        return afterstates

    def get_valid_actions(self, eq):
        afterstates = self.get_afterstates(eq)
        valid_actions = torch.ones(len(self.actions), dtype=torch.float, requires_grad=False, device=self.args.play_device)
        for i, x in enumerate(afterstates):
            if type(x) == int:
                valid_actions[i] = 0.
        return valid_actions, afterstates

    def create_action_dict(self):
        """
        Creates a dictionary with possible actions.
        Each entry is a dictionary with different entries depending on the type of action
        """
        actions_delete = [{'description': f"delete_{eq_side}_{word_side} ",
                           'type': 'delete',
                           'eq_side': eq_side,
                           'word_side': word_side,
                           } for eq_side in [0,1] for word_side in [0,-1]]
        actions_move = [{'description': f"move_{eq_side}_{word_side}",
                           'type': f'move_{word_side}',
                           'eq_side': eq_side,
                           'word_side': word_side,
                           } for eq_side in [0,1] for word_side in [0,1]]

        self.actions = actions_delete + actions_move
        self.actions = {i: self.actions[i] for i in range(len(self.actions))}

    def get_action_size(self):
        self.create_action_dict()
        return len(self.actions)

    def create_fast_action_dict(self):
        'returns a dictionary that allows easy access to the number index each action has in self.action'
        alph = self.args.ALPHABET
        vars = self.args.VARIABLES
        fast_dict = {}
        fast_dict['delete'] = {'left': len(alph), 'right': len(alph)}
        fast_dict['move'] = {'left': len(alph), 'right': len(alph)}
        self.fast_dict = fast_dict
