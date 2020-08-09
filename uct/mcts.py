# -*- coding: utf-8 -*-

import torch
import numpy as np
import math
from .word_equation.we import WE
import time
import random
from .utils import seed_everything
EPS = 1e-8


class MCTS():
    """
    """

    def __init__(self, nnet, args, num_mcts_sims, mode, seed=None):

        if seed is not None:
            seed_everything(seed)

        self.nnet = nnet
        self.args = args
        self.we = WE(args, seed)

        self.state_action_values = {}
        self.num_times_taken_state_action = {}
        self.num_times_visited_state = {}
        self.final_state_value = {}
        self.prior_state_value = {}
        self.is_leaf = {}
        self.registered_ids = set({})
        self.afterstates = {}
        self.valid_actions = {}

        self.root_action = -1

        self.prev_state = ''
        self.num_rec = 0
        self.num_mcts_simulations = num_mcts_sims
        self.device = self.args.play_device

        self.depth = 0
        self.search_time = time.time()
        self.edges = {}
        self.mode = mode

        self.noise_param = self.args.noise_param
        self.discount = args.discount

        self.new_leaf=''
        self.old_leaf = ''
        self.found_sol =[False,0]


    def get_action_prob(self, eq, s_eq, temp=1, previous_state=None):
        """
        """
        self.num_rec = 0
        self.num_times_taken_state_action_during_get_action_prob = {}
        if eq.id not in self.edges:
            self.edges[eq.id]=[]

        self.initial_time = time.time()
        self.root_w = eq.w
        for i in range(self.num_mcts_simulations):

            self.search(eq, s_eq, temp, previous_state=previous_state)
            if time.time() - self.initial_time > self.args.timeout_time:
                break

            self.num_rec = 0
            self.root_action = -1
            self.new_leaf = ''
            self.old_leaf = ''
            self.new_leaf_available = False


        self.depth += 1
        num_actions = self.args.num_actions if not self.args.sat else 2*eq.sat_problem.num_vars
        counts = [self.num_times_taken_state_action[(eq.w, a)] if (eq.w, a) in self.num_times_taken_state_action else 0
                  for a in range(self.args.num_actions)]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[int(bestA)] = 1
            return 0, probs

        su = float(sum(counts))
        assert su > 0
        probs = [x/su for x in counts]

        return 0, probs


    def check_if_new_node(self, state, next_state):
        for child_state in self.edges[state.id]:
            if child_state.get_string_form() == next_state.get_string_form():
                next_state.id = child_state.id
                next_state = child_state
                return next_state
        self.edges[state.id].append(next_state)
        self.edges[next_state.id] = []
        return next_state

    def network_output(self, eq_name, eq_s, smt=None, eq=None):

        output_pi, output_v = self.nnet.predict(eq_s, smt)
        if self.args.oracle:
            output_v = self.oracle_rollout(eq)
        output = (output_pi, output_v)
        #self.nn_outputs[eq_name] = output
        return output


    def search(self, state, s_eq, temp, previous_state=None):
        """
        """

        state_id = state.id
        state_w = state.w

        if state_w not in self.final_state_value:
            state = self.we.utils.check_satisfiability(state, smt_time=self.args.mcts_smt_time_max)
            self.final_state_value[state_w] = state.sat
            if state.sat == 1:
                self.found_sol = [True, self.num_rec]

        val = self.final_state_value[state_w]
        if val != self.args.unknown_value:
            if val == self.args.sat_value:
                return 10000, state_id, previous_state
            return val, state_id, previous_state

        if state_id not in self.registered_ids:

            state_nn_dist, state_value = self.network_output(state_w, s_eq, eq=state)

            self.prior_state_value[state_w] = state_value
            self.registered_ids.update({state_id})

            if state.w not in self.afterstates:
                valid_actions, afterstates = self.we.moves.get_valid_actions(state)
                self.afterstates[state_w]=afterstates
                self.valid_actions[state_w] = valid_actions

            if state_w not in self.num_times_visited_state:
                self.num_times_visited_state[state_w] = 0

            return self.prior_state_value[state_w], state_id, previous_state

        if self.num_rec >= 3000:
            return self.args.unsat_value,  state_id, previous_state

        self.num_rec += 1

        valid_actions = self.valid_actions[state_w]
        cur_best = -float('inf')
        best_act = -1

        cpuct = 1.25
        self.num_times_visited_state[state_w] += 1

        num_actions = self.args.num_actions

        for a in range(num_actions):
            if valid_actions[a] != 0.:
                if (state_w, a) in self.state_action_values and (state_w, a) in self.num_times_taken_state_action:
                    UCT = cpuct *math.sqrt(np.log(self.num_times_visited_state[state_w]))
                    UCT = UCT / math.sqrt((1 + self.num_times_taken_state_action[(state_w, a)]))
                    u = self.state_action_values[(state_w, a)]
                    u = u + UCT
                else:
                    if not self.args.oracle:
                        u = 10.
                    else:
                        print('anomaly')
                        u = 1.

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        if self.num_rec == 1:
            self.root_action = a

        # todo: inefficiwent since we computed the aftersates
        next_state = self.we.moves.act(state, a)
        next_state = self.check_if_new_node(state, next_state)
        next_s_eq = self.we.utils.format_state(next_state, self.device)

        v, new_leaf, old_leaf = self.search(next_state, next_s_eq, temp, state)

        self.update_state_action_value(state, a, v)
        return self.discount*v, new_leaf, old_leaf


    def update_state_action_value(self, state, a, v=0):
        state_w = state.w
        if type(v)==torch.Tensor:
            v = v.item()

        if (state_w, a) in self.state_action_values:
            if (state_w, a) not in self.num_times_taken_state_action:
                self.num_times_taken_state_action[(state_w, a)] = 1
            else:
                self.num_times_taken_state_action[(state_w, a)] += 1
            if v >= 1:
                self.state_action_values[(state_w, a)] = v
            else:
                nsa= self.num_times_taken_state_action[(state_w, a)]-1
                self.state_action_values[(state_w, a)] = (self.state_action_values[(state_w, a)]*nsa) +v
                self.state_action_values[(state_w, a)] /= nsa+1
        else:
            self.num_times_taken_state_action[(state_w, a)] = 1
            self.state_action_values[(state_w, a)] = v
