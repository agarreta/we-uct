# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:31:06 2019

@author: garre
"""

import torch
import numpy as np
import math
from .word_equation.we import WE
from z3 import sat, unsat, unknown
import time
import random
from .utils import seed_everything
from .SMTSolver import SMT_eval
# torch.set_default_tensor_type(torch.HalfTensor)
EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, nnet, args, num_mcts_sims, timeout_mcts, max_steps, level, nn_outputs, mode, seed=None):

        if seed is not None:
            seed_everything(seed)

        self.nnet = nnet
        self.args = args
        self.use_value_nn_head = self.args.use_value_nn_head
        self.we = WE(args, seed)
        #self.Z3converter = SMTSolver(self.args.VARIABLES, self.args.ALPHABET, self.args.solver_timeout, self.args.use_length_constraints)

        self.state_action_values = {}       # stores Q values for s,a (as defined in the paper)
        self.num_times_taken_state_action = {}       # stores #times edge s,a was visited
        self.num_times_visited_state = {}        # stores #times board s was visited
        self.prior_probs_state = {}        # stores initial policy (returned by neural net)

        self.final_state_value = {}        # stores game.getGameEnded ended for board s
        self.prior_state_value = {}
        self.is_leaf = {}
        self.next_state_context = {}
        self.previous_state_context = {}
        self.registered_ids = set({})
        self.afterstates = {}
        self.valid_actions = {}

        self.leaf_values = {}
        self.root_action = -1

        self.nn_outputs = nn_outputs

        # self.afterstates_in_state_and_value = {}      # stores all afterstates and their r value (-1,0,1) for state s

        self.prev_state = ''
        self.num_rec = 0
        self.num_reps  =0
        self.resign = False
        self.num_fails = 0
        self.repeat = 0
        self.noise_level = 0.
        self.num_mcts_simulations = num_mcts_sims
        self.max_steps = max_steps
        self.device = self.args.play_device
        self.search_times = []

        self.debug_log= []

        self.depth = 0
        # self.level = level
        self.search_time = time.time()
        self.edges = {}
        self.timeout = timeout_mcts
        self.mode = mode
        self.set_of_equations = set({})

        self.coef_solver = self.args.solver_proportion
        self.timeout_value = self.args.timeout_value_smt
        self.num_z3_evals = 0
        self.num_unknown_evals = 0
        self.value_mode = self.args.value_mode

        self.noise_param = self.args.noise_param

        self.meanz3times = [0,0]

        self.level = args.level
        self.discount = args.discount
        # self.max_mcts_time = max_mcts_time

        self.new_leaf=''
        self.old_leaf = ''
        self.found_sol =[False,0]

        self.use_leafs = self.args.use_leafs  # True if self.mode == 'train' else True
        self.affine_transf = self.args.affine_transf

    def clear(self):
        self.edges = {}
        self.state_action_values = {}  # stores Q values for s,a (as defined in the paper)
        self.num_times_taken_state_action = {}  # stores #times edge s,a was visited
        self.num_times_visited_state = {}  # stores #times board s was visited
        self.prior_probs_state = {}  # stores initial policy (returned by neural net)

        self.final_state_value = {}  # stores game.getGameEnded ended for board s

        # self.afterstates_in_state_and_value = {}      # stores all afterstates and their r value (-1,0,1) for state s

        self.num_rec = 0
        self.depth = 0

    def get_value_from_Q_values(self, eq, prior_pi):
        value = 0
        values = []
        new_pi = []
        for a in range(self.args.num_actions):
            if eq in self.afterstates:
                s = self.afterstates[eq][a]
                if s in self.final_state_value:
                    if self.final_state_value[s] == 1:
                        new_pi = [0]*self.args.num_actions
                        new_pi[a] = 1
                        values = [0] * self.args.num_actions
                        values[a] = 1
                        return 1, new_pi, values


            if (eq, a) in self.state_action_values:
                v = self.state_action_values[(eq, a)]
                value += prior_pi[a]*v
                new_pi.append(v)
                values.append(value)
                if v == 1:
                    new_pi = [0]*self.args.num_actions
                    new_pi[a] = 1
                    values = [0] * self.args.num_actions
                    values[a] = 1
                    return 1, new_pi, values
            else:
                new_pi.append(0)
                values.append(0)
        #ar = np.argmax(np.array(prior_pi))
        #value = values[ar]
        if sum(new_pi) != 0:
            return value, np.array(new_pi)/sum(new_pi), values
        else:
            return value, np.array(self.args.num_actions*[1/self.args.num_actions]), values

    def get_action_prob(self, eq, s_eq, temp=1, previous_state=None):
        """
        This function performs num_mcts_simulations simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to num_times_taken_state_action[(s,a)]**(1./temp)

        """
        if False:
            eq = self.we.utils.check_satisfiability(eq)
            self.final_state_value[eq.w] = eq.sat
            valid_actions, afterstates = self.we.moves.get_valid_actions(eq)
            values = []
            for i in range(len(afterstates)):
                if valid_actions[i] > 0:
                    values.append(self.nnet.predict(s_eq)[1].item())
                else:
                    values.append(0)
            probs = [0. for i in range(len(valid_actions))]
            probs[int(np.argmax(values))] = 1.
            return probs

        if self.args.use_clears:  # True: #self.mode == 'train':
            self.clear()
        self.num_rec = 0

        self.num_times_taken_state_action_during_get_action_prob = {}
        ctx = None
        if eq.id not in self.edges:
            self.edges[eq.id]=[]

        self.initial_time = time.time()
        self.root_w = eq.w
        for i in range(self.num_mcts_simulations):

            self.search(eq, s_eq, temp, previous_state=previous_state, ctx= ctx)
            if time.time() - self.initial_time > self.args.timeout_time:
                break
            if self.args.active_tester and eq.sat == self.args.sat_value:
                break
            self.num_rec = 0
            self.root_action = -1
            self.new_leaf = ''
            self.old_leaf = ''
            self.new_leaf_available = False
            if self.args.forbid_repetitions:
                self.set_of_equations = []

        self.depth += 1

        num_actions = self.args.num_actions if not self.args.sat else 2*eq.sat_problem.num_vars


        #if self.mode == 'test':
        #    counts = [self.num_times_taken_state_action_during_get_action_prob[(eq.w, a)] if (eq.w, a) in self.num_times_taken_state_action_during_get_action_prob else 0 for a in range(self.args.num_actions)]
        #else:
        #    counts = [self.num_times_taken_state_action_during_get_action_prob[(eq.w, a)]
        #              if (eq.w, a) in self.num_times_taken_state_action_during_get_action_prob else 0
        #              for a in range(num_actions)]
        if self.mode == 'test':
            counts = [self.num_times_taken_state_action[(eq.w, a)] if (eq.w, a) in self.num_times_taken_state_action else 0 for a in range(self.args.num_actions)]
        else:
            counts = [self.num_times_taken_state_action[(eq.w, a)]
                      if (eq.w, a) in self.num_times_taken_state_action else 0
                      for a in range(num_actions)]
        if self.args.mcts_value_type == 'Qvalues':
            su=sum(counts)
            if su != 0:
                value, pi, values = self.get_value_from_Q_values(eq.w, np.array(counts) / su)
                if random.random() < 0.01:
                    print(value, pi, values,np.array(counts)/su)

                #
                pi = counts
                if temp == 0:
                    bestA = np.argmax(pi)
                    probs = [0] * len(pi)
                    probs[int(bestA)] = 1
                    return value, probs
                else:
                    #pi = pi/sum(pi)
                    return value, pi
            else:
                valids = self.we.moves.get_valid_actions(eq)
                print('strange error', counts)
                print(eq.get_string_form())
                probs = np.ones(self.args.num_actions)*np.array(valids)
                print(probs)
                probs /= np.sum(probs)
                return 0., probs
        else:

            #if len(self.final_state_value) == 1 and self.mode == 'test':
            #    return 'already_sat'

            if temp == 0:
                bestA = np.argmax(counts)
                probs = [0] * len(counts)
                probs[int(bestA)] = 1
                return 0, probs


            su = float(sum(counts))
            if su != 0:
                 probs = [x/su for x in counts]
            else:
                valids = self.we.moves.get_valid_actions(eq)
                print('strange error', counts)
                print(eq.get_string_form())
                probs = np.ones(self.args.num_actions) * np.array(valids)
                print(probs)
                probs /= np.sum(probs)

                 # raise ValueError
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

    #def network_output(self, eq_name, eq_s, smt = None, ctx=None):
    #    if eq_name in self.nn_outputs:
    #        return self.nn_outputs[eq_name]
    #    else:
    #        output = self.nnet.predict(eq_s, smt, ctx)
    #        self.nn_outputs[eq_name] = output
    #        return output

    def get_state_value(self, eq_string, dist_nn, value_nn):
        value_nn = 1*value_nn #+ 0.25*np.random.vonmises(0,0.1)/3.141592
        return value_nn


    def network_output(self, eq_name, eq_s, smt=None, ctx=None, eq=None):

        if eq_name in self.nn_outputs:
            return self.nn_outputs[eq_name]
        else:
            output_pi, output_v = self.nnet.predict(eq_s, smt, ctx)
            # print(self.args.oracle)
            if self.args.oracle:
                output_v = self.oracle_rollout(eq)
            output = (output_pi, output_v)
            self.nn_outputs[eq_name] = output
            return output

    def oracle_rollout(self, eq):
        # print('rollout')
        num_rollouts = 1
        scores = []
        original_eq = eq.deepcopy()
        for _ in range(num_rollouts):
            eq = original_eq.deepcopy()
            for i in range(1):
                # z3out = SMT_eval(self.args, eq)
                # if z3out > 0:
                #     scores.append(z3out)
                #     break
                print(self.args.mcts_smt_time_max)
                if eq.w not in self.final_state_value:
                    print(self.args.mcts_smt_time_max)
                    eq = self.we.utils.check_satisfiability(eq, smt_time = self.args.mcts_smt_time_max)
                    self.final_state_value[eq.w]=eq.sat
                else:
                    eq.sat = self.final_state_value[eq.w]
                #                print(eq.sat)

                if eq.sat != 0:
                    if eq.sat > 0:
                        self.found_sol = [True, self.num_rec]
                        return 1.
                    else:
                        scores.append(-1)
                        break
                if not eq.w in self.valid_actions:
                    valid_act, afterstates = self.we.moves.get_valid_actions(eq)
                    self.valid_actions[eq.w] = valid_act
                    self.afterstates[eq.w]= afterstates
                valid_a = self.valid_actions[eq.w]
                valid_a = [i for i, x in enumerate(list(valid_a)) if x > 0]
                if len(valid_a)==0:
                    print('no valid actions during random rollout')
                    print(eq.w, valid_a)
                    action = random.choice(range(self.args.num_actions))
                else:
                    action = random.choice(valid_a)
                eq = self.we.moves.act(eq, action)
            scores.append(0)
        #print(sum(scores)/len(scores))
        if len(scores)==0: scores = [0]
        sc = sum(scores)/num_rollouts
        if sc > 0:
            self.found_sol = [True, self.num_rec]
        return sc

    def oracle_rollout_wrong(self, eq):
        # print('rollout')
        num_rollouts = 5
        scores = []
        original_eq = eq.deepcopy()
        for _ in range(num_rollouts):
            eq = original_eq.deepcopy()
            for i in range(5):
                # z3out = SMT_eval(self.args, eq)
                # if z3out > 0:
                #     scores.append(z3out)
                #     break
                if eq.w not in self.final_state_value:
                    eq = self.we.utils.check_satisfiability(eq, smt_time = self.args.mcts_smt_time_max)
                    self.final_state_value[eq.w]=eq.sat
                else:
                    eq.sat = self.final_state_value[eq.w]
                #                print(eq.sat)
                scores.append(-1)

                if eq.sat != 0:
                    if eq.sat > 0:
                        self.found_sol = [True, self.num_rec]
                        return 1.
                    break
                #if not eq.w in self.valid_moves_in_state:
                #    valid_act, afterstates = self.we.moves.get_valid_actions(eq)
                #    self.valid_moves_in_state[eq.w] = valid_act
                valid_a = self.valid_actions[eq.w]
                valid_a = [i for i, x in enumerate(list(valid_a)) if x > 0]
                if len(valid_a)==0:
                    print('no valid actions during random rollout')
                    print(eq.w, valid_a)
                    action = random.choice(range(self.args.num_actions))
                else:
                    action = random.choice(valid_a)
                eq = self.we.moves.act(eq, action)
        #print(sum(scores)/len(scores))
        if len(scores)==0: scores = [0]
        sc = sum(scores)/num_rollouts
        if sc > 0:
            self.found_sol = [True, self.num_rec]
        return sc


    def search(self, state, s_eq, temp, previous_state=None, ctx=None):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of num_times_visited_state, num_times_taken_state_action, state_action_values are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """


        state_id = state.id
        state_w = state.w

        # print( self.registered_true_ids)
        #if state_w not in self.registered_true_ids:
        #    self.registered_true_ids.update({state_w})
        #else:
        #    self.num_reps += 1



        if state_w not in self.final_state_value:
            t = time.time()
            state = self.we.utils.check_satisfiability(state, smt_time=self.args.mcts_smt_time_max)  # max(1000-self.meanz3times[0], 50))
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
            else:
                valid_actions = self.valid_actions[state_w]

            if self.args.mcts_type =='alpha0':
                self.prior_probs_state[state_w] = state_nn_dist

                if temp!=0:
                    epsilon = 1.25
                else:
                    epsilon = 0

                noise = np.random.dirichlet(self.noise_param * np.ones(self.prior_probs_state[state_w].shape))

                self.prior_probs_state[state_w] = (1 - epsilon) * self.prior_probs_state[state_w] + \
                                                   epsilon * torch.tensor(noise,
                                                                          dtype=torch.float).to(self.device)
                self.prior_probs_state[state_w] = self.prior_probs_state[
                                                       state_w] * valid_actions  # .cpu().numpy()      # masking invalid moves

                sum_Ps_state = torch.sum(self.prior_probs_state[state_w])
                if sum_Ps_state > 0:
                    self.prior_probs_state[state_w] /= sum_Ps_state  # renormalize
                else:
                    print("All valid moves were masked, do workaround.")
                    self.prior_probs_state[state_w] = self.prior_probs_state[state_w] + valid_actions  # .cpu().numpy()

            #self.valid_moves_in_state[state_w] = valid_actions
            if state_w not in self.num_times_visited_state:
                self.num_times_visited_state[state_w] = 0

            return self.prior_state_value[state_w], state_id, previous_state

        if self.num_rec >= 3000: #550:  # self.max_steps:
            return self.args.unsat_value,  state_id, previous_state


        if False:# (self.args.hanoi  and self.num_rec  > 2**self.args.num_discs):
                    #or (self.num_rec + self.depth > 2 * self.level and not 'plain' in self.args.folder_name):  # and
                # (not self.args.values_01):  # self.max_steps_fun(max(self.level - self.depth,3)):  # time.time()-self.search_time > self.timeout:
            self.prior_state_value[state_w] = self.args.value_timed_out_simulation  # min(0, self.prior_state_value[state_id])


        self.num_rec += 1

        valid_actions = self.valid_actions[state_w]
        cur_best = -float('inf')
        best_act = -1

        cpuct = 1.25 #math.log((self.num_times_visited_state[state_w] + self.args.pb_c_base + 1) /
                  #       self.args.pb_c_base) + self.args.pb_c_init

        self.num_times_visited_state[state_w] += 1


        num_actions = self.args.num_actions if not self.args.sat else 2*state.sat_problem.num_vars


        for a in range(num_actions):
            if valid_actions[a] != 0.:
                if (state_w, a) in self.state_action_values and (state_w, a) in self.num_times_taken_state_action:
                    #print('hi', cpuct)
                    if self.args.mcts_type=='alpha0': # False: #not self.args.oracle:
                        # d = [math.sqrt(self.prior_probs_state[state_w][a]) for a in range(self.args.num_actions)]
                        # d = d[a]/sum(d)
                        d =self.prior_probs_state[state_w][a]
                        UCT = cpuct * d * \
                              math.sqrt(np.log(self.num_times_visited_state[state_w]))
                        UCT = UCT / math.sqrt((1 + self.num_times_taken_state_action[(state_w, a)]))
                    elif self.args.mcts_type == 'alpha0np':
                        UCT = cpuct *math.sqrt(np.log(self.num_times_visited_state[state_w]))
                        UCT = UCT / math.sqrt((1 + self.num_times_taken_state_action[(state_w, a)]))
                    else:
                        #UCT = math.sqrt(2*np.log(self.num_times_visited_state[state_w])/
                        #                (1 + self.num_times_taken_state_action[(state_w, a)]))
                        UCT = cpuct  * math.sqrt(np.log(self.num_times_visited_state[state_w]))
                        UCT = UCT / math.sqrt((1 + self.num_times_taken_state_action[(state_w, a)]))

                    u = self.state_action_values[(state_w, a)]
                    #u = 1. + (u-1.)/2.
                    u = u + UCT
                else:
                    #print('ho', cpuct)

                    # print('B')
                    if  False: #not self.args.oracle:
                        u = 10.#cpuct *math.sqrt(np.log(self.num_times_visited_state[state_w])) #cpuct * math.sqrt(self.prior_probs_state[state_w][a])
                    else:
                        u = 0.2*cpuct

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

        ctx = None
        if True: #time.time() - self.search_time < self.timeout/10:
            v, new_leaf, old_leaf = self.search(next_state, next_s_eq, temp, state, ctx)
        else:
            v, new_leaf, old_leaf = -1, next_state, state

        self.update_state_action_value(state, a, v)
        return self.discount*v, new_leaf, old_leaf


    def update_state_action_value(self, state, a, v=0):
        state_id = state.id
        state_w = state.w
        if type(v)==torch.Tensor:
            v = v.item()
        #if state_w == self.root_w:
        #    if (state_w, a) not in self.num_times_taken_state_action_during_get_action_prob:
        #        self.num_times_taken_state_action_during_get_action_prob[(state_w, a)] = 1
        #    else:
        #        self.num_times_taken_state_action_during_get_action_prob[(state_w, a)] += 1

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

