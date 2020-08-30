import time
import numpy as np
from .mcts import MCTS
from .word_equation.we import WE
import logging
from .utils import seed_everything

class Player(object):

    def __init__(self, args, nnet, mode='train', name='player', pool=None, seed=None):
        self.seed = seed
        if seed is not None:
            seed_everything(self.seed)

        self.args = args
        self.init_log(args.folder_name, mode)
        self.nnet = nnet
        self.we = WE(args, self.seed)
        print(f'seed {self.seed}')
        self.train_examples = []
        self.mode = mode
        self.device = self.args.play_device
        self.score = []
        self.num_actions_taken_if_successful = []
        self.eqs_solved = []
        self.name = name
        self.pool = pool
        self.execution_times = []

        if mode != 'train':
            self.args.temperature = 0.   # policy temperature: 1 for training mode and 0 for test mode (i.e.\ in the latter case we pick the action with the highest prob)

    def init_log(self, folder, mode='train'):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=folder + f'/log_{mode}.log')
        console = logging.StreamHandler()
        from PIL import PngImagePlugin
        logger = logging.getLogger(PngImagePlugin.__name__)
        logger.setLevel(logging.INFO)  # tame the "STREAM" debug messages
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    def timeout(self, elapsed_time):
        return elapsed_time > self.args.timeout_time

    def play(self):
        """
        Main method. Will attempt to solve the equations in self.pool (if this is 'None' then it will first generate a
        random pool.
        """
        self.nnet.model.eval()

        if self.pool is not None:
            pool = self.pool
            print(pool)

        # ::::: Use this code to generate test pools :::::
        #elif self.mode == 'test' and not self.args.active_tester:
        #    file_name = self.args.pool_name_load
        #    print('Loading {}'.format(file_name))
        #    if os.path.exists(file_name):
        #        with open(file_name, "rb") as f:
        #            pool = Unpickler(f).load()
        #            self.pool_generation_time = 0
        #        print([x.level for x in pool[:self.args.test_pool_length]])
        #        npool = []
        #        level_slots = [range(math.floor(10 + 1.6 * i), math.floor(10 + 1.6 * (i + 1))) for i in range(0, 9)]
        #        num_slots = [0 for _ in range(0, 9)]
        #        while len(npool) < self.args.test_pool_length:
        #            a = int(np.argmin([x for x in num_slots]))
        #            l = level_slots[a]
        #            for x in [y for y in pool if y not in npool]:
        #                if x.level in l:
        #                    npool.append(x)
        #                    num_slots[a] += 1
        #                    break
        #        assert len(pool) >= self.args.test_pool_length
        #        pool = npool[:self.args.test_pool_length]
        #        print('TEST LEVELS: ', num_slots)

        else:
            print(self.args.VARIABLES)
            print('Generating pool..')
            self.we.generator.generate_pool(self.args.num_equations)  # , self.seed)
            pool = self.we.generator.pool
            self.pool_generation_time = self.we.generator.pool_generation_time
            print('pool generated')

        level_order = [[eq.level, i] for i, eq in enumerate(pool)]
        level_order.sort(reverse=True)
        pool = [pool[lvl[1]] for lvl in level_order]

        for i, eq in enumerate(pool):
            eq_original=eq.w

            self.mcts = MCTS(self.nnet, self.args, self.args.num_mcts_simulations, self.mode, self.seed)
            self.mcts.name = self.name

            initial_local_time = time.time()
            eq = self.we.transformations.normal_form(eq)
            examples, log = self.execute_episode(eq)

            local_execution_time = round(time.time() - initial_local_time, 4)

            len_eq = len(log)
            printout_info = [len_eq, None, i, local_execution_time,
                             round(self.args.timeout_time,2), self.args.num_mcts_simulations, log]

            if examples[0][2] >= self.args.sat_value  or log[-1] == 'early_sol':
                self.execution_times.append(local_execution_time)
                self.score.append(1)
                printout_info[1] = 'New sol'
                logging.info(self.execution_printout(*printout_info))
                self.num_actions_taken_if_successful.append(len_eq)
                if self.args.active_tester:
                    self.eqs_solved.append([eq.level, eq_original, local_execution_time])

            elif  examples[0][2] == self.args.unknown_value:
                self.score.append(0)
                printout_info[1] = 'Stopped'
                logging.info(self.execution_printout(*printout_info))

            elif examples[0][2] == self.args.unsat_value:
                self.score.append(0)
                printout_info[1] = 'Fail'
                logging.info(self.execution_printout(*printout_info))

            if log[0] != 'already_solved':
                self.train_examples += examples


    def execute_episode(self, eq=None):
        """
        Attempts to solve the equation 'eq' (i.e.\ an episode)
        Returns the episode data described in the paper
        """
        self.mcts.initial_time = time.time()

        train_examples = []
        episode_step = 0
        num_repetitions = 0

        log = [eq.w]
        t = time.time()
        eq.s_eq=None
        eq.sat = self.args.unknown_value
        eq.initial_state = True
        prev_state=eq.deepcopy(copy_id = True)
        prev_state.initial_state = True
        self.mcts.is_leaf[eq.w] = True
        self.mcts.is_leaf[prev_state.w] = True

        while True:
            if not self.args.use_oracle:
                s_eq = self.we.utils.format_state(eq)
            else:
                s_eq = None
            eq.s_eq = s_eq
            pi = self.mcts.get_action_prob(eq, s_eq, temp=self.args.temperature, previous_state=prev_state)
            if pi == 'already_sat':
                print(pi, eq)
                return [(0,0,1)], ['already_solved']
            examples = [(s_eq, pi, None)]
            train_examples += examples

            action = np.random.choice(len(pi), p=np.array(pi))

            prev_state = eq.deepcopy(copy_id = True)

            new_eq = self.we.moves.act(eq, action)
            eq = new_eq

            episode_step += 1
            tt = round(time.time() - t, 2)
            log += [eq.get_string_form_for_print()]

            if self.timeout(tt):
                r =  self.args.evaluation_after_exceeded_steps_or_time
                return [(x[0], x[1], r*(self.args.discount**i)) for i, x in enumerate(train_examples[::-1])], log

            if self.mcts.found_sol[0]:
                if self.mode == 'test' or self.args.oracle:
                    print('found early solution')
                    return [(x[0], x[1], ((self.args.discount**(i))*self.args.sat_value)) for i,x in enumerate(train_examples[::-1])],  log

            if eq.w in self.mcts.final_state_value:
                eq.sat = self.mcts.final_state_value[eq.w]
            else:
                eq = self.we.utils.check_satisfiability(eq, smt_time=self.args.mcts_smt_time_max)

            if eq.sat != self.args.unknown_value:
                if eq.sat == self.args.sat_value:
                    return [(x[0], x[1], (self.args.discount**i)*self.args.sat_value) for i, x in enumerate(train_examples[::-1])], log
                else:
                    return [(x[0], x[1], (self.args.discount**i)*self.args.unsat_value) for i, x in enumerate(train_examples[::-1])], log



    def execution_printout(self, steps,outcome, num_eqs_attempted, local_execution_time, timout_time, num_mcts, log =[]):
        """
        For logging
        """
        return f'{self.name} - {outcome} - Eq: ' \
            f'{self.get_score()}/{num_eqs_attempted + 1}/{self.args.num_equations}: ' \
            f'Steps {steps}, Time: {local_execution_time},' \
            f' Timeout (episode/mcts): {timout_time},' \
            f' Num mcts sims: {num_mcts},  {log}'

    def get_score(self):
        return sum(self.score)
