
from we import *
from we.word_equation.we import WE
from we.arguments import Arguments
from we.utils import Utils
import os
import gc
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.set(style="white", palette='muted')
import numpy as np

class Plotter(object):
    def __init__(self, args = None, folder_name = None, filename = None, load=False):
        if args is not None:
            self.args = args
        elif False: #filename is not None:
            args = Arguments()
            args.load_model = False
            utils = Utils(args)
            args.folder_name = 'we/test/' + filename
            self.args = utils.load_object('arguments')
        else:
            raise ValueError
        #self.filename = filename
        self.folder_name = folder_name  #+ '/evaluation_' + str(self.filename)
        self.filename = filename
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)        #self.plot_steps_per_level()
        #self.plot_percentages_per_level()

        self.scores = pd.DataFrame(columns=['level', 'solver', 'score'])
        self.num_steps = pd.DataFrame(columns=['level', 'solver', 'num_steps'])
        self.time = pd.DataFrame(columns=['level', 'solver', 'time'])
        self.aggregated_scores = []
        self.eqs_solved =[]
        self.eqs_solved_Z3 = []

    @staticmethod
    def get_dataframe_steps(log, df, type_df, model_name):
        for entry in log:
            for level, y in enumerate(entry):
                for steps in y:
                    if steps is not None:
                        if  steps > 0 or type_df == 'score':
                            df = df.append(
                                pd.DataFrame([[int(level + 1), model_name, float(steps)]], columns=['level', 'solver', type_df]))
        df.index = range(df.shape[0])
        return df

    def plot_steps_per_level(self, model_name, plot):
        self.num_steps = self.get_dataframe_steps(self.args.sat_steps_taken_in_level_test,
                                                  self.num_steps, 'num_steps', model_name)
        if False: # len(self.num_steps['num_steps'].value_counts()) > 1:
            plt.close()
            plt.title('Number of actions taken in solved equations (non Z3)')
            sbn.swarmplot(data=self.num_steps, x='level', y='num_steps', hue = 'solver')  # , hue = 'type')
            plt.savefig(f'{self.folder_name}/number_actions_total_{self.filename}')
        if False: #plot:
            plt.close()
            plt.title('Average number of actions taken in solved equations (non Z3)')
            sbn.lineplot(data=self.num_steps, x='level', y='num_steps', hue='solver')  # , hue = 'type')
            plt.savefig(f'{self.folder_name}/number_actions_average_{self.filename}')

    def plot_times_per_level(self, model_name, plot):
        self.time = self.get_dataframe_steps(self.args.current_level_sat_times_test, self.time, 'time', model_name)
        if model_name[-1]=='Z':
            self.time = self.get_dataframe_steps(self.args.z3mctstime_sat_test, self.time, 'time', 'z3str3_in_' + model_name)

        if False: #plot:
            plt.close()
            plt.title('Average seconds taken in solved equations')
            sbn.lineplot(data=self.time, x='level', y='time', hue='solver')  # , hue = 'type')
            plt.savefig(f'{self.folder_name}/number_seconds_lineplot_{self.filename}')
            plt.close()

    @staticmethod
    def get_dataframe_percentage(log):
        df = pd.DataFrame(columns=['level', 'action', 'percentage'], dtype=int)
        action_dict = ['delete', 'compress', 'pop']
        for entry in log:
            for i, action in enumerate(entry):
                for level, percentage in enumerate(action):
                    if type(percentage) == float:
                        df = df.append(pd.DataFrame([[int(level+1), action_dict[i], float(percentage)]], columns=df.columns))
        df.index = range(df.shape[0])
        return df

    def plot_percentages_per_level(self, plot):
        pr = self.get_dataframe_percentage(self.args.percentage_of_actions_this_level_test)
        if pr.shape[0] == 0:
            pr.append(pd.DataFrame([[5,0,'test']], columns = pr.columns))
        if False: #plot:
            plt.close()
            plt.title('Percentage of actions taken in solved equations')
            sbn.lineplot(data=pr, x='level', y='percentage', hue='action', markers=True, dashes=False, style='action') #, hue = 'type')
            plt.savefig(f'percentage_actions_{self.filename}')

    def plot_scores(self, model_name, plot):
        self.scores = self.get_dataframe_steps(self.args.score_test_this_level, self.scores, 'score', model_name)
        if model_name[-1]=='Z':
            self.scores = self.get_dataframe_steps(self.args.z3score_level_test, self.scores, 'score', 'z3str3_in_' + model_name)

        m = self.scores.groupby(['solver']).mean()
        m['solver'] = m.index
        m.index = range(m.shape[0])
        m['level'] = 0
        self.aggregated_scores.append(m)  # .reset_index(drop=True)
        print(m)

        if False: #plot:
            plt.close()
            plt.title('Score')
            plt.figure(figsize=(10, 5))
            sbn.barplot(data=self.scores, x='level', y='score', hue='solver', ci=None)
            plt.savefig(f'{self.folder_name}/scores_{self.filename}')
        print(self.aggregated_scores)
        # self.scores = self.scores.iloc[:-1,:]

    def plot(self, model_name, plot = True):
        self.plot_steps_per_level(model_name, plot)
        self.plot_times_per_level(model_name, plot)
        # self.plot_steps_per_level(model_name, plot)
        self.plot_scores(model_name, plot)


if False: #__name__ == '__main__':
    # p = Plotter(filename='resnet00002lr100max25chunk256ch20layersmcts_-1_steps_episode_-1_time_0_clears')
    p = Plotter(filename='resnet00002lr100max25chunk256ch20layersmcts_-1_steps_episode_-1_time_0_clears/evaluation')
    p.plot_steps_per_level()
    p.plot_percentages_per_level()
    p.plot_times_per_level()
    # p.args.fails_train_this_level
