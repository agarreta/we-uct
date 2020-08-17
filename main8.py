
import time
import pandas as pd
import math
import os
import platform
from replay_buffer import ReplayBuffer
from utils import EpochLogger, entropy, KL_div, show_im, preprocess, accuracy
from networks import *
from copy import deepcopy
from env import DigitEnv
import itertools
import seaborn as sbn
import torch.nn.functional as F

import matplotlib.pyplot as plt
    #plt.ylim(0.9,1.)
plt.xlim(2000)

digits = [5,6]
batch_size = 100
num_samples_per_digit = 150
num_epochs = 130
num_channels = 8
num_actor_channels = 16
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
l2_regularization =0.
max_num_steps=10

def ce_min(p1,p2,labs,mode='ce'):
    return torch.min(p1, p2)
    if False:#mode == 'mean':
        return 0.5*(p1+p2)
    p=torch.cat([p1,p2])
    l = torch.cat([labs,labs])
    if  mode == 'ce':
        ce = F.cross_entropy(p,l.long(), reduction='none')
    else:
        ce = torch.randn(p.shape[0])# random.choice([p1, p2])
        #ce =entropy(p)
    ce1 = ce[:p1.shape[0]]
    ce2 = ce[p1.shape[0]:]
    c = torch.stack([ce1, ce2])
    idx = torch.min(c, dim=0)[1]
    q = torch.stack([p1,p2]).transpose(0,1)
    return q[[list(range(q.shape[0])), idx]]


def seed_everything(seed):
    # https://www.kaggle.com/hmendonca/fold1h4r3-arcenetb4-2-256px-rcic-lb-0-9759 cells 45-50
    # print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(10)


class MDP(object):
    def __init__(self, initial_data, test_data, normal_data):
        super(MDP, self).__init__()
        #self.discount_ext = 0.975
        self.discount = 0.99
        self.max_episode_time = 30
        self.lr =1e-3
        self.polyak = 0.995
        self.polyak_actor_output = 0.9# 1 - 1/(max_num_steps-1)
        self.replay_size = int(1e4)
        self.init_obs_size = int(1e3)
        self.max_num_steps = max_num_steps
        self.obs_dim = (1,28,28)
        self.act_dim = (1,28,28)
        self.act_u_limit = 1.
        self.act_l_limit = 0.
        self.action_noise = 0.001
        self.target_noise= 0.01
        self.noise_clip=  0.1
        self.steps_per_epoch = 1000
        self.epochs = 1000
        self.start_steps = -1 #1e4
        self.update_after =200
        self.update_every = 200
        self.save_freq = 1
        self.display_stats_frequency = 0.001
        self.epsilon = 1e-25
        self.multiply_frequency = 1.
        self.test_normals = False
        self.dropout = 0.
        self.use_twins = True
        self.clamp_params =not True
        self.alpha = .5
        self.use_class_net_in_pi=False
        self.num_blocks = 1
        #print(locals())

        #self.logger.save_config(locals()) #TODO: make work

        self.max_acc1=0
        self.true_max_acc=0
        self.max_acc2=0
        self.max_acc3=0
        self.max_acc_vfun=0
        self.max_acc_normal=0
        self.max_acc_mini_normal=0
        self.num_non_imrovements = 0
        self.test_history1=[]
        self.test_history2=[]
        self.test_history3=[]


        self.X_test = test_data[0]
        self.y_test = test_data[1]
        self.X_normal = normal_data[0]
        self.y_normal = normal_data[1]
        self.X_mini_normal = initial_data[0]
        self.y_mini_normal = initial_data[1]
        self.first_steps = [[],[]]

        #self.classifier = Classifier(num_channels,1,self.dropout, None).to(device) # Rfun(self.dimension, self.num_actions).to(device)
        #self.classifier_targ = deepcopy(self.classifier).to(device)
        #self.knowledge = Classifier(num_channels, 1, self.dropout, None).to(device)
        #self.knowledge = deepcopy(Classifier(num_channels, 1, self.dropout, None)).to(device)
       # self.vfun =  NormalModel(8, 2).to(device)

        self.acc = ActorCritic(num_channels,num_actor_channels,self.dropout, polyak=self.polyak_actor_output,
                               max_num_steps=self.max_num_steps,
                               class_net=self.use_class_net_in_pi, num_blocks=self.num_blocks).to(device)
        self.acc_targ = deepcopy(self.acc).to(device)
        self.env = DigitEnv(initial_data, self.acc_targ.classifier, self.acc_targ.knowledge, acc= self.acc, alpha=self.alpha)  #TODO: update classifier params here too after optimization?

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.acc_targ.parameters():
            p.requires_grad = False
        for p in self.acc.randomnet.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.acc.qfun1.parameters(), self.acc.qfun2.parameters())

        self.best_acc = 0
        self.best_epoch=0
        self.normal_best_acc = 0
        self.normal_mini_best_acc = 0
        self.normal_model = NormalModel(num_channels, 10, self.dropout).to(device)
        self.normal_mini_model = NormalModel(num_channels,10, self.dropout).to(device)

        self.optimizer_qfun = torch.optim.Adam(params=self.q_params, lr=self.lr, weight_decay=l2_regularization)
        self.optimizer_actor = torch.optim.Adam(params=self.acc.actor.parameters(), lr=self.lr, weight_decay=l2_regularization)
        self.optimizer_classifier = torch.optim.Adam(params=self.acc.classifier.parameters(), lr=self.lr, weight_decay=l2_regularization)
        self.optimizer_knowledge = torch.optim.Adam(params=self.acc.knowledge.parameters(), lr=self.lr, weight_decay=l2_regularization)
        self.optimizer_normal = torch.optim.Adam(params=self.normal_model.parameters(), lr=self.lr, weight_decay=l2_regularization)
        self.optimizer_normal_mini = torch.optim.Adam(params=self.normal_mini_model.parameters(), lr=self.lr, weight_decay=l2_regularization)
        #self.optimizer_vfun = torch.optim.Adam(params=self.vfun.parameters(), lr=self.lr, weight_decay=l2_regularization)

        self.log_folder = f'logs{time.time()}'
        self.logger = EpochLogger(output_dir=self.log_folder, output_fname='progress.txt', exp_name=None)

        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size, device='cpu')
        self.logger.setup_pytorch_saver([self.acc,  self.normal_model, self.normal_mini_model, self.acc_targ])

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self,data, step):

        o, a, r, o2, d, labels = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['labels']
        o.detach_(), a.detach_()
        v1 = self.acc.qfun1(o)
        if self.use_twins:
            v2 = self.acc.qfun2(o)

        # Bellman backup for Q function
        with torch.no_grad():
            a2, logp_pi, _, _ = self.acc.actor(o)

            # Target Q-values
            q1_pi_targ = self.acc_targ.qfun1(a2)
            if self.use_twins:
                q2_pi_targ = self.acc_targ.qfun2(a2)
                q_pi_targ = ce_min(q1_pi_targ, q2_pi_targ, labels)
            else:
                q_pi_targ = q1_pi_targ

            #r_new = self.env.compute_reward(o, a2, labels)
            backup = r  + self.discount * (1-d)*  q_pi_targ

        loss_q1 = (v1 - backup)**2
        self.logger.store_rlogs(qval=float(v1.mean().cpu().detach().numpy()))

        #loss_q1_=torch.nn.functional.cross_entropy(torch.softmax(v1, dim=-1) + self.epsilon,labels.long())

        loss_q = loss_q1.mean()

        if self.use_twins:
            loss_q2 = (v2 - backup)**2
            #loss_q2_=torch.nn.functional.cross_entropy(torch.softmax(v2, dim=-1) + self.epsilon,labels.long())
            loss_q += loss_q2.mean()

        loss_q = loss_q.mean()#+ (bound_q1.mean() + bound_q2.mean())

        if torch.isnan(loss_q):
            print(v1+self.epsilon)
            #print(torch.softmax(v2+self.epsilon, dim=-1))
            print(backup+self.epsilon)
            print(loss_q1)
            print(loss_q2)
            raise Exception

        if random.random() < 1*self.display_stats_frequency and  r[0] >0:
            print('reward', r.mean(0))
            print('q1', v1.mean(0))
            #print('knowledge', m1.mean(0))
            #print('classifier', t1.mean(0))
            #print('q2', v2.mean(0))
            show_im(o, labels, self.log_folder)


        # Useful info for logging
        self.logger.store(Q1Vals1=v1.cpu().detach().numpy(),
                         Rewards=r.cpu().detach().numpy(),
                         )

        # AG: we need to empty the gradients for the next optimization steps
        o.detach_(), a.detach_(), o2.detach_(), r.detach_(), q_pi_targ.detach_()
        return loss_q#, loss_vfun

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o,labels, d= data['obs'], data['labels'], data['done']
        #pi, raw_pi, log_pi=self.acc.actor(o)
        pi, log_pi, mu, sigma=self.acc.actor(o)
        qval1=self.acc_targ.qfun1(pi)
        qval2=self.acc_targ.qfun2(pi)
        r, _ = self.env.compute_reward(o, pi, labels, mu, sigma)
        qval = self.discount*(1-d)*torch.min(qval1, qval2)
        #loss1 = ((raw_a**2).view(raw_a.shape[0], -1).max(-1)[0]-6)**2

        loss = qval #+loss1
        loss_pi = -self.discount*( loss ).mean()-r.mean()#self.alpha*torch.clamp(log_pi,-1,+1).mean()


        self.logger.store(log_pi=log_pi.mean().cpu().detach().numpy() )
        self.logger.store_rlogs(log_pi=float(r.mean().cpu().detach().numpy()))
        self.logger.store_rlogs(loss_pi_q=float(loss.mean().cpu().detach().numpy()))

        #if random.random()< 0.01:
        #    print(q_pi, (a).mean())
        # AG: we need to empty the gradients for the next optimization steps
        o.detach_(), pi.detach_()
        return loss_pi


    def get_action(self, o, noise_scale):
        with torch.torch.no_grad():
            a, log_pi,mu,sigma = self.acc.actor(o)
        a.detach_()
        #a += (noise_scale * torch.randn(self.act_dim)).to(device)
        return a, log_pi, mu, sigma #torch.clamp(a, -self.act_l_limit, self.act_u_limit), log_pi

    def update(self, data, step):
        for p in self.acc.parameters():
            p.requires_grad = True
        for p in self.acc.randomnet.parameters():
            p.requires_grad = False
        self.acc.train()
        self.acc_targ.train()

        if step % 1 == 0:

            # First run one gradient descent step for Q.
            self.optimizer_qfun.zero_grad()

            loss_q = self.compute_loss_q(data, step)

            loss_q.backward()
            if self.clamp_params:
                for param in self.q_params:
                    param.grad.data.clamp_(-1,1)
            self.optimizer_qfun.step()

            self.logger.store(LossQ=loss_q.item())
            self.logger.store_rlogs(LossQ=loss_q.item())

        if step%1==0:#(step+ self.update_every) % (2*self.update_every) ==0:
            self.optimizer_classifier.zero_grad()
            o, labels,r = data['obs'], data['labels'], data['rew']
            idx = (r > 0.).view(-1)
            o =o[idx]
            labels =labels[idx]
            loss_knowledge = torch.tensor([0]).to(device)
            loss_classifier = F.cross_entropy(self.acc.classifier(o) , labels.long())
            loss_classifier.backward()
            if self.clamp_params:
                for param in self.acc.classifier.parameters():
                    param.grad.data.clamp_(-1,1)
            self.optimizer_classifier.step()

        else:
            loss_classifier=torch.tensor([-1])

        # Record things
        self.logger.store_rlogs(LossClass=loss_classifier.item())
        self.logger.store(LossClass=loss_classifier.item())

        if (step+ self.update_every) % (2*self.update_every) ==0:
            for p in self.acc.knowledge.parameters():
                p.requires_grad = True
            self.optimizer_knowledge.zero_grad()
            o = data['obs']
            loss_know = KL_div(self.acc.knowledge(o), self.acc_targ.randomnet(o))
            loss_know.backward()
            if self.clamp_params:
                for param in self.acc.knowledge.parameters():
                    param.grad.data.clamp_(-1,1)
            self.optimizer_knowledge.step()
            # Record things
        else:
            loss_know = torch.tensor([-1.])
        self.logger.store_rlogs(LossKnow=loss_know.item())
        self.logger.store(LossKnow=loss_know.item())

        if step %1 == 0:
            # Freeze Q-network so you don't waste computational effort
            # computing gradients for it during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.optimizer_actor.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            if self.clamp_params:
                for param in self.acc.actor.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.optimizer_actor.step()

            # Record things
            self.logger.store(#LossQ=loss_q.item(),
                              LossPi=loss_pi.item(),
                              #LossClass=loss_classifier.item(),
                              #LossKnow = loss_knowledge.item(),
                              #**loss_info
            )
            self.logger.store_rlogs(LossPi=loss_pi.item())


        if random.random() < self.display_stats_frequency:
            try:
                print('loss q', loss_q)
                print('loss pi', loss_pi)
                print('loss class', loss_classifier)
                print('loss know', loss_know)
                #print('loss vfun', loss_vfun)
            except:
                pass

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.acc.parameters(), self.acc_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        for p in self.acc.parameters():
            p.requires_grad = False

        self.acc.eval()
        self.acc_targ.eval()

    def test_agent(self):
        def main_fun(X, y, model, labels=None):
            model.eval()
            answers = []
            for i, batch in enumerate([X[32 * i: 32 * (i + 1)] for i in range(math.ceil(len(X) / 32))]):
                #out = torch.softmax(model(batch), dim=-1)
                #answer = [int(np.argmax(out[i].cpu().detach().numpy())) for i in range(len(out))]
                answers += accuracy(batch, model, y[32 * i :32* (i + 1)])
            acc =sum(answers).detach().cpu().detach().numpy() / len(X)
            model.train()
            return acc

        acc = main_fun(self.X_test, self.y_test, self.acc.classifier)

        if acc >self.max_acc1:
            self.max_acc1 = acc
            self.num_non_imrovements =0

        self.logger.store(Test_acc1=acc)

        self.test_history1.append(acc)
        sbn.lineplot(data=pd.Series(self.test_history1))
        sbn.lineplot(data=pd.Series(self.test_history2))
        plt.savefig(self.log_folder + '/fig2.png')
        plt.show()


        if self.test_normals:

            self.normal_model.train()
            self.normal_mini_model.train()
            for i, batch in enumerate([self.X_normal[32 * i: 32 * (i + 1)] for i in range(math.ceil(len(self.X_normal) / 32))]):
                out = self.normal_model(batch)
                loss = nn.functional.cross_entropy(out, self.y_normal[32 * i: 32 * (i + 1)])
                loss.backward()
                self.optimizer_normal.step()
                self.optimizer_normal.zero_grad()


            acc = main_fun(self.X_test, y_test, self.normal_model)
            if acc > self.max_acc_normal:
                self.max_acc_normal = acc
            self.logger.store(Test_acc_normal=acc)


            for i, batch in enumerate([self.X_mini_normal[32 * i: 32 * (i + 1)] for i in range(math.ceil(len(self.X_mini_normal) / 32))]):
                out = self.normal_mini_model(batch)
                loss = nn.functional.cross_entropy(out, self.y_mini_normal[32 * i: 32 * (i + 1)])
                loss.backward()
                self.optimizer_normal_mini.step()
                self.optimizer_normal_mini.zero_grad()

            acc = main_fun(self.X_test, y_test, self.normal_mini_model)
            if acc > self.max_acc_mini_normal:
                self.max_acc_mini_normal = acc
            self.logger.store(Test_acc_normal_mini=acc)
            self.logger.store(Test_acc_normal_max=self.max_acc_normal)
            self.logger.store(Test_acc_normal_mini_max=self.max_acc_mini_normal)


        self.logger.store(Test_acc1_max=self.max_acc1)
        self.logger.store(True_max_acc=self.true_max_acc)

        self.logger.store(Test_acc2_max=self.max_acc2)
        self.logger.store(Test_acc3_max=self.max_acc3)
        #self.logger.store(Test_acc_Vfun_max=self.max_acc_vfun)



    def interact_with_environment(self):
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        o, label, ep_ret, ep_len = *self.env.reset(), 0, 0


        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            self.acc.eval()
            self.acc_targ.eval()
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).

            if t > self.start_steps:
                a, logpi, mu, sigma = self.get_action(o, self.action_noise)
            else:
                a = self.env.sample_action()

            # Step the env
            o2, r, d, _ = self.env.step(a, mu, sigma, label)
            ep_ret += r
            ep_len += 1

            if ep_len == 1 and random.random() <self.multiply_frequency:
                self.first_steps[0].append(o2)
                self.first_steps[1].append(label)


            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            #d = False if ep_len == self.max_num_steps else d

            #if random.random() < 0.01:
            #    print(r)

            # Store experience to replay buffer

            if r > 0. or t <=10:
                self.replay_buffer.store(o, a, r, o2, d, label)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_num_steps):
                self.logger.store(EpRet=ep_ret.item(), EpLen=ep_len)
                o, label, ep_ret, ep_len = *self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(batch_size, device)
                    self.update(data=batch, step=t)
                if self.multiply_frequency > 0. and len(self.first_steps[0])>0:
                    ini= self.env.initial_observations
                    X, y = ini[0], ini[1]
                    X_new, y_new = torch.cat(self.first_steps[0]), torch.cat(self.first_steps[1])
                    X = torch.cat([preprocess(X), preprocess(X_new)])
                    y = torch.cat([y, y_new])

                    self.env.initial_observations = (X,y)
                    self.first_steps = [[], []]
                    self.logger.store(InitObs=len(self.env.initial_observations[0]))

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()
                self.logger.show_rlogs(self.log_folder)
                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.store(Replay_buffer_idx=self.replay_buffer.ptr)
                self.logger.log_tabular('Replay_buffer_idx',  average_only=True)
                if False:# self.use_twins:
                    self.logger.log_tabular('Q2Vals1', with_min_and_max=True)
                    self.logger.log_tabular('Q2Vals2', with_min_and_max=True)
                self.logger.log_tabular('Rewards', with_min_and_max=True)
                self.logger.log_tabular('log_pi', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('LossClass', average_only=True)
                self.logger.log_tabular('LossKnow', average_only=True)
                self.logger.log_tabular('Test_acc1', average_only=True)
                if self.test_normals:
                    self.logger.log_tabular('Test_acc_normal', average_only=True)
                    self.logger.log_tabular('Test_acc_normal_mini', average_only=True)
                self.logger.log_tabular('Test_acc1_max', average_only=True)
                self.logger.log_tabular('True_max_acc', average_only=True)

                if self.test_normals:
                    self.logger.log_tabular('Test_acc_normal_max', average_only=True)
                    self.logger.log_tabular('Test_acc_normal_mini_max', average_only=True)
                self.logger.log_tabular('Time', time.time() - start_time)
                self.logger.dump_tabular()


def shuffle_df(df):
    idx = list(range(df.shape[0]))
    random.shuffle(idx)
    df = df.iloc[idx, :]
    return df


if __name__ == '__main__':
    num = 3
    dir = 'fashion-mnist_train.csv' if (
                'albert' in platform.node() or 'wordeq' in platform.node()) else '../input/digit-recognizer/train.csv'
    dir = 'train.csv'
    ds = pd.read_csv(dir)


    def preprocess_data(ds, range_):
        def main (ds_all,ds_current,num, range_):
            ds0 = ds_all[ds_all['label'] == num]
            range_ = range(min(len(range_), ds0.shape[0]))
            ds0 = ds0.iloc[range_, :]
            ds0 = shuffle_df(ds0)
            if ds_current is not None:
                return pd.concat([ds_current, ds0])
            else:
                return ds0

        ds_new = None
        for num in range(10):
            ds_new = main(ds, ds_new, num, range_)

        ds_ = shuffle_df(ds_new)
        X = torch.tensor(np.array(ds_.iloc[:, 1:]), dtype=torch.float).view(ds_.shape[0], 1, 28, 28).to(device) / 255
        y = torch.tensor(np.array(ds_.iloc[:, 0])).to(device)
        return X, y


    X, y = preprocess_data(ds, range(num_samples_per_digit))

    X_test, y_test = preprocess_data(ds, range(4700, ds.shape[0]))
    X_normal_train, y_normal_train = preprocess_data(ds, range(ds.shape[0]))
    print('Normal database size: 10x', X_normal_train.shape[0])

    mdp = MDP((X, y), (X_test, y_test), (X_normal_train, y_normal_train))
    mdp.interact_with_environment()