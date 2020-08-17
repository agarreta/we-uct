import random
import torch
import torch.nn.functional as F

from utils import *
# from OpenAi

class Env(object):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return

    @property
    def unwrapped(self):
        """Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        self.close()
        # propagate exception
        return False



class DigitEnv(Env):
    def __init__(self, data, classifier_targ, knowledge_targ=None, acc=None , alpha=None):
        super(DigitEnv, self).__init__()
        self.initial_observations = data
        self.action_space=None
        self.observation_space=None
        self.state = None
        self.mu = None
        self.sigma = None
        self.classifier = classifier_targ
        self.knowledge_targ = knowledge_targ
        self.acc = acc
        self.alpha= alpha

    def sample_action(self):
        pass

    def compute_reward(self, state_1, state_2, label, mu1, sigma1):


        with torch.no_grad():
            acc2 = absolute_prediction(state_2, self.classifier, label)
            acc1 = absolute_prediction(state_1, self.classifier, label)

        #c1, c2 = self.acc.classifier(state_1), self.acc.classifier(state_2)
        kl =1* ( torch.min(torch.abs(acc2 -acc1), torch.ones(acc2.shape, dtype=torch.float).to('cuda:0'))) # KL_div(c1, c2)
        ce2 = F.cross_entropy(self.acc.classifier(state_2), label.long())

        #kk=(ce2-2.3026)/torch.abs(ce2-2.3026)
        k =ce2# 100*ce2*(kk)

        l =1-  torch.clamp(1 * (k - 0), -1., 1.).mean()
        #l = torch.clamp(10e1*((self.acc.randomnet(state_2) - self.acc.knowledge(state_2))**2).mean(-1),0.,1.)
        _, logpi, mu2, sigma2 = self.acc.actor(state_2)

        reward =  (1-self.alpha)*l -self.alpha*torch.clamp(logpi, -1., 0.)

        if random.random() < 0.008: #and kl.shape[0]>1:
            print('kl', kl.mean())
            print('know', l.mean())
            print('logpi', logpi.mean())

        reward= torch.clamp(reward, -1., 1.)
            #reward[acc<1] = -1.

        reward = reward.mean()

        return reward, kl


    def step(self, action, mu2=None, sigma2=None, label=None):
        with torch.no_grad():

            rew,  dist = self.compute_reward(self.state, action, label, self.mu, self.sigma)
        self.state = action
        self.mu = mu2
        self.sigma = sigma2
        d =  False if dist < 0. else False
        return self.state, rew, d, ''

    def reset(self):
        i = random.sample(list(range(len(self.initial_observations[0]))),k=1)
        o = self.initial_observations[0][i]
        label = self.initial_observations[1][i]
        self.state = o
        self.state.requires_grad_(False)
        with torch.no_grad():
            _, _, mu, sigma= self.acc.actor(o)
        self.mu, self.sigma = mu, sigma
        return o, label