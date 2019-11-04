import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from main_simple import Card, Game
import multiprocessing
import matplotlib.pyplot as plt
import random

def _print(*args, **kwargs):
    # return print(*args, **kwargs)
    return

class Env(gym.Env):

    def __init__(self, average_over = 50, visualize = False):
        # 0 = white
        # 1 = red
        # 2 = blue
        # 3 = green
        # 4 = yellow
        # -1 - 14 = values (-1 = no card there)
        self.visualize = visualize
        cards_space = spaces.Box(low = np.array([0,-1]), high = np.array([4,14]))
        score_space = spaces.Box(low = np.array([0]), high = np.array([100]))
        # TODO make player space to abstract further

        self.observation_space = spaces.Tuple(
            [score_space, # enemies score
             score_space, # enemies trick guess
             cards_space, # enemies first card
             cards_space, # enemies second card
             #
             score_space, # my score
             score_space, # my trick guess
             cards_space, # my first card
             cards_space, # my second card
             #
             cards_space, #first card in trick, lets play 2 players first
             cards_space, #second card in trick
            ]
        )

        self.i = 0

        self.action_space = spaces.Discrete(3)

        self.scores = []
        self.last_score = 0

        self.average_over = average_over
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'x', lw=1, label='1 game')
        self.avgline, = self.ax.plot(
        [], [], lw=2, label='{} game average'.format(self.average_over))
        self.ax.set_xlabel('games')
        self.ax.set_ylabel('score')
        self.ax.legend(loc='best', ncol=2)
        self.lasttile = 1
        plt.show(block=False)

        self.seed()
        self.reset()


    def replot(self):
        self.i += 1
        self.line.set_data(range(len(self.scores)), self.scores)

        if len(self.scores) % self.average_over == 0:
            averages = np.mean(
                np.array(self.average_over * [0] + self.scores).reshape(
                    -1, self.average_over),
                axis=1)
            self.avgline.set_data(
                self.average_over * np.array(range(len(averages))), averages)
            self.fig.savefig('./wurst.pdf')

        self.ax.set_xlim(0, len(self.scores))
        self.ax.set_ylim(-5, 5)
        self.fig.canvas.draw()

    def seed(self, seed=None): # TODO NOOP for now
        return 42
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]

    def recv_state(self):
        state = self.my_pipe_end.recv()
        self.observation = np.array(state[:-2])
        self.round_done = state[-2]
        self.game_done = state[-1]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.my_pipe_end.send(action)
        self.recv_state()
        _print(f"recieved state {self.observation}")
        def calculate_reward():
            nothing_changed = all(self.last_observation == self.observation)
            self.last_observation = self.observation
            if self.round_done or self.game_done:
                # my score - opponents score
                reward = self.observation[6]# - self.observation[0]# - self.last_scores #TODO maybe but it back in
                self.last_score = reward
                return reward
            else:
                return -1*nothing_changed

        reward = calculate_reward()
        _print("reward: ", reward)
        return self.observation, reward, self.game_done, {}

    def reset(self):
        if self.visualize:
            self.scores.append(self.last_score)
            self.replot()
        self.my_pipe_end, other_pipe_end = multiprocessing.Pipe()
        multiprocessing.Process(target = Game(2, [0], other_pipe_end, print_function = _print).play).start()
        self.recv_state()
        self.last_score = 0
        self.last_observation = 0
        self.round_done = False
        self.game_done = False
        return self.observation

    def render(self, mode='human'):
        raise NotImplementedError('this does not exist yet')

    def close(self):
        pass

