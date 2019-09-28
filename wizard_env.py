import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from main_simple import Card, Game
import multiprocessing

class Env(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num Observation                 Min         Max
        0   Cart Position             -4.8            4.8
        1   Cart Velocity             -Inf            Inf
        2   Pole Angle                 -24 deg        24 deg
        3   Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num Action
        0   Push cart to the left
        1   Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # 0 = white
        # 1 = red
        # 2 = blue
        # 3 = green
        # 4 = yellow
        # -1 - 14 = values (-1 = no card there)
        cards_space = spaces.Box(low = np.array([0,-1]), high = np.array([4,14]))
        score_space = spaces.Box(low = np.array([0]), high = np.array([100]))

        self.observation_space = spaces.Tuple(
            [score_space, #my score
            score_space, #enemy score
            score_space, #my trick guess
            score_space, #enemy trick guess
            cards_space, #enemies first card
            cards_space, #enemies second card
            cards_space, #my first card
            cards_space, #my second card
            cards_space, #first card in trick, only 2 players at max
            ]
        )

        self.action_space = spaces.Discrete(2)

        self.seed()
        self.reset()

    def seed(self, seed=None): ##NOOP for now
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def recv_state(self):
        self.state = self.my_pipe_end.recv()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        print(f"sending action {action}")
        self.my_pipe_end.send(action)
        self.recv_state()
        print(f"recieved state {self.state}")
        reward = 0
        done = False
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.my_pipe_end, other_pipe_end = multiprocessing.Pipe()
        self.game_process = multiprocessing.Process(target = Game(2, [0], other_pipe_end).play)
        self.game_process.start()
        self.recv_state()

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

