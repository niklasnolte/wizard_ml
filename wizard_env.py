import math
import gym
from gym import spaces, logger
import numpy as np
from main_simple import Card, Game
import multiprocessing
import matplotlib.pyplot as plt
import random
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from collections.abc import Iterable


def _print(*args, **kwargs):
    return print(*args, **kwargs)
    # return


class Env(py_environment.PyEnvironment):
    def __init__(self, average_over=50, visualize=False):
        # 0 = white
        # 1 = red
        # 2 = blue
        # 3 = green
        # 4 = yellow
        # -1 - 14 = values (-1 = no card there)
        self.visualize = visualize
        cards_spec = lambda name: array_spec.BoundedArraySpec(
            shape=(2,), minimum=[0, -1], maximum=[4, 14], name=name, dtype=np.int32
        )
        score_spec = lambda name: array_spec.BoundedArraySpec(
            shape=(1,), minimum=-100, maximum=100, name=name, dtype=np.int32
        )
        # TODO make player space to abstract further

        self.observation_keys = (
            "enemy_score",
            "enemy_trick_guess",
            "enemy_first_card",
            "enemy_second_card",
            "my_score",
            "my_trick_guess",
            "my_first_card",
            "my_second_card",
            "first_in_trick",
            "second_in_trick",
        )

        self.observation_specs = [
            score_spec,
            score_spec,
            cards_spec,
            cards_spec,
            score_spec,
            score_spec,
            cards_spec,
            cards_spec,
            cards_spec,
            cards_spec,
        ]

        self._observation_spec = {
            i: j(i) for i, j in zip(self.observation_keys, self.observation_specs)
        }

        self._state = 0
        self.game_done = False

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), minimum=0, maximum=2, name="action", dtype=np.int32
        )

        self.last_score = 0

        if self.visualize:
            self.scores = []
            self.i = 0
            self.average_over = average_over
            self.fig, self.ax = plt.subplots()
            (self.line,) = self.ax.plot([], [], "x", lw=1, label="1 game")
            (self.avgline,) = self.ax.plot(
                [], [], lw=2, label="{} game average".format(self.average_over)
            )
            self.ax.set_xlabel("games")
            self.ax.set_ylabel("score")
            self.ax.legend(loc="best", ncol=2)
            self.lasttile = 1
            plt.show(block=False)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def replot(self):
        self.i += 1
        self.line.set_data(range(len(self.scores)), self.scores)

        if len(self.scores) % self.average_over == 0:
            averages = np.mean(
                np.array(self.average_over * [0] + self.scores).reshape(
                    -1, self.average_over
                ),
                axis=1,
            )
            self.avgline.set_data(
                self.average_over * np.array(range(len(averages))), averages
            )
            self.fig.savefig("./wurst.pdf")

        self.ax.set_xlim(0, len(self.scores))
        self.ax.set_ylim(-5, 5)
        self.fig.canvas.draw()

    def recv_state(self):
        state = self.my_pipe_end.recv()
        observation_states = [
            np.array(value, dtype=np.int32)
            if isinstance(value, Iterable)
            else np.array([value], dtype=np.int32)
            for value in state[:-2]
        ]
        self._state = {
            key: value for key, value in zip(self.observation_keys, observation_states)
        }
        self.round_done = state[-2]
        self.game_done = state[-1]

    def _step(self, action):
        if self.game_done:
            return self._reset()
        self.my_pipe_end.send(action)
        self.recv_state()
        _print(f"recieved state {self._state}")

        def calculate_reward():
            nothing_changed=True
            for new,old in zip(self._state.values(), self.last_state.values()):
                if any(new != old):
                    nothing_changed=False
                    break

            self.last_state = self._state
            if self.round_done or self.game_done:
                # my score - opponents score
                reward = float(self._state['my_score'][0])# - self._state[0]# - self.last_scores #TODO maybe but it back in
                self.last_score = reward
                return reward
            else:
                return -1 * nothing_changed

        reward = calculate_reward() * 1.0
        _print("reward: ", reward)
        if self.game_done:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=1.0)  # TODO discount?

    def _reset(self):
        self.game_done = False
        self._state = dict()
        if self.visualize:
            self.scores.append(self.last_score)
            self.replot()
        self.my_pipe_end, other_pipe_end = multiprocessing.Pipe()
        multiprocessing.Process(
            target=Game(2, [0], other_pipe_end, print_function=_print).play
        ).start()
        self.recv_state()
        self.last_score = 0
        self.last_state = dict()
        self.round_done = False
        self.game_done = False
        return ts.restart(self._state)


from tf_agents.environments.utils import validate_py_environment

validate_py_environment(Env(), episodes=3)
