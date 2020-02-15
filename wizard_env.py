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
from collections import defaultdict


class Env(py_environment.PyEnvironment):
    def __init__(self, average_over=50, visualize=False, with_print=True):
        global _print
        _print = print if with_print else (lambda *args, **kwargs: None)
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

        self._state = dict()
        self.game_done = False

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), minimum=0, maximum=2, name="action", dtype=np.int32
        )

        n_actions = self._action_spec._maximum - self._action_spec._minimum + 1

        self._action_mask_spec = array_spec.BoundedArraySpec(
            shape=(n_actions,), minimum=0, maximum=1, dtype=np.int32
        )

        self._observation_spec = {
            "state": {
                "Player_0": {
                    "score": score_spec("enemy_score"),
                    "trick_guess": score_spec("enemy_trick_guess"),
                    "cards": {
                        "0": cards_spec("enemy_first_card"),
                        "1": cards_spec("enemy_first_card"),
                    },
                },
                "Player_1": {
                    "score": score_spec("my_score"),
                    "trick_guess": score_spec("my_trick_guess"),
                    "cards": {
                        "0": cards_spec("my_first_card"),
                        "1": cards_spec("my_first_card"),
                    },
                },
                "trick": {
                    "0": cards_spec("first_in_trick"),
                    "1": cards_spec("second_in_trick"),
                },
            },
            "constraint": self._action_mask_spec,
        }

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

        self.reset()

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

    def register_state(self, game_state):
        self.round_done = game_state.pop("round_done")
        self.game_done = game_state.pop("game_over")
        self._state = game_state

    def _step(self, action):
        if self.game_done:
            _print("new game!")
            return self.reset()
        next_state = self.game_com_channel.send(int(action))
        self.register_state(next_state)
        _print(f"recieved state {self._state}")

        def calculate_reward():
            self.last_state = self._state
            if self.round_done or self.game_done:
                # my score is the reward
                reward = float(self._state["state"]["Player_1"]["score"][0])
                self.last_score = reward
                return reward
            else:
                return 0

        reward = calculate_reward() * 1.0
        _print("reward: ", reward)
        if self.game_done:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=0.9)  # TODO discount?

    def _reset(self):
        self.game_done = False
        if self.visualize:
            self.scores.append(self.last_score)
            self.replot()
        self.game_com_channel = Game(2, [0], print_function=_print).play()
        initial_game_state = next(self.game_com_channel)
        self.register_state(initial_game_state)
        self.last_score = 0
        self.last_state = dict()
        self.round_done = False
        self.game_done = False
        return ts.restart(self._state)


# to flatten the observation spec
def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def flatten_observation(obs):
    flat_dict = flatten_dict(obs)
    return list(flat_dict.values())


if __name__ == "__main__":
    from tf_agents.environments.utils import validate_py_environment

    myEnv = Env(with_print=False)
    validate_py_environment(myEnv, episodes=5)
