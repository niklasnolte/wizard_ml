import numpy as np
from main_simple import Game
from tf_agents.environments import py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from functools import partial
from collections.abc import Iterable


class WizardEnv(py_environment.PyEnvironment):
    def __init__(self, with_print=True):
        global _print
        _print = print if with_print else (lambda *args, **kwargs: None)
        # 0 = invalid
        # 1 = white
        # 2 = red
        # 3 = blue
        # 4 = green
        # 5 = yellow
        # -1 - 14 = values (-1 = no card there)
        cards_spec = lambda name: array_spec.BoundedArraySpec(
            shape=(2,), minimum=[0, -1], maximum=[5, 14], name=name, dtype=np.int32
        )
        score_spec = lambda name: array_spec.ArraySpec(
            shape=(1,), name=name, dtype=np.int32
        )

        self._state = dict()
        self.game_done = False

        self.last_round = 2

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,),
            minimum=0,
            maximum=self.last_round,
            name="action",
            dtype=np.int32,
        )

        self._action_mask_spec = array_spec.BoundedArraySpec(
            shape=(self.last_round+1,), minimum=0, maximum=1, dtype=np.int32
        )

        self._observation_spec = {
            "state": {
                "Player_0": {
                    "score": score_spec("enemy_score"),
                    "trick_guess": score_spec("enemy_trick_guess"),
                    "cards": {
                        i: cards_spec(f"enemy_{i}_card") for i in range(self.last_round)
                    },
                },
                "Player_1": {
                    "score": score_spec("my_score"),
                    "trick_guess": score_spec("my_trick_guess"),
                    "cards": {
                        i: cards_spec(f"my_{i}_card") for i in range(self.last_round)
                    },
                },
                "trick": {
                    i: cards_spec(f"trick_{i}_card") for i in range(self.last_round)
                },
            },
            "constraint": self._action_mask_spec,
        }

        self.last_score = 0

        #        super().__init__()

        self.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

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
        self.game_com_channel = Game(
            nplayers=2,
            random_idxs=[0],
            last_round=self.last_round,
            print_function=_print,
        ).play()
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
            prefix + separator + str(k) if prefix else str(k): v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def flatten_observation(obs):
    flat_dict = flatten_dict(obs)
    return list(flat_dict.values())


class MultiWizardEnv(parallel_py_environment.ParallelPyEnvironment):
    def __init__(self, n_envs=1, with_print=False, **kwargs):

        env_constructors = [partial(WizardEnv, with_print=with_print)] * n_envs
        super().__init__(env_constructors, **kwargs)


if __name__ == "__main__":
    from tf_agents.environments.utils import validate_py_environment

    myEnv = WizardEnv(with_print=False)
    validate_py_environment(myEnv, episodes=5)

    # my_multi_env = MultiWizardEnv(n_envs=1)
    # validate_py_environment(my_multi_env)
