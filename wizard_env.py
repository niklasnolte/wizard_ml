from game import Game

# 0 = invalid
# 1 = white
# 2 = red
# 3 = blue
# 4 = green
# 5 = yellow
# -1 - 14 = values (-1 = no card there)
# card type = Tuple(color, value)
# action type = Int[0,n_rounds]
# action_mask type = List[bool]
# trick guess type = Int[0,n_rounds]

# observation =
#     "state": {
#         "Player_0": {
#             "score": score_spec("enemy_score"),
#             "trick_guess": score_spec("enemy_trick_guess"),
#             "n_tricks": score_spec("enemy_n_tricks"),
#             "cards": {
#                 i: cards_spec(f"enemy_{i}_card") for i in range(self.n_rounds)
#             },
#         },
#         "Player_1": {
#             "score": score_spec("my_score"),
#             "trick_guess": score_spec("my_trick_guess"),
#             "n_tricks": score_spec("my_n_tricks"),
#             "cards": {
#                 i: cards_spec(f"my_{i}_card") for i in range(self.n_rounds)
#             },
#         },
#         "trick": {
#             i: cards_spec(f"trick_{i}_card") for i in range(self.n_rounds)
#         },
#     },
#     "constraint": self._action_mask_spec,
# }


class WizardEnv:
    def __init__(self, debug=False, nplayers=2, n_rounds=2):
        global debug_print
        debug_print = print if debug else (lambda *args, **kwargs: None)

        self.game_done = False
        self.n_rounds = n_rounds
        self.action_space = list(range(n_rounds + 1))
        self.nplayers = nplayers

        self.observation_dim = (
            nplayers * 3  # score, trick_guess, n_tricks
            + 7  # card repr, 6 for color, 1 for value
            * self.n_rounds  # so many cards per player
            * (nplayers + 1)  # player + current trick
        )
        self.reset()
        assert len(obs2vec(self._state["state"])) == self.observation_dim

    def register_state(self, game_state):
        self.round_done = game_state.pop("round_done")
        self.game_done = game_state.pop("game_over")
        self._state = game_state

    def calculate_reward(self):
        self.last_state = self._state
        if self.round_done or self.game_done:
            reward = float(self._state["state"]["Player_1"]["score"])
            self.last_score = reward
            return reward
        else:
            # FIXME might not be the best way to calculate reward
            # will favor greedy action
            return -1.0 * abs(
                self._state["state"]["Player_1"]["n_tricks"]
                - self._state["state"]["Player_1"]["trick_guess"]
            )

    def step(self, action):
        next_state = self.game.send(int(action))
        self.register_state(next_state)
        debug_print(f"recieved state {self._state}")

        reward = self.calculate_reward()
        debug_print("reward: ", reward)
        return obs2vec(self._state["state"]), reward, self.game_done, self._state["constraint"]

    def reset(self):
        self.game_done = False
        self.game = Game(
            nplayers=self.nplayers,
            random_idxs=[0],
            n_rounds=self.n_rounds,
            print_function=debug_print,
        ).play()
        initial_game_state = next(self.game)
        self.register_state(initial_game_state)
        self.last_score = 0
        self.last_state = dict()
        self.round_done = False
        self.game_done = False
        return obs2vec(self._state["state"])


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


def one_hot(i, n):
    a = [0] * n
    a[i] = 1
    return a


def obs2vec(obs):
    vec = []
    for name, cardholder in obs.items():
        if "Player" in name:
            vec.append(cardholder["score"])
            vec.append(cardholder["trick_guess"])
            vec.append(cardholder["n_tricks"])
            cardstack = cardholder["cards"]
        elif "trick" == name:
            cardstack = cardholder
        else:
            raise Exception(f"wtf {name}")
        for card in cardstack.values():
            vec.extend(one_hot(card[0], 6))
            vec.append(card[1])
    return vec


if __name__ == "__main__":
    env = WizardEnv(debug=True)
    state = env.reset()
