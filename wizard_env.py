from game import Game, Card

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


class WizardEnv:
    def __init__(self, debug=False, nplayers=2, n_rounds=2):
        self.debug_print = print if debug else (lambda *args, **kwargs: None)

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
            return 0
            # return -1.0 * abs(
            #     self._state["state"]["Player_1"]["n_tricks"]
            #     - self._state["state"]["Player_1"]["trick_guess"]
            # )

    def step(self, action):
        next_state = self.game_it.send(int(action))
        self.register_state(next_state)
        self.debug_print(f"recieved state {self._state}")

        reward = self.calculate_reward()
        self.debug_print("reward: ", reward)
        return obs2vec(self._state["state"]), reward, self.game_done, self._state["constraint"]

    def reset(self):
        self.game_done = False
        self.game = Game(
            nplayers=self.nplayers,
            random_idxs=[0],
            n_rounds=self.n_rounds,
            print_function=self.debug_print,
        )
        self.game_it = self.game.play()
        initial_game_state = next(self.game_it)
        self.register_state(initial_game_state)
        self.last_score = 0
        self.last_state = dict()
        self.round_done = False
        self.game_done = False
        return obs2vec(self._state["state"])

    @property
    def game_state(self):
      return self.game.game_state

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
            if name == "Player_1":
              cardstack = cardholder["cards"]
            else:
              cardstack = {i:(0,-1) for i,_ in enumerate(cardholder["cards"])} # FIXME cards visible?
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
