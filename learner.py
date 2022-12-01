import random
from copy import deepcopy

import torch
from numpy import mean
from tqdm import tqdm

# envs
from wizard_env import WizardEnv
from util import argmax, RARingBuffer
from game import GameState

#TODO weight sharing between two networks
#TODO fix card predictor?

class Agent:
    def __init__(
        self,
        env,
        card_value_predictor,
        trick_value_predictor,
        card_decider,
        trick_decider,
        explore_prob=0.5,
        discount=0.95,
        device=None,
        bufsize=5000,
    ):
        self.device = device or torch.device("cuda:0")
        self.card_value_predictor = card_value_predictor.to(self.device)
        self.trick_value_predictor = trick_value_predictor.to(self.device)
        self.card_decider = card_decider.to(self.device)
        self.trick_decider = trick_decider.to(self.device)
        self.env = env
        self.bufsize = bufsize

        self.trick_states = RARingBuffer(
            (bufsize, env.observation_dim), dtype=torch.float32, device=self.device
        )
        self.trick_actions = RARingBuffer(
            (bufsize, 1), dtype=torch.long, device=self.device
        )
        self.card_states = RARingBuffer(
            (bufsize, env.observation_dim), dtype=torch.float32, device=self.device
        )
        self.card_actions = RARingBuffer(
            (bufsize, 1), dtype=torch.long, device=self.device
        )

        # TODO two different state scores, for trick and card?
        self.trick_state_scores = RARingBuffer(
            (bufsize, 1), dtype=torch.float32, device=self.device
        )
        self.card_state_scores = RARingBuffer(
            (bufsize, 1), dtype=torch.float32, device=self.device
        )
        self.action_mask = torch.ones(
            len(self.env.action_space), dtype=bool, device=self.device
        )
        self.discount = discount
        self.explore_prob = explore_prob
        self.card_optimizer = torch.optim.Adam(
            self.card_value_predictor.parameters(), lr=1e-3
        )
        self.trick_optimizer = torch.optim.Adam(
            self.trick_value_predictor.parameters(), lr=1e-3
        )

    def run_episode(self):
        state = self.env.reset()
        done = False
        current_rewards = []
        guessing_tricks = []
        while not done:
            guessing_tricks.append(self.env.game_state == GameState.GuessingTricks)
            action = self.get_next_action(state, self.explore_prob)
            state, reward, done, constraint = self.env.step(action)
            self.action_mask = constraint
            if guessing_tricks[-1]:
                self.trick_states.add(state)
                self.trick_actions.add(action)
            else:
                self.card_states.add(state)
                self.card_actions.add(action)
            current_rewards.append(reward)

        # calculate cumulative rewards
        for i in range(len(current_rewards)):
            G = 0
            for j in range(i, len(current_rewards)):
                G += current_rewards[j] * self.discount ** (j - i)
            if guessing_tricks[i]:
                self.trick_state_scores.add(G)
            else:
                self.card_state_scores.add(G)

        return sum(current_rewards)

    def get_next_action(self, state, eps):
        if random.uniform(0, 1) < eps:
            return random.choice(
                [a for a, i in zip(self.env.action_space, self.action_mask) if i]
            )
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        state = state.reshape(1, -1)
        if self.env.game_state == GameState.GuessingTricks:
            values = self.trick_decider(state)
        else:
            values = self.card_decider(state)
        out = argmax(values[0], self.action_mask)
        return out

    def train(self, epochs=20, update_action_decider=True):
        losses = []
        for optim, predictor, decider, for_tricks in zip(
            (self.card_optimizer, self.trick_optimizer),
            (self.card_value_predictor, self.trick_value_predictor),
            (self.card_decider, self.trick_decider),
            (False, True),
        ):
            states, actions, scores = self.sample_memory(
                size=1024, for_tricks=for_tricks
            )
            for _ in range(epochs):
                optim.zero_grad()
                state_values = predictor(states)
                # maybe FIXME:
                # are we losing something if we consider only
                # the actions that were actually performed?
                # consider keeping the other outputs similar
                loss = torch.nn.functional.mse_loss(
                    state_values.take_along_dim(actions, dim=1), scores
                )
                loss.backward()
                optim.step()

            # test the decider
            with torch.no_grad():
              states, actions, scores = self.sample_memory(
                  size=256, for_tricks=for_tricks
              )
              state_values = predictor(states)
              decider_loss = torch.nn.functional.mse_loss(
                  state_values.take_along_dim(actions, dim=1), scores
              )
              losses.append(decider_loss.item())

              if update_action_decider:
                  decider.load_state_dict(predictor.state_dict())

        return losses

    def sample_memory(self, size, for_tricks):
        idxs = torch.tensor(random.sample(range(self.bufsize), size))
        if for_tricks:
            states = self.trick_states.gather(idxs)
            actions = self.trick_actions.gather(idxs)
            rewards = self.trick_state_scores.gather(idxs)
        else:
            states = self.card_states.gather(idxs)
            actions = self.card_actions.gather(idxs)
            rewards = self.card_state_scores.gather(idxs)

        return states, actions, rewards


def get_model(n_inputs, n_outputs):
    return torch.nn.Sequential(
        torch.nn.Linear(n_inputs, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, n_outputs),
    )


if __name__ == "__main__":
    env = WizardEnv(debug=False)
    n_inputs = env.observation_dim
    n_outputs = len(env.action_space)
    state_value_predictor = get_model(n_inputs, n_outputs)
    action_decider = get_model(n_inputs, n_outputs)
    action_decider.load_state_dict(state_value_predictor.state_dict())
    action_decider.requires_grad_(False)
    agent = Agent(
        env,
        state_value_predictor,
        deepcopy(state_value_predictor),
        action_decider,
        deepcopy(action_decider),
    )
    agent.explore_prob = 1
    for _ in tqdm(range(3000)):
        agent.run_episode()

    bar = tqdm(range(1000))
    rewards = []
    horizon = 100
    agent.explore_prob *= .1
    greedy_decay = (0.01) ** (1 / len(bar))
    for i in bar:
        cumrewards = agent.run_episode()
        trick_loss, card_loss = agent.train(10, i % 5 == 0)
        # explore prob should be very small at the end
        agent.explore_prob *= greedy_decay
        rewards += [cumrewards]
        bar.set_description(
            f"Trick loss: {trick_loss:.3f}, Card loss: {card_loss:.3f}, current reward: {cumrewards:<2.0f} reward mean: {mean(rewards[-horizon:]):.3f}"
        )
    from IPython import embed; embed()
