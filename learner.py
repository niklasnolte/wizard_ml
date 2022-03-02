import random

import torch
from numpy import mean
from tqdm import tqdm

# envs
from wizard_env import WizardEnv
from util import argmax, RARingBuffer

# TODO next:
# two predictors: one for trick guess and one for trick play

class Agent:
    def __init__(
        self,
        env,
        state_value_predictor,
        action_decider,
        explore_prob=0.5,
        discount=0.95,
        device=None,
        bufsize=5000
    ):
        self.device = device or torch.device("cuda:0")
        self.state_value_predictor = state_value_predictor.to(self.device)
        self.action_decider = action_decider.to(self.device)
        self.env = env
        self.bufsize = bufsize
        self.states = RARingBuffer((bufsize, env.observation_dim), dtype=torch.float32, device=self.device)
        self.actions = RARingBuffer((bufsize, 1), dtype=torch.long, device=self.device)
        self.state_scores = RARingBuffer((bufsize, 1), dtype=torch.float32, device=self.device)
        self.action_mask = torch.ones(len(self.env.action_space), dtype=bool, device=self.device)
        self.discount = discount
        self.explore_prob = explore_prob
        self.optimizer = torch.optim.Adam(
            self.state_value_predictor.parameters(), lr=1e-2
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.98
        )

    def run_episode(self):
        state = self.env.reset()
        done = False
        current_rewards = []
        n = 0
        while not done:
            action = self.get_next_action(state, self.explore_prob)
            self.states.add(state)
            state, reward, done, constraint = self.env.step(action)
            self.action_mask = constraint
            self.actions.add(action)
            current_rewards.append(reward)
            n += 1

        # calculate cumulative rewards
        for i in range(n):
            G = 0
            for j in range(i, n):
                G += current_rewards[j] * self.discount ** (j - i)
            self.state_scores.add(G)

        return sum(current_rewards)


    def get_next_action(self, state, eps):
        if random.uniform(0, 1) < eps:
            return random.choice([a for a,i in zip(self.env.action_space, self.action_mask) if i])
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        state = state.reshape(1, -1)
        values = self.action_decider(state)
        out = argmax(values[0], self.action_mask)
        return out

    def train(self, epochs=20):
        states, actions, scores = self.sample_memory(size=256)
        for _ in range(epochs):
            self.optimizer.zero_grad()
            state_values = self.state_value_predictor(states)
            # maybe FIXME:
            # are we losing something if we consider only
            # the actions that were actually performed?
            # consider keeping the other outputs similar
            loss = torch.nn.functional.mse_loss(
                state_values.take_along_dim(actions, dim=1), scores
            )
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        self.action_decider.load_state_dict(self.state_value_predictor.state_dict())
        return loss.item()

    def sample_memory(self, size):
        idxs = torch.tensor(random.sample(range(self.bufsize), size))
        states = self.states.gather(idxs)
        actions = self.actions.gather(idxs)
        rewards = self.state_scores.gather(idxs)

        return states, actions, rewards


def get_model(n_inputs, n_outputs):
    return torch.nn.Sequential(
        torch.nn.Linear(n_inputs, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, n_outputs),
    )


def main():
    env = WizardEnv(debug=False)
    n_inputs = env.observation_dim
    n_outputs = len(env.action_space)
    state_value_predictor = get_model(n_inputs, n_outputs)
    action_decider = get_model(n_inputs, n_outputs)
    action_decider.load_state_dict(state_value_predictor.state_dict())
    action_decider.requires_grad_(False)
    agent = Agent(env, state_value_predictor, action_decider)
    agent.explore_prob = 1
    for _ in tqdm(range(3000)):
        agent.run_episode()

    agent.explore_prob = .5
    bar = tqdm(range(1000))
    rewards = []
    horizon = 100
    for _ in bar:
        cumrewards = agent.run_episode()
        loss = agent.train(50)
        agent.explore_prob *= .99
        rewards += [cumrewards]
        bar.set_description(f"Loss: {loss:.3f}, current reward: {cumrewards:<2.0f} reward mean: {mean(rewards[-horizon:]):.3f}")


if __name__ == "__main__":
    main()
