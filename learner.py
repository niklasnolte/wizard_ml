import random

import torch
from numpy import vstack
from tqdm import tqdm

# envs
from wizard_env import WizardEnv
from util import argmax


class Agent:
    def __init__(
        self,
        env,
        state_value_predictor,
        action_decider,
        explore_prob=0.5,
        discount=0.95,
        device=None,
    ):
        self.device = device or torch.device("cpu")
        self.state_value_predictor = state_value_predictor.to(self.device)
        self.action_decider = action_decider.to(self.device)
        self.env = env
        # TODO implement these with static memory
        self.states = []
        self.actions = []
        self.action_mask = torch.ones(len(self.env.action_space), dtype=bool, device=self.device)
        self.state_scores = []
        self.discount = discount
        self.explore_prob = explore_prob
        self.optimizer = torch.optim.Adam(
            self.state_value_predictor.parameters(), lr=1e-2
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.97
        )

    def run_episode(self):
        state = self.env.reset()
        done = False
        current_rewards = []
        n = 0
        while not done:
            action = self.get_next_action(state, self.explore_prob)
            self.states.append(state)
            state, reward, done, constraint = self.env.step(action)
            self.action_mask = constraint
            self.actions.append(action)
            current_rewards.append(reward)
            n += 1

        # calculate cumulative rewards
        discounted_rewards = []
        for i in range(n):
            G = 0
            for j in range(i, n):
                G += current_rewards[j] * self.discount ** (j - i)
            discounted_rewards.append(G)

        self.state_scores.extend(discounted_rewards)
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
        states, actions, scores = self.sample_memory(size=1000)
        for _ in range(epochs):
            self.optimizer.zero_grad()
            state_values = self.state_value_predictor(states)
            loss = torch.nn.functional.mse_loss(
                state_values.take_along_dim(actions, dim=1), scores
            )
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        self.action_decider.load_state_dict(self.state_value_predictor.state_dict())
        return loss.item()

    def sample_memory(self, size):
        # this can be made better (with static mem)
        idxs = random.sample(range(len(self.states)), size)
        states = [self.states[i] for i in idxs]
        actions = [self.actions[i] for i in idxs]
        rewards = [self.state_scores[i] for i in idxs]

        states = torch.tensor(vstack(states), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)

        return states, actions.view(-1, 1), rewards.view(-1, 1)


def get_model(n_inputs, n_outputs):
    return torch.nn.Sequential(
        torch.nn.Linear(n_inputs, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, n_outputs),
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
    for _ in bar:
        cumrewards = agent.run_episode()
        loss = agent.train(100)
        agent.explore_prob *= .99
        bar.set_description(f"Loss: {loss:.3f}, total reward: {cumrewards:.3f}")


if __name__ == "__main__":
    main()
