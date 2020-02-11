from time import sleep

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from wizard_env import Env

num_eval_episodes = 1

env = tf_py_environment.TFPyEnvironment(Env(with_print=True))


def concat_fun(x):
    return tf.keras.layers.Concatenate()(list(x))


concat_layer = tf.keras.layers.Lambda(concat_fun)


q_net = q_network.QNetwork(
    env.observation_spec()['state'], env.action_spec(), preprocessing_combiner=concat_layer
)


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-1)


train_step_counter = tf.Variable(0)


def split_observation_and_constraint(obs):
    print(obs)
    return obs["state"], obs["constraint"]


agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    observation_and_action_constraint_splitter=split_observation_and_constraint,
    train_step_counter=train_step_counter,
)

agent.initialize()


random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())

# metric for evaluation of the agent
def compute_avrg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec, batch_size=env.batch_size, max_length=1000
)


def collect_step(env, policy, buf):
    ts = env.current_time_step()
    action_step = policy.action(ts)
    next_ts = env.step(action_step.action)
    traj = trajectory.from_transition(ts, action_step, next_ts)
    buf.add_batch(traj)


def collect_data(env, policy, buf, steps):
    for _ in range(steps):
        collect_step(env, policy, buf)


collect_data(env, random_policy, replay_buffer, steps=2)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=1, sample_batch_size=1, num_steps=2
)

it = iter(dataset)

agent.train = common.function(agent.train)

agent.train_step_counter.assign(0)

avg_return = compute_avrg_return(env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(10000):
    collect_step(env, agent.collect_policy, replay_buffer)

    experience, _ = next(it)
    train_loss = agent.train(experience).loss
    step = agent.train_step_counter.numpy()

    if step % 1 == 0:
        print("step = {0}: loss = {1}".format(step, train_loss))
    if step % 1 == 0:
        avg_return = compute_avrg_return(env, agent.policy, num_eval_episodes)
        print("step = {0}: Average Return = {1}".format(step, avg_return))
        returns.append(avg_return)
