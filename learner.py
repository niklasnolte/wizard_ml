# %%
from time import sleep

import numpy as np
import tensorflow as tf
from tf_agents.eval import metric_utils
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics

from wizard_env import MultiWizardEnv, WizardEnv, flatten_observation


# %%
num_parallel_environments = 50
env = tf_py_environment.TFPyEnvironment(
    MultiWizardEnv(n_envs=num_parallel_environments, with_print=False)
)

# %%

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

# %%
train_step_counter = tf.Variable(0)

# %%
def split_observation_and_constraint(obs):
    return obs["state"], obs["constraint"]


# %%
concat_layer = tf.keras.layers.Lambda(
    lambda x: tf.keras.layers.Concatenate()(flatten_observation(x))
)

q_net = q_network.QNetwork(
    env.observation_spec()["state"],
    env.action_spec(),
    preprocessing_combiner=concat_layer,
)
# %%
agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    n_step_update=1,
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    observation_and_action_constraint_splitter=split_observation_and_constraint,
    train_step_counter=train_step_counter,
    gamma=0.9,
)

# %%
agent.initialize()

# %%
# make it faster(?)
agent.train = common.function(agent.train)

# %%
agent.train_step_counter.assign(0)

# %%

random_policy = random_tf_policy.RandomTFPolicy(
    env.time_step_spec(),
    env.action_spec(),
    observation_and_action_constraint_splitter=split_observation_and_constraint,
)

# %%
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec, batch_size=env.batch_size, max_length=10000
)
# %%
# initial data collection
dynamic_step_driver.DynamicStepDriver(
    env,
    random_policy,
    observers=[replay_buffer.add_batch],
    num_steps=100,
).run()

# %%
dataset = replay_buffer.as_dataset(
    num_parallel_calls=5,
    sample_batch_size=30,#env.batch_size,
    num_steps=agent._n_step_update + 1
    # sample batch size seems to change a great deal
).prefetch(5)

# %%
collect_driver = dynamic_step_driver.DynamicStepDriver(
    env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=1,
)

# %%
eval_env = tf_py_environment.TFPyEnvironment(WizardEnv(with_print=False))

def evaluate(policy, n_episodes=100):
    eval_env.reset()

    eval_metrics = dict(
        n_eps=tf_metrics.NumberOfEpisodes(),
        n_steps=tf_metrics.EnvironmentSteps(),
        avrg_return=tf_metrics.AverageReturnMetric(
            batch_size=eval_env.batch_size, buffer_size=n_episodes
        ),
    )
    # consider metric_utils.eager_compute here
    dynamic_episode_driver.DynamicEpisodeDriver(
        eval_env, policy, list(eval_metrics.values()), num_episodes=n_episodes
    ).run()

    _which_policy = (
        "random" if isinstance(policy, random_tf_policy.RandomTFPolicy) else "trained"
    )

    print(f"Number of Steps: {eval_metrics['n_steps'].result().numpy()}")
    print(f"Number of Episodes: {eval_metrics['n_eps'].result().numpy()}")
    print(
        f"Avrg return with {_which_policy} policy: {eval_metrics['avrg_return'].result().numpy()}"
    )


# %%
def train(num_iterations=5000):
    it = iter(dataset)

    time_step = None
    policy_state = agent.collect_policy.get_initial_state(env.batch_size)

    for _ in range(0, num_iterations, num_parallel_environments):
        time_step, policy_state = collect_driver.run(
            time_step=time_step, policy_state=policy_state
        )

        experience, _ = next(it)
        train_loss = agent.train(experience).loss
        step = agent.train_step_counter.numpy() * num_parallel_environments

        if step % 500 in list(range(num_parallel_environments)):
            print("step = {0}: loss = {1}".format(step, train_loss))
            evaluate(agent.policy, 50)


# %%
evaluate(random_policy, 500)

# %%
train(5000)

# %%
evaluate(agent.policy, 500)