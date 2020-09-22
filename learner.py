# %%
import numpy as np
import tensorflow as tf

#debugging
from traceback import print_stack

from sys import argv

# agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent

# envs
from tf_agents.environments import tf_py_environment
from wizard_env import MultiWizardEnv, WizardEnv, flatten_observation

# networks
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.networks import mask_splitter_network

# misc
from tf_agents.eval import metric_utils
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics

from IPython import embed

# %%
num_parallel_environments = 1
use_rnn = False

if num_parallel_environments == 1:
    env = tf_py_environment.TFPyEnvironment(WizardEnv(with_print=False))
else:
    env = tf_py_environment.TFPyEnvironment(
        MultiWizardEnv(n_envs=num_parallel_environments, with_print=False)
    )

# %%

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

# %%
train_step_counter = tf.Variable(0)

# %%
def split_observation_and_constraint(obs):
    # print_stack()
    return obs["state"], obs["constraint"]


# %%
concat_layer = tf.keras.layers.Lambda(
    lambda x: tf.keras.layers.Concatenate()(flatten_observation(x))
)
# %%

# if not use_rnn:
#     q_net = q_network.QNetwork(
#         env.observation_spec()["state"],
#         env.action_spec(),
#         preprocessing_combiner=concat_layer,
#         fc_layer_params=(16, 16),
#     )
# else:
#     q_net = q_rnn_network.QRnnNetwork(
#         env.observation_spec()["state"],
#         env.action_spec(),
#         preprocessing_combiner=concat_layer,
#         input_fc_layer_params=(16,16),
#         output_fc_layer_params=(16,),
#         lstm_size=(40,),
#     )

# agent = dqn_agent.DdqnAgent(
#     env.time_step_spec(),
#     env.action_spec(),
#     q_network=q_net,
#     optimizer=optimizer,
#     td_errors_loss_fn=common.element_wise_huber_loss,
#     observation_and_action_constraint_splitter=split_observation_and_constraint,
#     train_step_counter=train_step_counter,
#     epsilon_greedy=None,
#     boltzmann_temperature=0.1,
#     target_update_period=20,
# )


# actor_net = mask_splitter_network.MaskSplitterNetwork(
#     split_observation_and_constraint,
#     actor_distribution_network.ActorDistributionNetwork(
#         env.observation_spec()['state'],
#         env.action_spec(),
#         preprocessing_combiner=concat_layer,
#         fc_layer_params=(16, 16)
#     ),
#     passthrough_mask = True
# )
# value_net = mask_splitter_network.MaskSplitterNetwork(
#     split_observation_and_constraint, 
#     value_network.ValueNetwork(
#         env.observation_spec()['state'],
#         preprocessing_combiner=concat_layer,
#         fc_layer_params=(16, 16),
#     ),
#     passthrough_mask = False
# )

actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec()['state'],
        env.action_spec(),
        preprocessing_combiner=concat_layer,
        fc_layer_params=(16, 16)
    )


value_net = value_network.ValueNetwork(
        env.observation_spec()['state'],
        preprocessing_combiner=concat_layer,
        fc_layer_params=(16, 16),
)


agent = ppo_agent.PPOAgent(
    env.time_step_spec(),
    env.action_spec(),
    optimizer,
    actor_net=actor_net,
    value_net=value_net,
    num_epochs=10,
    # normalize_observations=False,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    train_step_counter=train_step_counter,
    observation_and_action_constraint_splitter=split_observation_and_constraint,
)

# %%
agent.initialize()


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
    data_spec=agent.collect_data_spec, batch_size=env.batch_size, max_length=1000
)

# %%
# initial data collection, only with dqn, as for ppo the collect data spec changes
# dynamic_step_driver.DynamicStepDriver(
#     env,
#     random_policy,
#     observers=[replay_buffer.add_batch],
#     num_steps=env.batch_size * 100,
# ).run()


# %%
dataset = replay_buffer.as_dataset(
    num_parallel_calls=num_parallel_environments,
    sample_batch_size=100,  # env.batch_size,
    num_steps= (agent._n_step_update if hasattr(agent, "_n_step_update") else 1) + 1
    # sample batch size seems to change a great deal
).prefetch(num_parallel_environments)

# %%
i = []

def count(x):
    print(len(i))
    print(agent.collect_policy.action(env.current_time_step()))
    i.append(1)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps= 5,
)

# %%
# make it faster(?)
# agent.train = common.function(agent.train)
# collect_driver.run = common.function(collect_driver.run)

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
    dyn_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        eval_env, policy, list(eval_metrics.values()), num_episodes=n_episodes
    )

    # dyn_driver.run = common.function(dyn_driver.run)

    dyn_driver.run()

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

    for _ in range(0, num_iterations):
        time_step, policy_state = collect_driver.run(
            time_step=time_step, policy_state=policy_state
        )
        experience, _ = next(it)
        train_loss = agent.train(experience).loss
        step = agent.train_step_counter.numpy()
        print("step = {0}: loss = {1}".format(step, train_loss))
        if step % 100 == 0:
            evaluate(agent.policy, 100)


# %%
#evaluate(random_policy, 500)

# %%

if len(argv) > 1:
    train(int(argv[1]))
else:
    train(10000)

# # %%
# evaluate(agent.policy, 500)

# %%
