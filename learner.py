import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, LSTM, TimeDistributed
from keras.optimizers import Adam

from wizard_env import Env

import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network


fc_layer_params = (100,)


env = tf_py_environment.TFPyEnvironment(Env())

q_net = q_network.QNetwork(
    env.observation_spec(),
    env.action_spec(),
    preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
    fc_layer_params=fc_layer_params)


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

train_step_counter = tf.Variable(0)

from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

from tf_agents.policies import random_tf_policy

random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                env.action_spec())

ts = env._reset()

next_action = random_policy.action(ts)
env.step(next_action)

#CONTINUE here
