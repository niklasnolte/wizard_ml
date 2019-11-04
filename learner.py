import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, LSTM, TimeDistributed
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from wizard_env import Env
import gym

env = Env(visualize=False)
np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Dense(16, input_shape=(1,16)))
# model.add(Flatten(input_shape=(1,16)))
# model.add(Dense(16, init='random_uniform', activation='relu'))
model.add(LSTM(3, return_sequences=True))
# model.add(LSTM(32, return_sequences=True))
model.add(Flatten())
model.add(Dense(nb_actions, init='random_uniform', activation='linear'))
model.summary()


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=10000, window_length=1)
args = {
    'model':               model,
    'nb_actions':          nb_actions,
    'memory':              memory,
    'batch_size':          64,
    'target_model_update': 1e-2,
    'policy':              BoltzmannQPolicy(),
}
args['nb_steps_warmup'] = max(30, args['batch_size'])

dqn = DQNAgent(**args)
dqn.compile(Adam(lr=1e-1), metrics=['mae'])

# fitting step
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights('dqn_learned_weights.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=50, visualize=False)
