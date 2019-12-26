import os
from collections import deque
from random import sample

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class DQNAgent:

    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.sample_size = 32
        self.epsilon = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.num_actions = env.action_space.n
        self.num_of_inputs = env.observation_space.shape[0]
        self.tau = 1.
        self.clip = (-500., 500.)
        self.model = self.build_keras_model()
        self.target_model = self.build_keras_model()

    def build_keras_model(self):
        # create model
        model = Sequential()
        model.add(Dense(128, input_dim=self.num_of_inputs, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        # compile model
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state):
        q_values = self.model.predict(state)[0]

        # Boltzmann exploration: https://github.com/keras-rl/keras-rl
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)

        return action

    def remember(self, state, action, r, next_s, done):
        self.memory.append([state, action, r, next_s, done])

    def replay(self):
        if len(self.memory) < self.sample_size:
            return

        mini_batch = sample(self.memory, self.sample_size)

        y_batch = np.zeros((self.sample_size, self.num_actions), dtype=np.float64)
        state_batch = np.zeros((self.sample_size, self.num_of_inputs), dtype=np.float64)
        for i in range(0, len(mini_batch)):
            state, action, reward, next_state, done = mini_batch[i]
            if done:
                y = reward
            else:
                next_q_values = self.target_model.predict(next_state)
                y = reward + (self.gamma * np.max(next_q_values))

            q_values = self.target_model.predict(state)
            q_values[0][action] = y
            y_batch[i] = q_values
            state_batch[i] = state

        self.model.fit(state_batch, y_batch, batch_size=self.sample_size, epochs=1, verbose=False)

        self.update_target_model_weight()

    # Make network more stable and converge faster:
    # https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
    # https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    def update_target_model_weight(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def save_model(self, model_name):
        current_directory = os.getcwd()
        model_path = os.path.join(current_directory, "data/model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model.save(filepath="{0}/{1}".format(model_path, model_name))


def main(env, render=False):
    episodes = 350
    reward_per_episode = 0
    dqnAgent = DQNAgent(env=env)

    for episode in range(episodes):
        current_state = env.reset()
        current_state = np.asarray(current_state).reshape(1, 8)
        done = False
        while not done:
            if render: env.render()

            action = dqnAgent.act(state=current_state)

            next_state, r, done, _ = env.step(action)

            next_state = np.asarray(next_state).reshape(1, 8)

            dqnAgent.remember(current_state, action, r, next_state, done)

            dqnAgent.replay()

            current_state = next_state

            reward_per_episode += r

        print("Total reward for episode {0} = {1}".format(episode, reward_per_episode))
        reward_per_episode = 0

    dqnAgent.save_model("lunarlandar.model")

    print("Done.")


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    env = gym.wrappers.Monitor(env, 'data/videos',
                               video_callable=lambda episode_id: episode_id % 10 == 0,
                               force=True)
    main(env, render=False)
