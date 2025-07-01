import random
import logging
import os
from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import keras


@keras.saving.register_keras_serializable()
def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, window_size, n_features, strategy="t-dqn", reset_every=1000, pretrained=False, model_name=None):
        self.strategy = strategy

        # agent config
        self.window_size = window_size
        self.n_features = n_features
        self.state_size = (window_size - 1) + n_features + 1  # price_diffs + tech_features + portfolio_feature
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(learning_rate=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Creates the model
        """
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(units=128, activation="relu"),
            Dense(units=256, activation="relu"),
            Dense(units=256, activation="relu"),
            Dense(units=128, activation="relu"),
            Dense(units=self.action_size)
        ])

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        # In training, force a buy on the first step to kickstart exploration
        if self.first_iter and not is_eval:
            self.first_iter = False
            return 1  # Action 1: BUY

        action_probs = self.model.predict(state, verbose=0)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
        mini_batch = random.sample(self.memory, batch_size)

        # Vectorized processing for efficiency
        states = np.array([transition[0] for transition in mini_batch]).reshape(batch_size, self.state_size)
        actions = np.array([transition[1] for transition in mini_batch])
        rewards = np.array([transition[2] for transition in mini_batch])
        next_states = np.array([transition[3] for transition in mini_batch]).reshape(batch_size, self.state_size)
        dones = np.array([transition[4] for transition in mini_batch])

        # Get current Q-values for the states in the batch
        q_values_current = self.model.predict(states, verbose=0)
        
        # Initialize targets with current Q-values
        targets = np.copy(q_values_current)
        
        # Calculate future Q-values for non-terminal states
        future_rewards = np.zeros(batch_size)
        
        non_terminal_indices = np.where(dones == False)[0]
        if len(non_terminal_indices) > 0:
            non_terminal_next_states = next_states[non_terminal_indices]

            if self.strategy == "dqn":
                q_futures = self.model.predict(non_terminal_next_states, verbose=0)
                future_rewards[non_terminal_indices] = self.gamma * np.amax(q_futures, axis=1)
            
            elif self.strategy == "t-dqn":
                q_futures = self.target_model.predict(non_terminal_next_states, verbose=0)
                future_rewards[non_terminal_indices] = self.gamma * np.amax(q_futures, axis=1)

            elif self.strategy == "double-dqn":
                q_futures_main = self.model.predict(non_terminal_next_states, verbose=0)
                actions_from_main = np.argmax(q_futures_main, axis=1)
                
                q_futures_target = self.target_model.predict(non_terminal_next_states, verbose=0)
                future_rewards[non_terminal_indices] = self.gamma * q_futures_target[np.arange(len(q_futures_target)), actions_from_main]
        
        # Update targets: target = reward (for terminal states) or reward + future_reward
        targets[np.arange(batch_size), actions] = rewards + future_rewards
        
        # Fit the model
        loss = self.model.fit(states, targets, epochs=1, verbose=0).history["loss"][0]

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network for t-dqn and double-dqn
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter += 1
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())

        return loss

    def save(self, episode):
        os.makedirs("models", exist_ok=True)
        self.model.save("models/{}_{}.keras".format(self.model_name, episode))
    
    def save_best(self, suffix="best"):
        """Sauvegarde le meilleur modèle"""
        os.makedirs("models", exist_ok=True)
        path = "models/{}_{}.keras".format(self.model_name, suffix)
        self.model.save(path)
        logging.info(f"💾 Meilleur modèle sauvegardé: {path}")

    def load(self):
        """
        Loads a Keras model.
        It can handle a full path or just a model name in the `models` directory.
        """
        path = self.model_name
        
        # If the provided name is a full path that exists, use it directly.
        if os.path.isfile(path):
            return load_model(path, custom_objects=self.custom_objects)
        
        # Otherwise, assume it's a name and construct the path.
        path = f"models/{self.model_name}"
        if not path.endswith((".keras", ".h5")):
            path += ".keras"
            
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Le fichier du modèle n'a pas été trouvé. Tentative de chemin: {path}")
            
        return load_model(path, custom_objects=self.custom_objects)
