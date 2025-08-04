import numpy as np
import tensorflow as tf
import random
from collections import deque
import os
import pandas as pd

class DQNAgent:
    """Deep Q-Network agent for playing 2048."""
    
    def __init__(self, load_weights=True):
        """Initialize the DQN agent."""
        # Game parameters
        self.epsilon = 0.0  # No exploration in play mode
        self.learning_rate = 0.001  # Initialize learning_rate before _build_model()
        
        # Initialize the model
        self.model = self._build_model()
        
        # Load pre-trained weights if specified
        if load_weights and os.path.exists('model'):
            self.load_weights()
    
    def _build_model(self):
        """Build the DQN model architecture."""
        inputs = tf.keras.layers.Input(shape=(4, 4, 16))
        
        conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 2), padding='valid',
                                    activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 1), padding='valid',
                                    activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))(inputs)
        
        conv11 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 2), padding='valid',
                                        activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))(conv1)
        conv12 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 1), padding='valid',
                                        activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))(conv1)
        conv21 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 2), padding='valid',
                                        activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))(conv2)
        conv22 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 1), padding='valid',
                                        activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))(conv2)
        
        flat1  = tf.keras.layers.Flatten()(conv1)
        flat2  = tf.keras.layers.Flatten()(conv2)
        flat11 = tf.keras.layers.Flatten()(conv11)
        flat12 = tf.keras.layers.Flatten()(conv12)
        flat21 = tf.keras.layers.Flatten()(conv21)
        flat22 = tf.keras.layers.Flatten()(conv22)
        concat = tf.keras.layers.Concatenate()([flat1, flat2, flat11, flat12, flat21, flat22])
        
        fc1 = tf.keras.layers.Dense(256, activation='relu',
                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01),
                                    bias_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))(concat)
        fc1 = tf.keras.layers.Dropout(0.3)(fc1)
        outputs = tf.keras.layers.Dense(4, activation=None,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01),
                                        bias_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01))(fc1)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def load_weights(self):
        """Load pre-trained weights from CSV files."""
        # Load convolutional layer weights
        conv1_layer1_weights = np.array(pd.read_csv('model/conv1_layer1_weights.csv', header=None))
        conv1_layer2_weights = np.array(pd.read_csv('model/conv1_layer2_weights.csv', header=None))
        conv2_layer1_weights = np.array(pd.read_csv('model/conv2_layer1_weights.csv', header=None))
        conv2_layer2_weights = np.array(pd.read_csv('model/conv2_layer2_weights.csv', header=None))
        
        # Load biases
        conv1_layer1_biases = np.array(pd.read_csv('model/conv1_layer1_biases.csv', header=None)).flatten()
        conv1_layer2_biases = np.array(pd.read_csv('model/conv1_layer2_biases.csv', header=None)).flatten()
        conv2_layer1_biases = np.array(pd.read_csv('model/conv2_layer1_biases.csv', header=None)).flatten()
        conv2_layer2_biases = np.array(pd.read_csv('model/conv2_layer2_biases.csv', header=None)).flatten()
        
        # Load fully connected layer weights
        fc_layer1_weights = np.array(pd.read_csv('model/fc_layer1_weights.csv', header=None))
        fc_layer2_weights = np.array(pd.read_csv('model/fc_layer2_weights.csv', header=None))
        
        # Load biases for fully connected layers
        fc_layer1_biases = np.array(pd.read_csv('model/fc_layer1_biases.csv', header=None)).flatten()
        fc_layer2_biases = np.array(pd.read_csv('model/fc_layer2_biases.csv', header=None)).flatten()
        
        # Set weights to the model layers
        self.model.get_layer('conv2d').set_weights([conv1_layer1_weights, conv1_layer1_biases])
        self.model.get_layer('conv2d_1').set_weights([conv1_layer2_weights, conv1_layer2_biases])
        self.model.get_layer('conv2d_2').set_weights([conv2_layer1_weights, conv2_layer1_biases])
        self.model.get_layer('conv2d_3').set_weights([conv2_layer2_weights, conv2_layer2_biases])
        self.model.get_layer('dense').set_weights([fc_layer1_weights, fc_layer1_biases])
        self.model.get_layer('dense_1').set_weights([fc_layer2_weights, fc_layer2_biases])
    
    def get_action(self, state):
        """Get action for the current state.
        
        Args:
            state: Current state (4x4x16 one-hot encoded)
            
        Returns:
            action: Integer in [0, 1, 2, 3] representing [up, left, right, down]
        """
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)