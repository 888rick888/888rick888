import numpy as np
from tensorflow.python.keras.engine.data_adapter import ALL_ADAPTER_CLS
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.saving.save import load_model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Flatten, Add, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializations import normal
import tensorflow.keras.backend as backend
from collections import deque
tf.compat.v1.disable_eager_execution()

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 500

class ActorNet(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        # tf.compat.v1.keras.backend.set_session(sess)

        self.actor_model, self.state_input, self.actor_model_weights = self.create_actor_network(state_size, action_size)
        self.target_action_model, self.target_state_input, self.target_action_model_weights = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None,action_size])
        self.actor_grads = tf.gradients(self.actor_model.output, self.actor_model_weights, -self.action_gradient)
        grads = zip(self.actor_grads, self.actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)

        self.graph = tf.compat.v1.get_default_graph()

        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state_input: states,
            self.action_gradient:action_grads
        })

    def target_train(self):
        actor_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_action_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1- self.TAU)*actor_target_weights[i]
        self.target_action_model.set_weights(actor_target_weights)


    def create_actor_network(self, state_size, action_size):
        S_input_A = Input(shape=[state_size, state_size, 3])
        AA_1 = Xception(weights=None, include_top=False )(S_input_A)
        AA_2 = GlobalAveragePooling2D()(AA_1)
        AA_3 = Flatten()(AA_2)
        h1 = Dense(HIDDEN1_UNITS, activation='relu')(AA_3)
        h2 = Dense(HIDDEN2_UNITS, activation='relu')(h1)
        Steering = Dense(1,activation='tanh')(h2)
        Acceleration = Dense(1,activation='sigmoid')(h2)
        Brake = Dense(1,activation='sigmoid')(h2)
        action_output = Concatenate()([Steering, Acceleration, Brake])
        model_A = Model(inputs=S_input_A, outputs=action_output)
        print(model_A.summary())
        model_A.reset_states()
        model_A.reset_metrics()
        model_A.compile(loss="mse",optimizer=Adam(learning_rate=0.001),metrics=["acc"])
        return model_A, S_input_A, model_A.trainable_weights

