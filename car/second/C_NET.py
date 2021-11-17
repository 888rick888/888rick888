from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.saving.save import load_model

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Flatten, Add, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as backend
 
from collections import deque
import numpy as np

tf.compat.v1.disable_eager_execution()

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 500

class CriticNet(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        # tf.compat.v1.keras.backend.set_session(sess)

        self.critic_model, self.action_input, self.state_input = self.create_critic_network(state_size, action_size)
        self.target_critic_model, self.target_critic_action_input, self.target_critic_state_input = self.create_critic_network(state_size,action_size)
        self.action_grads = tf.gradients(self.critic_model.output, self.action_input)

        self.graph = tf.compat.v1.get_default_graph()

        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads,feed_dict={
            self.state_input: states,
            self.action_input: actions
        })[0]
    
    def target_train(self):
        critic_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1-self.TAU) * critic_target_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_size):
        S_input_S = Input(shape=[state_size,state_size,3])
        SS = Xception(weights=None, include_top=False)(S_input_S)
        SS_1 = GlobalAveragePooling2D()(SS)
        SS_2 = Flatten()(SS_1)
        S1 = Dense(HIDDEN1_UNITS, activation='relu')(SS_2)
        S2 = Dense(HIDDEN2_UNITS, activation='linear')(S1)
        A_input = Input(shape=[action_size])
        A1 = Dense(HIDDEN2_UNITS, activation='linear')(A_input)
        merged = Concatenate()([S2, A1])
        h1 = Dense(HIDDEN2_UNITS,activation='relu')(merged)
        critic_output = Dense(action_size, activation='linear')(h1)
        model = Model(inputs=[S_input_S,A_input],outputs=critic_output)
        print(model.summary())
        model.reset_states()
        model.reset_metrics()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.LEARNING_RATE))
        return model, A_input, S_input_S

    def get_qs(self,state):
        return self.critic_model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]