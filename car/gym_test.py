import os
import time
 
import numpy as np
 
import gym
from tensorflow.python.keras.saving.save import load_model
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
 
import random
from collections import deque

tf.compat.v1.disable_eager_execution()



def stack_samples(samples):
    array = np.array(samples, dtype=object)
    s_ts = np.stack(array[:,0]).reshape((array.shape[0],-1))
    actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
    rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
    s_ts1 = np.stack(array[:,3]).reshape((array.shape[0],-1))
    dones = np.stack(array[:,4]).reshape((array.shape[0],-1))
    return s_ts, actions, rewards, s_ts1, dones
 
class Agent(object):
    def __init__(self,sess):
        self.sess = sess
        self.epsilon = 0.9
        self.gamma = 0.90
        self.epsilon_decay = 0.95
        self.tau = 0.02
        self.memory = deque(maxlen=4000)
 
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        self.actor_critic_grad = tf.placeholder(tf.float32,[None,1])
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,actor_model_weights,-self.actor_critic_grad)
        grads = zip(self.actor_grads,actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(0.0001).apply_gradients(grads)
 
        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)
 
        self.sess.run(tf.initialize_all_variables())
 
    def create_actor_model(self):
        state_input = Input(shape=(4,))
        h1 = Dense(300,activation='relu')(state_input)
        h2 = Dense(400,activation='relu')(h1)
        output = Dense(1,activation='tanh')(h2)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.0001)
        model.compile(loss="mse",optimizer=adam,metrics=["acc"])
        return state_input, model
    
    def create_critic_model(self):
        state_input = Input(shape=(4,))
        state_h1 = Dense(300,activation='relu')(state_input)
        state_h2 = Dense(400)(state_h1)
        action_input = Input(shape=(1,))
        action_h1 = Dense(400)(action_input)
        merged = Concatenate()([state_h2,action_h1])
        merged_h1 = Dense(500, activation='relu')(merged)
        output = Dense(1,activation='linear')(merged_h1)
        model = Model(inputs=[state_input,action_input], outputs=output)
        adam = Adam(lr=0.0001)
        model.compile(loss="mse",optimizer=adam)
        return state_input, action_input, model
 
    def remember(self,s_t,action,reward,s_t1,done):
        self.memory.append([s_t,action,reward,s_t1,done])
 
    def train(self):
        batch_size = 256
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        self.samples = samples
        self.train_critic(samples)
        self.train_actor(samples)
    
    def train_actor(self,samples):
        s_ts, _, _, _, _ = stack_samples(samples)
        predicted_actions = self.actor_model.predict(s_ts)
        grads = self.sess.run(self.critic_grads,feed_dict={
                self.critic_state_input: s_ts,
                self.critic_action_input: predicted_actions
        })[0]
        self.sess.run(self.optimize,feed_dict={
                self.actor_state_input: s_ts,
                self.actor_critic_grad: grads
        })
 
    def train_critic(self,samples):
        s_ts, actions, rewards, s_ts1, dones = stack_samples(samples)
        target_actions = self.target_actor_model.predict(s_ts1)
        futurn_rewards = self.target_critic_model.predict([s_ts1,target_actions]) 

        rewards += self.gamma * futurn_rewards * (1-dones)
        self.critic_model.fit([s_ts,actions], rewards, verbose=0)
    
    def update_target(self):
        self.update_actor_target()
        self.update_critic_target()
    
    def update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)
 
    def update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)
 
    def act(self,s_t):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.actor_model.predict(s_t)*2 + np.random.normal()
        return self.actor_model.predict(s_t)*2
 
if __name__ == "__main__":
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
 
    # env = gym.make('InvertedPendulum-v2')
    env = gym.make('CartPole-v1')
 
    agent = Agent(sess)
    agent.actor_model=load_model(f"gym_actor.h5")
    agent.critic_model=load_model(f"gym_critic.h5")
    print("------------------------------- load model ----------------------------------")

    adam = Adam(lr=0.0001)
    agent.critic_model.compile(loss="mse",optimizer=adam)
    agent.actor_model.compile(loss="mse",optimizer=adam)

    step = 0
    for e in range(10000):
        obs = env.reset()
        s_t = obs.reshape((1,4))
        epoch_reward = 0
        start = time.time()
        
        for t in range(1000):
            env.render()
            step += 1
            action = agent.act(s_t)
            action = action.reshape((1,1))
            if action < 0:
                action = 0
            elif action > 0:
                action = 1
            obs_, reward, done, info = env.step(action)

            r_t = obs_[2]
            s_t1 = obs_.reshape((1,4))

            if abs(r_t) < 0.02:
                reward = 10.0
            elif abs(r_t) < 0.15:
                reward = 4.0
            elif abs(r_t) < 0.3:
                reward = 1.5
            elif abs(r_t) >0.3:
                reward = -3.0
            elif abs(r_t) >0.4:
                reward = -10.0
            # reward += (time.time()- start)
            epoch_reward +=reward
            agent.remember(s_t, action, reward, s_t1, done)

            #train
            agent.train()
            agent.update_target()
 
            print("episode:{} step:{} action:{} reward:{}".format(e,step,action,reward))
        
            s_t = s_t1
            if done:
                break

        if e%5 == 0:
            # agent.actor_model.save(f"gym_actor.h5", include_optimizer=False)
            # agent.critic_model.save(f"gym_critic.h5", include_optimizer=False)
            print("-----------save model-----")

        print("episode:{} step:{} action:{}  OOOOOtotal reward:{}".format(e,step,action,epoch_reward))