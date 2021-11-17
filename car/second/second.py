#-- first.py ----> second.py   @pzs                         --
#-- use DDPG algorithm in carla simulator                   --
#--              --
from functools import lru_cache
import glob
import os
import sys
import time
import random
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
import json

import tensorflow.compat.v1.keras.backend as backend

from UE4 import *
from A_NET import ActorNet
from C_NET import CriticNet
from ReplayBuffer import ReplayBuffer
from UE4 import CarEnv


try:
    sys.path.append(glob.glob('D:\pzs\setup\CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# SECONDS_PER_EPISODE = 10
MEMORY_FRACTION = 0.6

def JUSTDOIT():
    MODEL_NAME = "DDPG_carla"
    train_indicator = 0

    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 32
    TAU = 0.001
    LRA = 0.0001
    LRC = 0.001

    action_dimesion = 3
    state_dimesion = IM_HEIGHT
    
    random.seed(1)
    np.random.seed(1)

    epsilon = 1
    EPSILON_DECAY = 0.92
    MIN_EPSILON = 0.001

    episode_count = 2000
    max_steps = 100000
    done = False
    step = 1
    
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.compat.v1.keras.backend.set_session(sess)
    
    tf.compat.v1.keras.backend.get_session().run(tf.global_variables_initializer())

    if not os.path.isdir("models_second"):
        os.makedirs("models_second")

    env = CarEnv()
    Critic = CriticNet(sess, state_dimesion, action_dimesion, BATCH_SIZE, TAU, LRC)
    Actor = ActorNet(sess, state_dimesion, action_dimesion, BATCH_SIZE, TAU, LRA)
    buff = ReplayBuffer(BUFFER_SIZE)

    try:
        Actor.actor_model = load_model(f'models_second/actormodel.h5')
        Actor.target_action_model = load_model(f'models_second/actormodel.h5')
        Critic.critic_model = load_model(f'models_second/criticmodel.h5')
        Critic.target_critic_model = load_model(f'models_second/criticmodel.h5')
    except:
        print('--------Can not load the models!!--------')

    for episode in range(episode_count):
        print("Episode : " + str(episode) + " Replay Buffer " + str(buff.count()))

        env.collision_hist = []
        # episode_start = time.time()
        total_reward = 0

        s_t = env.reset()

        for i in range(max_steps):

            a_t = np.zeros([1,action_dimesion])


            if np.random.random() > epsilon:
                a_t = Actor.actor_model.predict([[s_t]])
                a_t = np.argmax(a_t)
            else:
                a_t = np.random.randint(0,3)

            s_t = np.asarray(s_t)
            a_t = np.argmax(Actor.actor_model.predict(s_t))

            s_t1,r_t,done,_ = env.step(a_t)

            step +=1
            buff.add(s_t, a_t, r_t, s_t1,done)

            batch = buff.getbatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([ [e[1]] * action_dimesion for e in batch])
            new_states = np.asarray(new_states)

            # with Critic.graph.as_default(): 
            s_t_a = Actor.actor_model.predict(new_states)
            target_q_values = Critic.critic_model.predict([new_states, s_t_a])

            for k in range(len(batch)):
                rs_list = [rewards[k]]*len(target_q_values[k])
                if dones[k]:
                    y_t[k] = rs_list
                else:
                    # target_q_values[k].reshape((1,1))
                    y_t[k] = rs_list + GAMMA*target_q_values[k]

            loss = 0
            if(train_indicator):
                loss += Critic.critic_model.train_on_batch([states, actions],y_t)
                a_for_grad = Actor.actor_model.predict(states)
                grads = Critic.gradients(states, a_for_grad)
                Actor.train(states, grads)
                Actor.target_train()
                Critic.target_train()

            total_reward += r_t
            s_t = s_t1
            print("--Episode:", episode, "--Step:", step, "--Action:", a_t, "--Reward:", r_t, "--Loss:", loss)

            if done:
                break

        for actor in env.actor_list:
            actor.destroy()

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON,epsilon)
        
        if episode % 5 == 0:
            Actor.actor_model.save(f"models_second/actormodel.h5")
            Critic.critic_model.save(f"models_second/actormodel.h5")
            print('-------------model saved-------------')

if __name__ =='__main__':
    JUSTDOIT()

    