# ...


# .step(action)


# ...

# def reset(self):

# def step(self,action):
#     return obs,reward,done,extra_info

import glob
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import time
import random
import numpy as np
import cv2
import math
from collections import deque
import weakref

import pickle

# import tensorflow as tf
import tensorflow.compat.v1 as tf
print(tf.test.is_gpu_available())
tf.disable_v2_behavior()

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import EfficientNetB7 as EfficientNet
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tqdm import tqdm

import tensorflow.compat.v1.keras.backend as backend
from threading import Thread

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(glob.glob('D:\pzs\setup\CARLA_0.9.11\WindowsNoEditor\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW = True
IM_WIDTH = 224
IM_HEIGHT = 224
IMG_HEIGHT = IM_HEIGHT
IMG_WIDTH = IM_WIDTH
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 40
MIN_REPLAY_MEMORY_SIZE=10
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Efficient"

MEMORY_FRACTION = 0.6
MIN_REWARD = -200

DISCOUNT = 0.98
EPISODE = 100

epsilon = 1
EPSILON_DECAY = 0.92

MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs) 

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        # self._write_logs(stats, self.step)
        pass



class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    
    def __init__(self):
        self.client = carla.Client("localhost",2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        
        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

        

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.lane_hist = []

        self.transform = random.choice(self.world.get_map().get_spawn_points()[10:50])
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)

        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x",f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y",f"{self.im_height}")
        self.rgb_cam.set_attribute("fov",f"110")

        transform = carla.Transform(carla.Location(x=2.5,z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam,transform,attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor,transform,attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        # lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        # self.lanesensor = self.world.spawn_actor(lanesensor,transform,attach_to=self.vehicle)
        # self.actor_list.append(self.lanesensor)
        # weak_self = weakref.ref(self)
        # self.lanesensor.listen(lambda event: self._on_invasion(event))


        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))

        return self.front_camera

    def collision_data(self,event):
        self.collision_hist.append(event)
    
    
    def _on_invasion(self, event):
        # self = weak_self()
        # if not self:
        #     return
        # lane_types = set(x.type for x in event.crossed_lane_markings)
        # text = ['%r' % str(x).split()[-1] for x in lane_types]
        # self.hud.notification('Crossed line %s' % ' and '.join(text))
        self.lane_hist.append(event)
        # print(event)


    # def process_img(self,image):
    #     i = np.array(image.raw_data)
    #     print(i.shape)
    #     i2 = i.reshape((self.im_height,self.im_width,4))
    #     i3 = i2[:,:,:3]
    #     if self.SHOW_CAM:
    #         cv2.imshow("",i3)
    #         cv2.waitKey(1)
    #     return i3/255.0

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3
        
    def step(self,action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=1*self.STEER_AMT))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=-0.5*self.STEER_AMT))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0.5*self.STEER_AMT))

        v =  self.vehicle.get_velocity()
        kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh > 50 & kmh <30:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE <time.time():
            done = True
        return self.front_camera,reward,done,None

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False

        self.last_logged_episode = 0
        self.training_initialized = False
    

    # def create_model(self):
    #     base_model = Xception(weights=None,include_top=False,input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
    #     x = base_model.output
    #     x = GlobalAveragePooling2D()(x)
    #     predictions = Dense(3,activation="linear")(x)
    #     model = Model(inputs = base_model.input,outputs=predictions)
    #     model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["acc"])
    #     return model

    def create_model(self):
        base_model = EfficientNet(include_top=False, input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(3,"sigmoid")(x)
        model = Model(inputs = base_model.input, outputs = predictions)
        model.compile(loss="mse",optimizer=Adam(learning_rate=0.005),metrics=["acc"])
        return model


    def update_replay_memory(self,transition):
        # transition = (current_state,action,reward,new_state,done)
        self.replay_memory.append(transition)
    
    def train(self):
        if len(self.replay_memory)<MIN_REPLAY_MEMORY_SIZE:
            return 
        minibatch = random.sample(self.replay_memory,MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states,PREDICTION_BATCH_SIZE)
            
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states,PREDICTION_BATCH_SIZE)
            
        x = []
        y = []
        
        for index,(current_state,action,reward,new_state,done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
        
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)
            
        
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step
            
        with self.graph.as_default():
            backend.get_session().run(tf.global_variables_initializer())
            self.model.fit(np.array(x)/255,np.array(y),batch_size=TRAINING_BATCH_SIZE,verbose=0,shuffle=False,callbacks=[self.tensorboard] if log_this_step else None)
        if log_this_step:
            self.target_update_counter +=1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self,state):
        return self.model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]

    def train_in_loop(self):
        x = np.random.uniform(size = (1,IMG_HEIGHT, IM_WIDTH,3)).astype(np.float32)
        y = np.random.uniform(size=(1,3)).astype(np.float32)
        with self.graph.as_default():
            backend.get_session().run(tf.global_variables_initializer())
            self.model.fit(x,y,verbose=False,batch_size=1)
        
        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)




if __name__ =='__main__':
    FPS = 15
    ep_rewards = [-200]

    random.seed(1)
    np.random.seed(1)
    # tf.set_random_seed(1)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    
    # backend.set_session(tf.Session())
    # backend.set_session(tf.Session(config=tf.ConfigProto()))

    if not os.path.isdir("models"):
        os.makedirs("models")

    agent = DQNAgent()
    env = CarEnv()
    print("----------Begin Thread")
    trainer_thread = Thread(target=agent.train_in_loop,daemon=True)
    trainer_thread.start()

    print("------------Sleep")
    while not agent.training_initialized:
        time.sleep(0.01)

    print("-----------Get QS")
    agent.get_qs(np.ones((env.im_height,env.im_width, 3)))
    # print("check this outOOOOOOOOOOOOOOOOOOOOOO",agent.get_qs(np.ones((env.im_height,env.im_width, 3))))
   

    print("-----------Begin Episode")
    for episode in tqdm(range(1,EPISODE+1),ascii=True,unit="episodes"):
        env.collision_hist = []
        agent.tensorboard.step  = episode
        episode_reward=0
        step=1
        current_state = env.reset()
        done = False
        episode_start = time.time()
        
        while True:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0,5)
                time.sleep(1/FPS)

            new_state,reward,done,_ = env.step(action)

            episode_reward += reward
            agent.update_replay_memory((current_state,action,reward,new_state,done))

            step += 1
            if done:
                break
        for actor in env.actor_list:
            actor.destroy()

        #Append spisode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            #Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg__{min_reward:_>7.2f}min__{int(time.time())}.model')

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON,epsilon)

    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'model/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg__{min_reward:_>7.2f}min__{int(time.time())}.model')
    

