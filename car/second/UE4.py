import glob
import os
import sys
import time
import random
import numpy as np
import cv2
import math

from tensorflow.python.keras.utils.generic_utils import default

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
SECONDS_PER_EPISODE = 10

MAX_V = 50
MIN_V = 30

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    
    def __init__(self):
        self.client = carla.Client("localhost",2000)
        self.client.set_timeout(4.0)
        self.world = self.client.reload_world() # 重启世界
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

        self.transform = random.choice(self.world.get_map().get_spawn_points()[20:30])
        # self.transform = self.world.get_map().get_spawn_points()
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
        
        lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanesensor = self.world.spawn_actor(lanesensor,transform,attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        self.lanesensor.listen(lambda event: self.lane_type(event))


        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))

        return self.front_camera

    def collision_data(self,event):
        self.collision_hist.append(event)

    def lane_type(self, event):
        self.lane_hist.append(event)
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.text = text
    
    
    def _on_invasion(self, event):
        # self = weak_self()
        # if not self:
        #     return
        # lane_types = set(x.type for x in event.crossed_lane_markings)
        # text = ['%r' % str(x).split()[-1] for x in lane_types]
        # self.hud.notification('Crossed line %s' % ' and '.join(text))
        self.lane_hist.append(event)
        # print(event)

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
        v =  self.vehicle.get_velocity()
        kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))

        a_steer = action[0]
        a_accelerate = action[1]
        a_brake = action[2]

        self.vehicle.apply_control(carla.VehicleControl(throttle=a_accelerate,steer=a_steer*self.STEER_AMT))

        # if action == 0:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=a_steer*self.STEER_AMT))
        # elif action == 1:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0))
        # elif action == 2:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=a_steer*self.STEER_AMT))

        
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh > 50 & kmh <30:
            done = False
            reward = -1
        elif self.text[0] == "'Solid'" :
            done = False
            reward = -20
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE <time.time():
            done = True
        return self.front_camera,reward,done,None

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))
