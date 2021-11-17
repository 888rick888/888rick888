
#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time
import random
import numpy as np
import cv2
import weakref

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

IM_WIDTH=640
IM_HEIGHT=480



class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        # self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        # self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
        self.sensor.listen((lambda event: self.lane_data(event)))
        self.lanedec = []

    def lane_date(self, event):
        self.lanedec.append(event)
        print(self.lanedec)

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


def process_img(image):
    i = np.array(image.raw_data)
    # print(i.shape)
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4))
    i3 = i2[:,:,:3]
    cv2.imshow("",i3)
    cv2.waitKey(1)
    return i3/255.0

def lane_type( event):
    lanedec.append(event.crossed_lane_markings)
    lane_types = set(x.type for x in event.crossed_lane_markings)
    text = ['%r' % str(x).split()[-1] for x in lane_types]
    # print(text)
    # print(text[0])
    # print(len(text[0]),type(text[0]))
    if text[0] == "'Solid'" :
        print('+_+_+_+_+_+_')
    elif len(text[0]) == 7:
        print('===============')


actor_list=[]
lanedec = []

try:
    client=carla.Client('localhost',2000)
    client.set_timeout(5.0)

    world=client.get_world()
    blueprint_library=world.get_blueprint_library()

    bp=blueprint_library.filter("model3")[0]
    # print(bp)
   

    # spawn_point_raw = world.get_map().get_spawn_points()
    # print(spawn_point_raw)
    
    spawn_point=random.choice(world.get_map().get_spawn_points())

    vehicle=world.spawn_actor(bp,spawn_point)
    # vehicle.set_autopilot(True)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.1))
    actor_list.append(vehicle)
    
    
    cam_bp=blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov","110")
    
    spawn_point= carla.Transform(carla.Location(x=2.5,z=0.7))



    sensor=world.spawn_actor(cam_bp,spawn_point,attach_to=vehicle)
    actor_list.append (sensor)
    sensor.listen(lambda data:process_img(data))

    
    lanesensor = world.get_blueprint_library().find('sensor.other.lane_invasion')
    sensor_lane = world.spawn_actor(lanesensor, spawn_point, attach_to=vehicle)
    actor_list.append(lanesensor)
    sensor_lane.listen((lambda event: lane_type(event)))
    # lane_types = set(x.type for x in event.crossed_lane_markings)

    time.sleep(10)



finally:
    # for a in actor_list:
    #     a.destroy()
    actor_list.clear()
    print('All cleaned up ~')

