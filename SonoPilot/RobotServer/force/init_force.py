#encoding:utf-8
#from PyQt5.QtCore import *
import rospy, sys
#import moveit_commander
#from geometry_msgs.msg import PoseStamped, Pose
from franka_core_msgs.msg import EndPointState

#运行rostopic type查看对应的话题，得到robotiq_ft_sensor/ft_sensor  
#因此直接 from robotiq_ft_sensor.msg import ft_sensor，得到该话题的信息格式，用于接受该话题
from robotiq_ft_sensor.msg import ft_sensor  

from math import pi
#from tf.transformations import quaternion_from_euler
import numpy as np
import pandas as pd
#from PyQt5.Qt import *#

import numpy as np
import pyqtgraph as pq

import imagezmq
import traceback
import time
import simplejpeg
import copy
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
import threading


class ForceCalibrationThread(QThread):
    finished_signal = pyqtSignal(str) 

    def __init__(self):
        super(ForceCalibrationThread, self).__init__()
        rospy.Subscriber('/robotiq_ft_sensor', ft_sensor, self.forceCallback)

    # def listener_force(self):
    #     ## 一个节点需要发送多个话题和接受多个话题，如机器人阻抗控制，需要同时接受目标检测模型发送过来的目标位置和机械臂末端六维力的数据，
    #     ##由于Python中没有ros::spinOnce，用rospy.spin()会阻塞程序，主程序无法执行其他模块，所以需要引入线程来处理

    #     # 订阅话题
    #     self.force_subscriber() #末端力

    #     # 运行新线程,收话题线程
    #     t1 = threading.Thread(target=self.thread_spin)  # 末端位置订阅线程
    #     t1.start()

    # def force_subscriber(self):
    #     rospy.Subscriber('/robotiq_ft_sensor', ft_sensor, self.forceCallback)
    # def thread_spin(self):
        rospy.spin()

    def forceCallback(self, msg):        
        #后面加减，进行校准
        #力
        self.Fx = msg.Fx
        self.Fy = msg.Fy
        self.Fz = msg.Fz

        #扭矩
        self.Mx = msg.Mx
        self.My = msg.My
        self.Mz = msg.Mz


    def calculate_force_zero(self):
        self.all_num = 1500

        self.value_list_x = []
        self.value_list_y = []
        self.value_list_z = []
        self.value_list_mx = []
        self.value_list_my = []
        self.value_list_mz = []
        for i in range(self.all_num):
            self.value_list_x.append(self.Fx)
            self.value_list_y.append(self.Fy)
            self.value_list_z.append(self.Fz)
            self.value_list_mx.append(self.Mx)
            self.value_list_my.append(self.My)
            self.value_list_mz.append(self.Mz)

        #a = np.mean(value_list_x)
        x = np.mean(self.value_list_x)
        y = np.mean(self.value_list_y)
        z = np.mean(self.value_list_z)
        mx = np.mean(self.value_list_mx)
        my = np.mean(self.value_list_my)
        mz = np.mean(self.value_list_mz)
        # #np.mean([0.1, 0.2, 2,3,4,5,6,7,8])

        df = pd.DataFrame([{'x': x,
                                                'y': y,
                                                'z': z,
                                                'mx': mx,
                                                'my': my,
                                                'mz': mz}])
        df.to_csv('/home/uax/LiMD_example/Robot_arm/R_UI/force/force_zero.csv', index=False)
        print('Finished.')


    def finished(self):
        rospy.signal_shutdown('closed!')

if __name__ == '__main__':
    window = ForceCalibration()
    window.init_node()
    time.sleep(1)
    window.listener_force()
    time.sleep(1)
    window.calculate_force_zero()
    window.finished()
