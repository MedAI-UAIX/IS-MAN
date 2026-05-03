#!/usr/bin/env python2.7
#encoding:utf-8 
"""
v1
初步实现扫查

Thyroid_scanning_v1_multiThread
增加3D重建多线程采集

Thyroid_scanning_v1_multiThread_lxx
增加探头接触检测

Thyroid_scanning_v1_1_multiThread
增加扫查中断

Thyroid_scanning_v1_2_multiThread
横切的时候，先保证找到上级。如果一开始就有甲状腺，则机械臂应该继续往头侧移动
增加纵切面扫查
增加数据状态的记录

Thyroid_scanning_v1_2_multiThread_agent
跟 Thyroid_scanning_v1_2_multiThread 功能一样，更换成Agent控制而已

"""

import sys
sys.path.insert(1, '/home/uax/cv_bridge_ws/install/lib/python3/dist-packages')
sys.path.insert(2, '/home/uax/cv_bridge_ws/devel/lib/python3/dist-packages')
# sys.path.insert(3, '/home/uax/tf_ws/install/lib/python3/dist-packages')
# sys.path.insert(4, '/home/uax/tf_ws/devel/lib/python3/dist-packages')
sys.path.append('/home/usai/auto_RUSS/R_10_ROS_PositionForce')
from cv_bridge import CvBridge, CvBridgeError
from ForcePosition_calib import MoveItFranka
import rospy
import time
import threading
import copy
from std_msgs.msg import Float64, Float64MultiArray
import pandas as pd
# 运行rostopic type查看对应的话题，rostopic type /depth_to_rgb/image_raw
# 返回 sensor_msgs/Image 
# 因此直接 from sensor_msgs.msg import Image，得到该话题的信息格式，用于接受该话题
from robotiq_ft_sensor.msg import ft_sensor 
from sensor_msgs.msg import Image
import numpy as np
# import PIL.Image
import cv2
import os 
import traceback
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget, QAbstractItemView, QTableWidgetItem, QLabel
from PyQt5.QtGui import QImage, QPixmap, QTransform
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5  import QtWidgets
from R_20_Recording_Status.pub_recording_state import ContinuousPublisher, CheckPoint    #  是否采集扫查数据


    
class SavePoseImage_Thread(QThread):#线程类    
    """
    目的: 开启新的线程保存pose和对应的US图像, 用于后续3D重建, 防止

    参数: 
    franka: franka机械臂的类
    fps: 频率Hz, 控制每秒有多少对数据被存储 
    save_path: 数据存储路径
    """
    def __init__(self, franka, fps=2, scanning_position='', save_path=''):
        super(SavePoseImage_Thread, self).__init__()
        self.sleep_time = 1/fps
        self.save_path = save_path
        self.scanning_position = scanning_position
        self.pose = []
        self.img_num = 1

        self.unet_cv_img = None
        self.franka = franka

        self.is_pause = True  # 是否暂停记录
        self.start_save = True  # 是否持续记录的标记符

    def run(self): #线程执行函数
        self.pose = []
        self.img_num = 1
        while self.start_save:
            try:
                if not self.is_pause:
                    self.record_pose_img()
                    self.save_pd()
                    time.sleep(self.sleep_time)
                else:
                    time.sleep(1)  #如果是暂停模式, 则休眠1s后重新检测
            except:
                print('数据pose+Image保存的线程出现问题, 请检查')
                self.save_pd()
                print(traceback.format_exc())
        self.save_pd()

    def pause_save(self):
        """暂停保存数据"""
        self.is_pause = True

    def continue_save(self):
        """继续保存数据"""
        self.is_pause = False

    def end_and_save(self):
        """结束线程， 保存数据"""
        self.start_save = False
        self.save_pd()
        print('总共有{}条数据'.format(len(self.pose)))

    def record_pose_img(self):
        """
        记录对应的超声图像与探头姿态, 进行3D重建
        """
        actual_pose = copy.deepcopy(self.franka.current_pose)
        cv2.imwrite(os.path.join(self.save_path, '3D/img/{:0>4}_{}.png'.format(self.img_num, self.scanning_position)), self.unet_cv_img)
        pose = self.encode_TransMatrix2string(actual_pose)
        self.pose.append({'img_num': self.img_num,
                                                'pose': pose})
        self.img_num += 1
    
    def save_pd(self):
        df = pd.DataFrame(self.pose)
        df.to_csv(os.path.join(self.save_path, '3D/posemulti.csv'), index=False, mode='w')

    def encode_TransMatrix2string(self, TransMatrix):
        """numpy的4*4位姿矩阵转成空格的字符串"""
        vactor = TransMatrix.T.reshape(1, -1)
        string = ''
        for i in range(vactor.shape[1]):
            string = string + str(vactor[0, i]) + ' '
        string = string.strip()
        return string
 
    

class UltrasoundServo:
    def __init__(self, client_cmd, goal_force=2, init_node=True, 
                                        save_path='') :
        self.save_path = save_path
        if not os.path.exists(os.path.join(self.save_path, '3D/img')):
            os.makedirs(os.path.join(self.save_path, '3D/img'))
        self.bridge = CvBridge()
        self.franka = MoveItFranka(init_node=init_node)
        self.recording_state = ContinuousPublisher(need_init_node=False)
        self.check_point = CheckPoint()
        self.save_pose_image = SavePoseImage_Thread(self.franka, fps=2, scanning_position='', save_path=self.save_path)   #多线程, 保存位置和图像
        self.save_pose_image.start()  #但是此时还没开始记录
        time.sleep(2)

        self.client_cmd = client_cmd    # 发送指令/机器人状态给智能体
        self.goal_force = goal_force #目标压力，单位N
        # 定义队列的最大容量，用来存储接触力
        max_size = 1000
        # 创建一个具有固定容量的队列
        self.force_queue_x = deque(maxlen=max_size)
        self.force_queue_My = deque(maxlen=max_size)

        self.listener_ultrasound()  #另开新线程

        self.area_dict = {}  #存储 面积-位姿 信息
        self.scanning_position = ''   # 扫查部位left, right, mid，便于重建
        self.img_num = 1
        self.pose = []
        self.stop_scanning_flag = False
        self.baseline_time = 0.1 # 基线执行时间
        time.sleep(2)

        ## ======== phantom
        # self.left_pose_TransMatrix = np.array([[ 0.99817081,  0.02219032,  0.05623728,  0.5473916 ],
        #                                                         [ 0.05704044, -0.65394205, -0.75439127,  0.23021345],
        #                                                         [ 0.02003574,  0.75621914, -0.65401161,  0.29505653+0.1],
        #                                                         [ 0.        ,  0.        ,  0.        ,  1.           ]])
        # self.right_pose_TransMatrix = np.array([[   0.97942774, -0.13227697,  0.1523946 ,  0.51257617],
        #                                                             [-0.20178906, -0.64789906,  0.73451207, -0.15504545],
        #                                                             [ 0.00157729, -0.75015306, -0.66126235,  0.29904153+0.1],
        #                                                             [ 0.        ,  0.        ,  0.        ,  1.         ]])
        # self.mid_pose_TransMatrix = np.array([[0.99984464,  0.01703688, -0.00452129,  0.6022697 ],
        #                                                         [ 0.01678789, -0.99859114, -0.05033791,  0.04298014],
        #                                                         [-0.00537252,  0.05025419, -0.99872201,  0.39162972+0.1],
        #                                                         [ 0.        ,  0.        ,  0.        ,  1.        ]])

    def listener_ultrasound(self):
        #订阅话题
        self.contact_subscriber()  #订阅接触面像素强度话题
        self.center_subscriber()  #订阅图像面积，目标位置话题
        self.unet_img()    #订阅分割出的结果
        self.pose_subscriber()   #关键点检测结果
        self.force_subscriber()  # 接触力
        
        # 运行新线程，接收话题
        t1 = threading.Thread(target=self.thread_spin)  # 订阅线程
        t1.start()

    def force_subscriber(self):
        rospy.Subscriber('/robotiq_ft_sensor', ft_sensor, self.force_sensor_Callback)
    def force_sensor_Callback(self, msg):       
        Fx = msg.Fx 
        # Fy = msg.Fy 
        # Fz = -(msg.Fz)
        # Mx = msg.Mx
        My = msg.My
        # Mz = msg.Mz
        self.force_queue_x.append(Fx)
        self.force_queue_My.append(My)

    def contact_subscriber(self):
        rospy.Subscriber('/ultra_contact', Float64MultiArray, self.contact_Callback)
    def contact_Callback(self, msg):
        self.right_value = msg.data[0]
        # self.mid_value = msg.data[1]
        self.left_value = msg.data[1]
        # print(self.left_value, self.right_value)

    def center_subscriber(self):
        rospy.Subscriber('/unet_area', Float64MultiArray, self.center_Callback)
    def center_Callback(self, msg):
        #最大面积，目标中心点位置，目标在图像的最左侧，目标在图像的最右侧，全图像宽度，有效图像宽度起始，有效图像宽度终止
        self.area = msg.data[6]
        # self.target_cent = msg.data[1]
        # self.target_left = msg.data[2]
        # self.target_right = msg.data[3]
       
        # self.img_ultrasound_start =msg.data[5]   #超声区域图像的起始
        # self.img_ultrasound_end =msg.data[6]   #超声区域图像的终止
        self.thyroid_start_x, self.thyroid_w, self.qiguan_x, self.qiguan_w = msg.data[0:4]
        self.img_ultrasound_start =msg.data[4]   #超声区域图像的起始
        self.img_ultrasound_end =msg.data[5]   #超声区域图像的终止

    def unet_img(self):
        rospy.Subscriber('/Unet/pred', Image, self.unet_callback)
    def unet_callback(self, data):
        self.unet_cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")   #8UC1
        self.save_pose_image.unet_cv_img = copy.deepcopy(self.unet_cv_img)  #更新多线程里面的图像

    def pose_subscriber(self):
            rospy.Subscriber('/left_thyroid', Float64MultiArray, self.left_poseCallback)#/franka_ros_interface/custom_franka_state_controller/tip_state
            rospy.Subscriber('/right_thyroid', Float64MultiArray, self.right_poseCallback)
            rospy.Subscriber('/mid_thyroid', Float64MultiArray, self.mid_poseCallback)
    def left_poseCallback(self, msg):
        [T11, T12, T13, T14, T21, T22, T23, T24, T31, T32, T33, T34, T41, T42, T43, T44] = msg.data
        self.left_pose_TransMatrix = np.array([[T11, T12, T13, T14],
                                                                              [T21, T22, T23, T24],
                                                                              [T31, T32, T33, T34+0.1],
                                                                              [T41, T42, T43, T44]])
    def right_poseCallback(self, msg):
        [T11, T12, T13, T14, T21, T22, T23, T24, T31, T32, T33, T34, T41, T42, T43, T44] = msg.data
        self.right_pose_TransMatrix = np.array([[T11, T12, T13, T14],
                                                                              [T21, T22, T23, T24],
                                                                              [T31, T32, T33, T34+0.1],
                                                                              [T41, T42, T43, T44]])       
    def mid_poseCallback(self, msg):
        [T11, T12, T13, T14, T21, T22, T23, T24, T31, T32, T33, T34, T41, T42, T43, T44] = msg.data
        self.mid_pose_TransMatrix = np.array([[T11, T12, T13, T14],
                                                                              [T21, T22, T23, T24],
                                                                              [T31, T32, T33, T34+0.1],
                                                                              [T41, T42, T43, T44]])
        

    def thread_spin(self):
        rospy.spin()

    def get_current_post(self):
        #理论上的位置
        #返回的position带有时间戳等信息
        position = copy.deepcopy(self.franka.boject_pose)
        position.pose.position.x += self.franka.delta_x
        position.pose.position.y += self.franka.delta_y
        position.pose.position.z += self.franka.delta_z
        position.pose.orientation.x = self.franka.quater_x
        position.pose.orientation.y = self.franka.quater_y
        position.pose.orientation.z = self.franka.quater_z
        position.pose.orientation.w = self.franka.quater_w
        return position

    def encode_TransMatrix2string(self, TransMatrix):
        """numpy的4*4位姿矩阵转成空格的字符串"""
        vactor = TransMatrix.T.reshape(1, -1)
        string = ''
        for i in range(vactor.shape[1]):
            string = string + str(vactor[0, i]) + ' '
        string = string.strip()
        return string
    
    def get_actual_pose(self):
        #真实的位置
        actual_pose = copy.deepcopy(self.franka.current_pose)
        return actual_pose

    def get_T_base2end(self):
        x, y, z, quater_x, quater_y, quater_z, quater_w = self.get_actual_pose()
        R = self.franka.quat_to_pos_matrix_hm(quater_x, quater_y, quater_z, quater_w)
        T = np.zeros((4,4))
        T[0,0] = R[0,0]
        T[0,1] = R[0,1]
        T[0,2] = R[0,2]
        T[0,3] = x

        T[1,0] = R[1,0]
        T[1,1] = R[1,1]
        T[1,2] = R[1,2]
        T[1,3] = y

        T[2,0] = R[2,0]
        T[2,1] = R[2,1]
        T[2,2] = R[2,2]
        T[2,3] = z

        T[3,3] = 1
        return T

    def get_T_end2img(self, q_x=0, q_y=-0.019, q_z=0):
        T = np.zeros((4,4))
        T[0,0] = 1
        T[1,1] = 1
        T[2,2] = 1
        T[3,3] = 1
        T[0,3] = q_x
        T[1,3] = q_y
        T[2,3] = q_z
        return T

    def caculate_3D(self, x, y):
        P = np.array([0, x, y, 1]).reshape(4,-1)
        T_base2end = self.get_T_base2end()
        T_end2img = self.get_T_end2img()
        end_P = np.dot(T_end2img, P)
        base_P = np.dot(T_base2end, end_P)
        print(base_P)


    # def rotationAdjustmentNeeded(self, error_range=20):
    #     #判断接触是否良好，是否需要调整探头角度
    #     right_value = copy.deepcopy(self.right_value)
    #     mid_value = copy.deepcopy(self.mid_value)
    #     left_value = copy.deepcopy(self.left_value)

    #     #以中间的强度值为标准，在一定范围内，判断左右两侧的强度值是否处于允许范围内
    #     right_result = min(max(right_value, mid_value-error_range), mid_value+error_range) 
    #     left_result = min(max(left_value, mid_value-error_range), mid_value+error_range)

    #     if right_result==right_value and left_result==left_value:
    #         #若左右两侧均符合，则无需调整
    #         return 'No'
    #     elif right_result!=right_value and left_result==left_value:
    #         #若右侧不符合，左侧符合，则调整右侧
    #         return 'Left'
    #     elif right_result==right_value and left_result!=left_value:
    #         #若右侧符合，左侧不符合，则调整左侧
    #         return 'Right'
    #     else:
    #         #若双侧均不满足，考虑可能整体接触不好，平移探头
    #         return 'Down'
        
    def rotationAdjustmentNeeded(self, k=0.5):
        #判断接触是否良好，是否需要调整探头角度
        right_value = copy.deepcopy(self.right_value)
        left_value = copy.deepcopy(self.left_value)

        #根据左右某区域平均亮度调整
        # if right_value<th and left_value>th:
        #     return "Right"
        # elif right_value>th and left_value<th:
        #     return "Left"
        # else:
        #     return "No"
        
        #根据左右亮度值<5的个数判断
        print('图像左右两侧差异区域为: {}'.format(abs(right_value - left_value)))
        if abs(right_value - left_value)>2:
            diff = (right_value - left_value) *k
            if diff  > 0: #向右边旋转
                return "Right", diff
                # self.franka.rotate_y_by_quater(diff, motion_time=0.1, fix_frame=False,)
            if diff  < 0: #向左边旋转
                # self.franka.rotate_y_by_quater(diff, motion_time=0.1, fix_frame=False,)
                return "Left", diff
        else:
            return "No",None

        # if right_value<1 and left_value<1:
        #     return "No"
        # if right_value>1 and left_value>1:
        #     return "No"

        # #判断左右两侧的强度值是否处于允许范围内
        # diff_value = right_value - left_value
         
        # if np.abs(diff_value)>error_range:
        #     if diff_value>0:
        #         #若右侧过亮，则右侧不符合，左侧符合。则调整右侧
        #         return  'Right'
        #     else:
        #         #反之，调整左侧
        #         return  'Left'
        # else:
        #     #若左右两侧均符合，则无需调整
        #     return  'No'

    def auto_contact(self):
        rotation_num = 0
        adjustment, diff = self.rotationAdjustmentNeeded()
        print('接触检测 {}'.format(adjustment))
        if adjustment == 'Right' or adjustment=='Left':
            # print(adjustment, '进入阻抗控制，进行探头调整')
            # self.franka.update_param_safety(is_scaning=False, translational_stiffness=2000)  #进入 阻抗控制，更加准确
            while True:
                adjustment = self.rotationAdjustmentNeeded()

                print('接触检测 {}'.format(adjustment))
                if adjustment == 'Right':
                    self.franka.rotate_x_by_quater(diff, motion_time=0.2+self.baseline_time, fix_frame=False)

                elif adjustment == 'Left':
                    self.franka.rotate_x_by_quater(diff, motion_time=0.2+self.baseline_time, fix_frame=False)
                else:
                    break
                rotation_num = rotation_num + 1

                if rotation_num>5:
                    # 防止一次性调整太多。如果还是没接触好，下一轮继续调整
                    break
            # self.franka.update_param_safety(is_scaning=True, ext_force=self.goal_force)  #进入 力位控制，更加安全
            # print('恢复力位控制，共调整{}次探头'.format(rotation_num))
        else:
            print('无需调整探头')

    # def positionAdjustmentNeeded(self, pose, error_range=10):
    #     """expected_position """
    #     #判断目标是否位于图像中心
    #     target_center = copy.deepcopy(self.target_cent)
    #     img_center = copy.deepcopy(self.img_width)//2

    #     ##以图像中间为标准，在一定范围内，判断目标中心是否处于允许范围内
    #     center_result = min(max(target_center, img_center-error_range), img_center+error_range)
        
    #     if center_result==target_center:
    #         return 'center'
    #     if (target_center - img_center)<0:
    #         return 'Left'
    #     elif (target_center - img_center)>0:
    #         return 'Right'
    #     else:
    #         return 'center'
        
    def positionAdjustmentNeeded(self, pose, error_range=10):
        """expected_position """
        #判断目标是否位于图像中心
        thyroid_start_x=copy.deepcopy(self.thyroid_start_x)
        thyroid_w = copy.deepcopy(self.thyroid_w)
        qiguan_x= copy.deepcopy(self.qiguan_x)
        qiguan_w= copy.deepcopy(self.qiguan_w)
        img_ultrasound_start= copy.deepcopy(self.img_ultrasound_start )   #超声区域图像的起始
        img_ultrasound_end= copy.deepcopy(self.img_ultrasound_end)  #超声区域图像的终止
        img_center = (img_ultrasound_start+img_ultrasound_end)/2
        if self.area == 0:
            return 'center'
        else:
            ##以图像中间为标准，在一定范围内，判断目标中心是否处于允许范围内
            if pose =='left':
                if qiguan_x==0 and qiguan_w==0:
                    target_center=thyroid_start_x+thyroid_w/2
                else:
                    target_center=(thyroid_start_x+qiguan_x)/2

                center_result = min(max(target_center, img_center-error_range), img_center+error_range)
                if center_result==target_center:
                    return 'center'
                if (target_center - img_center)<0:
                    return 'Left'
                elif (target_center - img_center)>0:
                    return 'Right'
                else:
                    return 'center'

            elif pose =='right':
                if qiguan_x==0 and qiguan_w==0:
                    target_center=thyroid_start_x+thyroid_w/2
                else:
                    target_center=(thyroid_start_x+qiguan_x+thyroid_w+qiguan_w)/2
                center_result = min(max(target_center, img_center-error_range), img_center+error_range)
                if center_result==target_center:
                    return 'center'
                if (target_center - img_center)<0:
                    return 'Left'
                elif (target_center - img_center)>0:
                    return 'Right'
                else:
                    return 'center'
                
            elif pose =='mid':
                if qiguan_x==0 and qiguan_w==0:
                    target_center=img_center
                else:
                    target_center=qiguan_x+qiguan_w/2
                center_result = min(max(target_center, img_center-error_range), img_center+error_range)
                if center_result==target_center:
                    return 'center'
                if (target_center - img_center)<0:
                    return 'Left'
                elif (target_center - img_center)>0:
                    return 'Right'
                else:
                    return 'center'
                
            else:
                return 'center'
        
    # def positionAdjustmentNeeded_(self, expected_position, error_range=10, min_area=5000):
    #     """expected_position   字符串  mid ,left ,right"""
    #     #判断目标边缘是否位于预期边的旁边
    #     target_center = copy.deepcopy(self.target_cent)
    #     img_center = copy.deepcopy(self.img_width)//2
    #     target_right = copy.deepcopy(self.target_right)
    #     target_left = copy.deepcopy(self.target_left)
    #     img_left = copy.deepcopy(self.img_ultrasound_start)+error_range
    #     img_right = copy.deepcopy(self.img_ultrasound_end)-error_range



    #     if target_left>=img_left and target_right<=img_right:
    #         """两边都不接触，只需要判断是否在中间即可"""
    #         return self.positionAdjustmentNeeded()
    #     elif target_left < img_left and target_right> img_right:
    #         """两边都接触了, 需要将目标i扫描区域这边空出来"""
    #         pass
    #     elif target_left < img_left and 

    #     else:
    #         """其余情况(两边都不接触，)"""
    #         return self.positionAdjustmentNeeded()

    #     ##以图像中间为标准，在一定范围内，判断目标中心是否处于允许范围内
    #     center_result = min(max(target_center, img_center-error_range), img_center+error_range)
        
    #     if self.area<min_area or target_center==0 or center_result==target_center:
    #         return 'center'
    #     elif (target_center - img_center)<0:
    #         return 'Left'
    #     elif (target_center - img_center)>0:
    #         return 'Right'
    #     else:
    #         return ''


    def auto_center(self, pose):
        while True:
            adjustment = self.positionAdjustmentNeeded(pose)
            print('目标位于哪里 {}'.format(adjustment))
            if adjustment == 'Left':
                self.franka.translation_xyz(y=0.001, motion_time=0.1+self.baseline_time, fix_frame=False)
            elif adjustment == 'Right':
                self.franka.translation_xyz(y=-0.001, motion_time=0.1+self.baseline_time, fix_frame=False)
            else:
                break

    def record_pose_img(self):
        """
        记录对应的超声图像与探头姿态, 进行3D重建
        """
        frame = copy.deepcopy(self.unet_cv_img)
        actual_pose = copy.deepcopy(self.franka.current_pose)
        cv2.imwrite(os.path.join(self.save_path, '3D/img/{:0>4}_{}.png'.format(self.img_num, self.scanning_position)), frame)

        pose = self.encode_TransMatrix2string(actual_pose)
        self.pose.append({'img_num': self.img_num,
                                                'pose': pose})
        self.img_num += 1
    def save_pd(self):
        df = pd.DataFrame(self.pose)
        df.to_csv(os.path.join(self.save_path, '3D/pose.csv'), index=False, mode='w')

    def stop_scanning(self):
        """急停运行"""
        self.stop_scanning_flag = True
        self.franka.stop_moving = True

    def recover_stop_state(self):
        """重置急停状态"""
        self.stop_scanning_flag = False
        self.franka.stop_moving = False  

    def is_fan_scanning(self, threshold1 = 5):
        # 获取最新的20个元素
        #最新20个点与最老20个点均值对比，若超过某一阈值，则开始扇扫，并停止计算均值，直到扫另一侧的时候再开始计算
        force_queue_x = copy.deepcopy(self.force_queue_x)
        # force_queue_My = copy.deepcopy(self.force_queue_My)
        if self.Fx_init_mean is None:
            # 每次扫查完一侧，就会重置None
            self.Fx_init_mean = np.mean(list(force_queue_x)[-20:])

        Fx_new = np.mean(list(force_queue_x)[-10:])

        print('----- 当前Z轴压力为：{}N，  设定压力阈值为 ：{} N-------'.format(abs(Fx_new - self.Fx_init_mean), threshold1))

        if abs(Fx_new - self.Fx_init_mean) > threshold1 :
            # and abs(avg_My_new-avg_My_old) > threshold2
            print(f"到达锁骨,{self.Fx_init_mean},{Fx_new}")
            return True
        else:
            return False

    def longitudinal_section_scanning(self, contact=True):
        """
        全程  力位混合控制
        contact       是否进行探头接触检测
        """
        print('----- 开始纵切面扫查 ----')
        self.check_point.write_check_point('{}_scanning_纵扫开始'.format(self.scanning_position))
        step_num = 15  #纵切扫查距离
        for go_forward in range(step_num):
            print('======================')
            if contact:
                self.auto_contact()  #自动调整接触面

            print('-------- 纵切面-前进 {} --------'.format(go_forward+1))
            self.franka.translation_xyz(x=0.002, motion_time=0.1+self.baseline_time, fix_frame=False)     #如果已经检测到目标，移动间距短一些
        self.check_point.write_check_point('{}_scanning_纵扫结束'.format(self.scanning_position))

    def auto_scanning(self, centering=True, contact=True, pose='', fan_scanning=False, longitudinal=False):
        """
        centering   是否目标居中
        contact       是否进行探头接触检测
        pose             当前扫查位置, left/right/mid
        fan_scanning    末端是否进行扇扫
        longitudinal      横切扫查之后，是否进行纵扫
        """
        print('开始扫查')
        area_thyroid = 3000
        rotate_angle = 0
        epoch = 0
        start_recording = False
        # self.area_dict = {}
        self.check_point.write_check_point('{}_scanning_下降'.format(pose))
        self.franka.translation_xyz(z=-0.08, motion_time=2+self.baseline_time, fix_frame=True)
        self.franka.update_param_safety(is_scaning=True, ext_force=1.0)   # 先用一个小的力去接触皮肤，进入力位混合控制

        # 判断是否处于甲状腺，如果有甲状腺，则往上移动直至没有
        print('甲状腺面积：{}'.format(self.area))
        if self.area>area_thyroid:
            print('寻找甲状腺上级')
            max_excude_distance = 0.02  #单位m, 最大往上移动距离
            current_pose = copy.deepcopy(self.franka.current_pose)
            current_pose = np.dot(current_pose, self.franka.T_end2probe)  #探头末端坐标系
            point1 =   current_pose[:2, 3]
            self.check_point.write_check_point('{}_scanning_往上级移动'.format(pose))
            for i in range(20):
                print('{}_scanning_往上级移动: {}'.format(pose, i))
                self.franka.translation_xyz(x=-0.004, motion_time=0.2+self.baseline_time, fix_frame=False)    #如果还没检测到目标，移动间距长一些
                current_pose = copy.deepcopy(self.franka.current_pose)
                current_pose = np.dot(current_pose, self.franka.T_end2probe)  #探头末端坐标系
                point2 =   current_pose[:2, 3]  
                distance = self.calculate_distance(point1, point2)
                if distance > max_excude_distance:
                    break
                if self.area<area_thyroid:
                    break

        self.franka.update_param_safety(is_scaning=True, ext_force=self.goal_force)   # 恢复目标的扫查力

        # 开始从上级，往下扫查
        max_excude_distance = 0.05  #单位m, 最大扫查距离
        current_pose = copy.deepcopy(self.franka.current_pose)
        current_pose = np.dot(current_pose, self.franka.T_end2probe)  #探头末端坐标系
        point1 =   current_pose[:2, 3]  

        area_count = 0
        self.check_point.write_check_point('{}_scanning_横扫开始'.format(pose))
        while not self.stop_scanning_flag:
            print('======================')
            epoch += 1
            if contact:
                self.auto_contact()  #自动调整接触面
            if centering and self.area>area_thyroid:
                self.auto_center(pose)   #目标中心化

            if self.area > area_thyroid:
                start_recording = True
            if start_recording:
                # self.area_dict[epoch] = [self.area, self.franka.current_pose]
                print('小步前进')
                self.franka.translation_xyz(x=0.002, motion_time=0.1+self.baseline_time, fix_frame=False)     #如果已经检测到目标，移动间距短一些
                if fan_scanning and rotate_angle<20:
                    self.franka.rotate_y_by_quater(1, motion_time=0.1+self.baseline_time, fix_frame=False,)             
                    rotate_angle += 1
                    print("扫查中旋转1度, 目前总累计旋转{}度".format(rotate_angle))
            else:
                print('大步前进')
                self.franka.translation_xyz(x=0.004, motion_time=0.2+self.baseline_time, fix_frame=False)    #如果还没检测到目标，移动间距长一些

            # time.sleep(0.1)
            # self.record_pose_img()
            # self.save_pd()               
            current_pose = copy.deepcopy(self.franka.current_pose)
            current_pose = np.dot(current_pose, self.franka.T_end2probe)  #探头末端坐标系
            point2 =   current_pose[:2, 3]  
            distance = self.calculate_distance(point1, point2)

            # 共同停止条件
            if self.area < area_thyroid and start_recording:
                area_count = area_count+1
                print('------- 无甲状腺：{}/3 ------ '.format(area_count))
                if area_count>=3:   #连续3帧检测不到了，才结束
                    self.client_cmd.post(data='连续3帧无甲状腺，结束扫查')
                    print('------ 连续3帧无甲状腺，停止！！! -------')
                    break
            else:
                area_count = 0
            
            if pose =='right' or  pose =='left':
                print('----- 腺体执行距离{:.5f} cm'.format(distance*100))
                suogu = self.is_fan_scanning(threshold1=5.5)
                if suogu:
                    self.client_cmd.post(data='已到达锁骨，结束扫查')
                    print('--  到达锁骨，停止！！！---')
                    break
                if distance > (max_excude_distance+0.02):
                    self.client_cmd.post(data=' 已扫查{:.1f} 厘米, 超过最大预设距离{:.1f}厘米， 结束'.format(distance*100, (max_excude_distance+0.02)*100))
                    print(' ----- 执行距离{:.5f} cm, 超过最大预设距离{:.5f}, 停止!!! -------'.format(distance*100, (max_excude_distance+0.02)*100))
                    break
            elif pose =='mid':
                print('峡部执行距离{:.5f} cm'.format(round(distance, 5)))
                if distance > max_excude_distance:
                    print('--  目前距离{:.5f} cm，超过阈值{:.5f}cm，停止！！！---'.format(distance*100, max_excude_distance))
                    break
        
        # 纵扫的时候，也是先脱离，再变换，再下来扫查
        # 脱离
        # self.check_point.write_check_point('{}_scanning_纵扫-上升'.format(pose))
        self.check_point.write_check_point('{}_scanning_横扫结束-上升'.format(pose))
        self.franka.update_param_safety(is_scaning=False)
        self.franka.translation_xyz(z=-0.05, fix_frame=False, motion_time=3+self.baseline_time)  #先脱离体表，防止移动过程中出现意外
        self.franka.translation_xyz(z=0.30, fix_frame=True, motion_time=5+self.baseline_time)  #先脱离体表，防止移动过程中出现意外

        # # 纵扫
        # # --- 返回中点
        # if pose!='mid':
        #     print(' -----  开始纵切扫查 -----')
        #     # need_back_up = int(epoch/2)     
        #     for i in range(epoch):
        #         print('往上移动: {}/{}'.format(i+1, epoch))
        #         self.franka.rotate_y_by_quater(-1, motion_time=0.2, fix_frame=False)   
        #         self.franka.translation_xyz(x=-0.003, motion_time=0.1, fix_frame=False)    
        #     # --- 纵扫
        #     self.switch_planes(self.scanning_position)
                
        # # 脱离
        # self.check_point.write_check_point('{}_scanning_上升'.format(pose))
        # self.franka.update_param_safety(is_scaning=False)
        # self.franka.translation_xyz(z=0.35, fix_frame=True)  #先脱离体表，防止移动过程中出现意外

        # if pose == 'right':
        #     self.franka.rotate_z_by_quater(-90, fix_frame=False, motion_time=6)   # 以预估的点绕z转-90
        # else:
        #     self.franka.rotate_z_by_quater(90, fix_frame=False, motion_time=6)   # 以预估的点绕z转90
        


    def switch_planes(self, pose):
        """
        从横切面转换到指定的切面。
        """
        # self.check_point.write_check_point('{}_纵扫-旋转'.format(self.scanning_position))
        # if pose == 'right':
        #     self.franka.rotate_z_by_quater(90, fix_frame=False, motion_time=6)   # 以预估的点绕z转90
        # else:
        #     self.franka.rotate_z_by_quater(-90, fix_frame=False, motion_time=6)   # 以预估的点绕z转-90

        # self.check_point.write_check_point('{}_纵扫'.format(self.scanning_position))
        # self.longitudinal_section_scanning(contact=True)    # 纵切扫查

        self.check_point.write_check_point('{}_纵切'.format(self.scanning_position))
        if pose == 'right':
            self.franka.rotate_z_by_quater(90, fix_frame=False, motion_time=6+self.baseline_time)   # 以预估的点绕z转90
            R_tran = self.franka.operator_translation(x=0, y=-0.002)   
            R_rotate = self.franka.operator_rotate_z(90)   # 以预估的点绕z转
            left_end = np.dot(self.left_pose_TransMatrix, R_tran)  #右乘，以自身坐标系为参考（参考系会变）
            left_end = np.dot(left_end, R_rotate)
            # self.franka.translation_xyz(y=-0.05)
            self.franka.go_pose(left_end, is_robot_end=False, motion_time=8+self.baseline_time)  #TODO
        elif pose == 'left':
            self.franka.rotate_z_by_quater(-90, fix_frame=False, motion_time=6+self.baseline_time)   # 以预估的点绕z转90
            R_tran = self.franka.operator_translation(x=0, y=0.002)   
            R_rotate = self.franka.operator_rotate_z(-90)   # 以预估的点绕z转
            right_end = np.dot(self.right_pose_TransMatrix, R_tran)  #右乘，以自身坐标系为参考（参考系会变）
            right_end = np.dot(right_end, R_rotate)
            self.franka.go_pose(right_end, is_robot_end=False, motion_time=8+self.baseline_time)  #TODO
        else:
            print(' ------- 峡部无需纵扫 ------- ')
        
        self.franka.translation_xyz(z=-0.08, motion_time=2+self.baseline_time, fix_frame=True)
        self.franka.update_param_safety(is_scaning=True, ext_force=self.goal_force)
        self.longitudinal_section_scanning(contact=True)    # 纵切扫查
        # 脱离
        self.check_point.write_check_point('{}_scanning_上升'.format(pose))
        self.franka.update_param_safety(is_scaning=False)
        self.franka.translation_xyz(z=0.35, fix_frame=True)  #先脱离体表，防止移动过程中出现意外

        if pose == 'right':
            self.franka.rotate_z_by_quater(-90, fix_frame=False, motion_time=6+self.baseline_time)   # 以预估的点绕z转-90
        elif pose == 'left':
            self.franka.rotate_z_by_quater(90, fix_frame=False, motion_time=6+self.baseline_time)   # 以预估的点绕z转90
        else:
            print(' ------- 峡部无需纵扫 ------- ')


    def calculate_distance(self, point1, point2):
        diff = point2 - point1
        distance = np.sqrt(np.sum(diff**2))
        return distance
    
    #  ===== agent 的话，相当于将这个 start_scanning 的标准流程化的扫查过程进行解耦合，分解成多个小的执行动作，让agent进行组合调用
    def start_recording(self):
        print('开始扫查')
        # self.client_cmd.post_message(data='准备开始扫查啦，请放松，有不舒服的请告诉我。下面我将开始采集相关数据并进行存储和分析')
        self.recording_state.start_recording()  # 开始数据记录
        self.check_point.write_check_point('Start')
        self.client_cmd.post(data='sucess')

    def go_home(self):
        # self.client_cmd.post_message(data='下面，我将回到一开始的起始位置')
        print('下面，我将回到一开始的起始位置')
        self.franka.recovery_error()
        time.sleep(1)
        self.franka.go_home_by_joint_pose()
        print('回到初始位置执行完毕')
        self.client_cmd.post(data='sucess')

    def force_up(self):
        if self.goal_force < 10:
            self.goal_force = self.goal_force + 1
            # self.franka.update_param_safety(is_scaning=True, ext_force=self.goal_force)
            self.client_cmd.post(data='sucess')
        else:
            self.client_cmd.post(data='error')

    def force_down(self):
        if self.goal_force > 1:
            self.goal_force = self.goal_force - 1
            # self.franka.update_param_safety(is_scaning=True, ext_force=self.goal_force)
            self.client_cmd.post(data='sucess')
        else:
            self.client_cmd.post(data='error')

    def speed_up(self):
        "change the speed by time"
        try:
            if self.baseline_time > 0.1:
                self.baseline_time = self.baseline_time - 0.1
                self.client_cmd.post(data='sucess')
            else: 
                self.client_cmd.post(data='sucess')
        except:
            self.client_cmd.post(data='error')


    def speed_down(self):
        "change the speed by time"
        try:
            self.baseline_time = self.baseline_time + 0.1
            self.client_cmd.post(data='success')
        except:
            self.client_cmd.post(data='error')

    def change_control_mode(self, mode):
        try:
            self.franka.recovery_error()
            if mode == 'impedance_control':
                self.franka.update_param_safety(is_scaning=False)
                self.client_cmd.post(data='sucess')
            elif mode == 'admittance_control':
                self.franka.switch_exclusively_to(self.franka.cartesian_velocity_Z_impedance_controller_name)
                self.client_cmd.post(data='sucess')
            elif mode == 'hybrid_controller':
                self.franka.update_param_safety(is_scaning=True, ext_force=self.goal_force)
                self.client_cmd.post(data='sucess')
            elif mode == 'cartesian_control':
                self.franka.switch_exclusively_to(self.franka.cartesian_position_controller_name)
                self.client_cmd.post(data='sucess')
            self.franka.recovery_error()
        except:
            self.client_cmd.post(data='error')



    def scanning_left(self):
        try:
            self.franka.recovery_error()
            # self.client_cmd.post_message(data='下面，我将准备扫查您左侧的甲状腺。我会先进行横扫，当发现可疑结节，再进一步进行纵向扫查。请保持放松，不用紧张，如果有什么不舒服的请及时告诉我')
            self.franka.go_home()
            self.franka.translation_xyz(x=0.05)
            R_trans = self.franka.operator_translation(x=-0.02)    # 现在的关键点检测模型检测出的6个点的初始位置稍微偏下，用这个补偿一下
            R_rotate = self.franka.operator_rotate_y(10)   # 以预估的点绕y转35

            # 初始化探头，稍微朝锁骨方向旋转
            left_end = np.dot(self.left_pose_TransMatrix, R_trans)  #右乘，以自身坐标系为参考（参考系会变）
            left_end = np.dot(left_end, R_rotate)  #右乘，以自身坐标系为参考（参考系会变）
            self.scanning_position = 'right'   # 实际是患者的左侧
            self.check_point.write_check_point('{}_移动至初始点上方'.format(self.scanning_position))
            self.franka.go_pose(left_end, is_robot_end=False, motion_time=10+self.baseline_time)  #TODO
            self.save_pose_image.scanning_position = self.scanning_position
            self.save_pose_image.is_pause = False  #恢复记录
            self.Fx_init_mean = None
            self.auto_scanning(centering=True, contact=True, pose=self.scanning_position, fan_scanning=True)
            # 左边变右边，先进行一个变换，防止路径规划过大，导致抖动
            self.franka.rotate_x_by_quater(20, fix_frame=False, motion_time=4.0+self.baseline_time)
            self.save_pose_image.is_pause = True  #暂停记录
            self.client_cmd.post(data='success')
        except:
            self.client_cmd.post(data='error')


    def scanning_left_longitudinal(self):
        try:
            self.franka.recovery_error()
            # self.client_cmd.post_message(data='检测到您左侧甲状腺有可疑结节，下面，我将准备对您的左侧甲状腺进行纵扫，请放松，保持不动')
            # 左边变右边，先进行一个变换，防止路径规划过大，导致抖动
            self.franka.rotate_x_by_quater(-20, fix_frame=False, motion_time=4.0+self.baseline_time)
            self.scanning_position = 'right'   # 实际是患者的左侧
            self.switch_planes(pose=self.scanning_position)
            self.franka.translation_xyz(x=0.05)
            # 左边变右边，先进行一个变换，防止路径规划过大，导致抖动
            self.franka.rotate_x_by_quater(20, fix_frame=False, motion_time=4.0+self.baseline_time)
            self.client_cmd.post(data='sucess')
        except:
            self.client_cmd.post(data='error')


    def left_2_right(self):
        try:
            self.franka.recovery_error()
            # self.client_cmd.post_message(data='已完成左侧甲状腺的扫查，接下来准备对您的右侧甲状腺进行扫查，请保持不动，谢谢')
            # 左边变右边，先进行一个变换，防止路径规划过大，导致抖动
            self.franka.rotate_x_by_quater(20, fix_frame=False, motion_time=4.0+self.baseline_time)
            self.client_cmd.post(data='sucess')
        except:
            self.client_cmd.post(data='error')


    def scanning_right(self):
        try:
            self.franka.recovery_error()
            # self.client_cmd.post_message(data='下面，我将准备扫查您右侧的甲状腺。我会先进行横扫，当发现可疑结节，再进一步进行纵向扫查。请保持放松，不用紧张，如果有什么不舒服的请及时告诉我')
            self.franka.go_home()
            self.franka.translation_xyz(x=0.05)
            R_trans = self.franka.operator_translation(x=-0.02)    # 现在的关键点检测模型检测出的6个点的初始位置稍微偏下，用这个补偿一下
            R_rotate = self.franka.operator_rotate_y(10)   # 以预估的点绕y转35

            # 初始化探头，稍微朝锁骨方向旋转
            right_end = np.dot(self.right_pose_TransMatrix, R_trans)  #右乘，以自身坐标系为参考（参考系会变）
            right_end = np.dot(right_end, R_rotate)  #右乘，以自身坐标系为参考（参考系会变）
            self.scanning_position = 'left'
            self.check_point.write_check_point('{}_移动至初始点上方'.format(self.scanning_position))
            self.franka.go_pose(right_end, is_robot_end=False, motion_time=10+self.baseline_time)
            self.save_pose_image.scanning_position = self.scanning_position
            self.save_pose_image.is_pause = False  #恢复记录
            self.Fx_init_mean = None
            self.auto_scanning(centering=True, contact=True, pose=self.scanning_position, fan_scanning=True)
            self.save_pose_image.is_pause = True  #暂停记录
            self.client_cmd.post(data='sucess')
        except:
            self.client_cmd.post(data='error')


    def scanning_right_longitudinal(self):
        try:
            self.franka.recovery_error()
            # self.client_cmd.post_message(data='检测到您右侧甲状腺有可疑结节，下面，我将准备对您的右侧甲状腺进行纵扫，请放松，保持不动')
            self.scanning_position = 'left'   # 实际是患者的左侧
            self.switch_planes(pose=self.scanning_position)
            self.franka.translation_xyz(x=-0.15)
            self.client_cmd.post(data='sucess')
        except:
            self.client_cmd.post(data='error')


    def scanning_mid(self):
        self.franka.recovery_error()
        # self.client_cmd.post_message(data='已完成双侧甲状腺的扫查，接下来我会移动到您的气管前面，扫查甲状腺的峡部。请保持不动，谢谢')
        self.franka.go_home()
        self.scanning_position = 'mid'
        self.franka.go_pose(self.mid_pose_TransMatrix, is_robot_end=False, motion_time=10+self.baseline_time)
        self.save_pose_image.scanning_position = self.scanning_position
        self.save_pose_image.is_pause = False  #恢复记录
        self.auto_scanning(centering=True, contact=False, pose=self.scanning_position, fan_scanning=False)
        self.save_pose_image.start_save = False  #停止记录, 并保存
        print('已完成峡部')
        self.client_cmd.post(data='sucess')

    def stop_recording(self):
        # self.client_cmd.post(data='已完成您的甲状腺的扫查，非常感谢您的配合。接下来我会归位到初始位置，并对扫查的结果进行分析。')
        self.recording_state.stop_recording()
        self.client_cmd.post(data='sucess')

    def start_for_scanning(self):
        print('下面，我将回到一开始的起始位置')
        self.franka.recovery_error()
        time.sleep(1)
        self.franka.go_home_by_joint_pose()
        self.recording_state.start_recording()  # 开始数据记录
        self.check_point.write_check_point('Start')
        print('回到初始位置执行完毕')
        self.client_cmd.post(data='sucess')

    def stop_for_scanning(self):
        self.franka.recovery_error()
        # self.client_cmd.post_message(data='已完成您的甲状腺的扫查，接下来我会关闭数据记录，并回到初始位置，非常感谢您的配合。')
        self.recording_state.stop_recording()
        self.franka.go_home_by_joint_pose()
        # self.client_cmd.post(data='命令执行完毕')
        self.client_cmd.post(data='sucess')



    def start_scanning(self):
        print('开始扫查')
        # self.recording_state.start_recording()  # 开始数据记录
        # self.check_point.write_check_point('Start')
        
        # self.franka.translation_xyz(x=0.05)
        # R_trans = self.franka.operator_translation(x=-0.02)    # 现在的关键点检测模型检测出的6个点的初始位置稍微偏下，用这个补偿一下
        # R_rotate = self.franka.operator_rotate_y(10)   # 以预估的点绕y转35

        # # 初始化探头，稍微朝锁骨方向旋转
        # left_end = np.dot(self.left_pose_TransMatrix, R_trans)  #右乘，以自身坐标系为参考（参考系会变）
        # left_end = np.dot(left_end, R_rotate)  #右乘，以自身坐标系为参考（参考系会变）
        # self.scanning_position = 'right'   # 实际是患者的左侧
        # self.check_point.write_check_point('{}_移动至初始点上方'.format(self.scanning_position))
        # self.franka.go_pose(left_end, is_robot_end=False, motion_time=10)  #TODO
        # self.save_pose_image.scanning_position = self.scanning_position
        # self.save_pose_image.is_pause = False  #恢复记录
        # self.Fx_init_mean = None
        # self.auto_scanning(centering=True, contact=True, pose=self.scanning_position, fan_scanning=True)
        # self.save_pose_image.is_pause = True  #暂停记录
        # 纵扫
        # self.switch_planes(pose=self.scanning_position)
        # self.franka.translation_xyz(x=0.05)
        # print('已完成左叶，准备扫查右叶')

        # # 左边变右边，先进行一个变换，防止路径规划过大，导致抖动
        # self.franka.rotate_x_by_quater(20, fix_frame=False, motion_time=4.0)

        # 初始化探头，稍微朝锁骨方向旋转
        # right_end = np.dot(self.right_pose_TransMatrix, R_trans)  #右乘，以自身坐标系为参考（参考系会变）
        # right_end = np.dot(right_end, R_rotate)  #右乘，以自身坐标系为参考（参考系会变）
        # self.scanning_position = 'left'
        # self.check_point.write_check_point('{}_移动至初始点上方'.format(self.scanning_position))
        # self.franka.go_pose(right_end, is_robot_end=False, motion_time=10)
        # self.save_pose_image.scanning_position = self.scanning_position
        # self.save_pose_image.is_pause = False  #恢复记录
        # self.Fx_init_mean = None
        # self.auto_scanning(centering=True, contact=True, pose=self.scanning_position, fan_scanning=True)
        # self.save_pose_image.is_pause = True  #暂停记录
        # # 纵扫
        # self.switch_planes(pose=self.scanning_position)
        # self.franka.translation_xyz(x=-0.15)
        # print('已完成右叶，准备扫查峡部')

        # self.scanning_position = 'mid'
        # self.franka.go_pose(self.mid_pose_TransMatrix, is_robot_end=False, motion_time=10)
        # self.save_pose_image.scanning_position = self.scanning_position
        # self.save_pose_image.is_pause = False  #恢复记录
        # self.auto_scanning(centering=True, contact=False, pose=self.scanning_position, fan_scanning=False)
        # self.save_pose_image.start_save = False  #停止记录, 并保存
        # print('已完成峡部')
        # self.recording_state.stop_recording()

        #self.stop_cap_video()  # 结束录制视频
        # self.save_pd()

    def end_scanning(self):
        #结束任务
        self.franka.translation_xyz(z=0.1, motion_time=5+self.baseline_time, fix_frame=True)
        self.franka.go_home()
        self.franka.close()
        print('扫查结束')


    # def start_cap_video(self):
    #     self.cap_all_us = CaptureVideo_Thread('/Unet/capture_us', output_file=os.path.join(self.save_path, 'capture_all_US.avi'), 
    #                                                                         frame_width=900, frame_height=900)
    #     self.cap_us = CaptureVideo_Thread('/Unet/original', output_file=os.path.join(self.save_path, 'video_US.avi'), 
    #                                                                         frame_width=900, frame_height=900)
    #     self.cap_pred = CaptureVideo_Thread('/Unet/pred', output_file=os.path.join(self.save_path, 'video_pred.avi'), 
    #                                                                         frame_width=900, frame_height=900)
    #     self.cap_all_us.start()
    #     self.cap_us.start()
    #     self.cap_pred.start()

    # def stop_cap_video(self):
    #     self.cap_all_us.is_fineshed=True
    #     self.cap_us.is_fineshed=True
    #     self.cap_pred.is_fineshed=True
    #     self.cap_all_us.out.release()
    #     print('all us release')
    #     self.cap_us.out.release()
    #     print('us release')
    #     self.cap_pred.out.release()
    #     print('pred release')

    def for_testing(self):
        end_pose = np.array([
            [0.7258128697848866, -0.6820927750317126, 0.0891354267626769, 0.0], 
            [-0.6855018314977035, -0.7279858056025922, 0.011131300662492778, 0.0], 
            [0.05729674570096616, -0.06918173957542673, -0.9959573132626702, 0.0],
            [0.6172674625387101, 0.05494234992879699, 0.23087044298768433, 1.0]
            ]).T
        self.start_cap_video()  #开始录制视频
        self.record_pose_img()
        self.franka.go_pose(end_pose, motion_time=10+self.baseline_time, delta_time=0.001, is_robot_end=True)
        self.record_pose_img()
        # franka.update_param_safety(is_scaning=True,ext_force=1)
        self.franka.translation_xyz(x=-0.08, motion_time=5+self.baseline_time)
        self.record_pose_img()
        self.franka.translation_xyz(z=0.1, motion_time=5+self.baseline_time)
        self.record_pose_img()
        # franka.update_param_safety(is_scaning=False)
        self.franka.go_home(motion_time=10+self.baseline_time, delta_time=0.0005)
        self.record_pose_img()
        self.stop_cap_video()  # 结束录制视频
        


if  __name__ == "__main__":
    goal_force = 2  #压力，单位N
    US_servo = UltrasoundServo(goal_force)

    # US_servo.franka.go_home()
    # US_servo.auto_center()

    US_servo.start_scanning()  #开始扫查
    US_servo.end_scanning()
    