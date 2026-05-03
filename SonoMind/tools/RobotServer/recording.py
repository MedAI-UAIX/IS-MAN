import warnings
warnings.filterwarnings("ignore")

import sys
from config import rocord_save_path
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
from franka_example_controllers.msg import EndForceTorque
from robotiq_ft_sensor.msg import ft_sensor  
from geometry_msgs.msg import WrenchStamped
import threading
import time
import cv2
import numpy as np
import os
import fitz
from glob import glob
import datetime
from std_msgs.msg import Float64, Float64MultiArray
import traceback
# from yolo import YOLO
# from hrnet import Hrnet
from PIL import Image as PILImage
from franka_msgs.msg import FrankaState
import threading
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
# 导入你的自定义消息类型
import csv
from Threading_report_image import TI_ARDSThread
import pandas as pd
import json

class LoadThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.bridge = CvBridge()
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_seg_original = None
        self.latest_seg_pred = None
        self.latest_force_data = None
        self.latest_state_data = None
        self.init_video_node()

        # # 从JSON文件中读取参数信息
        with open('/home/usai/auto_RUSS/R_16_ForceCalibration/parameters.json', 'r') as json_file:
            loaded_parameters = json.load(json_file)

        # 打印读取的参数信息
        Lx = loaded_parameters['calibration']['Lx']
        Ly = loaded_parameters['calibration']['Ly']
        Lz = loaded_parameters['calibration']['Lz']
        Fx0 = loaded_parameters['calibration']['Fx0']
        Fy0 = loaded_parameters['calibration']['Fy0']
        Fz0 = loaded_parameters['calibration']['Fz0']
        self.x_offset = loaded_parameters['offset']['x']
        self.y_offset = loaded_parameters['offset']['y']
        self.z_offset = loaded_parameters['offset']['z']
        k1 = loaded_parameters['constants']['k1']
        k2 = loaded_parameters['constants']['k2']
        k3 = loaded_parameters['constants']['k3']
        self.G = np.array([[Lx, Ly, Lz]]).T
        self.F0 = np.array([[Fx0, Fy0, Fz0]]).T
        Mx0 = k1 - Fy0*self.z_offset + Fz0*self.y_offset
        My0 = k2 - Fz0*self.x_offset + Fx0*self.z_offset
        Mz0 = k3 - Fx0*self.y_offset + Fy0*self.x_offset
        self.M0 = np.array([[Mx0, My0, Mz0]]).T

        self.force_init_x = 0.0 
        self.force_init_y = 0.0
        self.force_init_z = 0.0
        self.force_init_mx =  0.0
        self.force_init_my =  0.0
        self.force_init_mz =  0.0
        self.update_num = 0
      

    def run(self):
        rospy.spin()

    def init_video_node(self):
   
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)

        rospy.Subscriber('/Unet/capture_us', Image, self.seg_original)
        rospy.Subscriber('/Unet/pred', Image, self.seg_pred)
        
        rospy.Subscriber('/robotiq_ft_sensor', ft_sensor, self.force_sensor_callback)
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.franka_state_callback)

    def get_Fe(self, fm, R):
        '''
        fm : 传感器测得实时力 [x,y,z]
        R: 末端位姿矩阵 3 x 3

        校准数值：
            G:自重 [x,y,z]
            F0:初始值 [x,y,z]
        '''
        # print(R.T)
        # print(self.G)
        return fm - self.F0 - R.T@self.G
    
    def get_Me(self, M, R):
        '''
        M : 传感器测得实时力 [Mx,My,Mz]
        R: 末端位姿矩阵 3 x 3
        '''

        Gx, Gy, Gz = R.T@self.G
        Mgx = Gz[0]*self.y_offset - Gy[0]*self.z_offset
        Mgy = Gx[0]*self.z_offset - Gz[0]*self.x_offset
        Mgz = Gy[0]*self.x_offset - Gx[0]*self.y_offset
        self.Mg = np.array([[Mgx, Mgy, Mgz]]).T
        
        return M - self.M0 - self.Mg
    
    def rgb_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        secs, nsecs = self.get_ros_time()
        self.latest_rgb_image = {
                                                            'secs': secs,
                                                            'nsecs': nsecs,
                                                            'data': cv_image
                                                            }

    def depth_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        secs, nsecs = self.get_ros_time()
        self.latest_depth_image = {
                                                            'secs': secs,
                                                            'nsecs': nsecs,
                                                            'data': cv_image
                                                            }
        
    def seg_original(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        secs, nsecs = self.get_ros_time()
        self.latest_seg_original = {
                                                            'secs': secs,
                                                            'nsecs': nsecs,
                                                            'data': frame
                                                            }

    def seg_pred(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")  #8UC1
        secs, nsecs = self.get_ros_time()
        self.latest_seg_pred = {
                                                        'secs': secs,
                                                        'nsecs': nsecs,
                                                        'data': frame
                                                        }

    def force_sensor_callback(self, msg):
        if self.update_num %  300 == 0:
            try:
                df = pd.read_csv('/home/usai/auto_RUSS/R_UI/force/force_zero.csv')
                self.force_init_x = df['x'].values[0]
                self.force_init_y = df['y'].values[0]
                self.force_init_z = df['z'].values[0]
                self.force_init_mx = df['mx'].values[0]
                self.force_init_my = df['my'].values[0]
                self.force_init_mz = df['mz'].values[0]
            except:
                pass

        # fx, fy, fz = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        # mx, my, mz = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        fx = msg.Fx 
        fy = msg.Fy 
        fz = msg.Fz
        mx = msg.Mx
        my = msg.My
        mz = msg.Mz

        # --- 压力校准
        fm = np.array([[fx, fy, fz]]).T
        M = np.array([[mx, my, mz]]).T
        R = self.T_base2wrist[:3, :3]
        Fx, Fy, Fz = self.get_Fe(fm, R)
        Mx, My, Mz = self.get_Me(M, R)

        fx = Fx[0] - self.force_init_x
        fy = Fy[0] - self.force_init_y
        fz = -Fz[0] - self.force_init_z
        mx = Mx[0] - self.force_init_mx
        my = My[0] - self.force_init_my
        mz = Mz[0] - self.force_init_mz

        secs, nsecs = self.get_ros_time()
        self.latest_force_data = {
                                                            'secs': secs,
                                                            'nsecs': nsecs,
                                                            'data': '{},{},{},{},{},{}'.format(fx, fy, fz, mx, my, mz)
                                                            }
        self.update_num = self.update_num  + 1

    def franka_state_callback(self, msg):
        self.T_base2wrist = np.array(msg.O_T_EE).reshape((4, 4)).T
        secs, nsecs = self.get_ros_time()
        self.latest_state_data = {
                                                            'secs': secs,
                                                            'nsecs': nsecs,
                                                            'data': self.encode_TransMatrix2string(self.T_base2wrist)
                                                            }
        
    def encode_TransMatrix2string(self, TransMatrix):
        """numpy的4*4位姿矩阵转成空格的字符串"""
        vactor = TransMatrix.T.reshape(1, -1)
        string = ''
        for i in range(vactor.shape[1]):
            string = string + str(vactor[0, i]) + ' '
        string = string.strip()
        return string

    # 获取方法
    def get_latest_rgb_image(self):
        return self.latest_rgb_image

    def get_latest_depth_image(self):
        return self.latest_depth_image
    
    def get_latest_seg_original(self):
        return self.latest_seg_original

    def get_latest_seg_pred(self):
        return self.latest_seg_pred

    def get_latest_force_data(self):
        return self.latest_force_data

    def get_latest_state_data(self):
        return self.latest_state_data
    
    def get_ros_time(self):
        # 获取当前的ROS时间
        current_time = rospy.get_rostime()
        return current_time.secs, current_time.nsecs
        # print("Current ROS time:")
        # print("Seconds: ", current_time.secs)
        # print("Nanoseconds: ", current_time.nsecs)



from std_msgs.msg import Bool
class RecordingController:
    def __init__(self, save_path=rocord_save_path):
        rospy.init_node('Recording_control', anonymous=True)
        self.save_path = save_path
        self.load_thread = LoadThread()
        self.listener_thread()
        self.create_new_path  = True
        self.RecodingState = False
        self.rate = rospy.Rate(15)
        self.start_analysis = False

    # =========== 话题订阅 ============
    def listener_thread(self):
        #订阅话题
        self.Subscriber_RecodingState()

        # 开启多线程
        t1 = threading.Thread(target=self.thread_spin)  # 末端位置订阅线程
        t1.start()

    def thread_spin(self):
        rospy.spin()

    def Subscriber_RecodingState(self):
        rospy.Subscriber('/start_recording', Bool, self.handle_recording_signal)
    def handle_recording_signal(self, msg):
        self.RecodingState = msg.data
        # print(self.RecodingState)

    def generate_save_path(self):
        # 创建以检查日期的子文件夹
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        check_date_path = os.path.join(self.save_path, date_str)
        if not os.path.exists(check_date_path):
            os.makedirs(check_date_path)

        # 获取该日期已有的检查次数，以便命名新的检查结果文件夹
        last_check_num = 0
        for folder_name in os.listdir(check_date_path):
            check_folder_path = os.path.join(check_date_path, folder_name)
            if os.path.isdir(check_folder_path) and folder_name.isdigit():
                last_check_num = max(last_check_num, int(folder_name))
        new_check_num = f"{last_check_num+1:03d}"

        # 创建检查结果文件夹
        check_result_path = os.path.join(check_date_path, new_check_num)
        os.makedirs(check_result_path)
        return check_result_path

    def wait_for_recording(self):
        while True:
            time.sleep(0.1)
            # print(self.RecodingState)
            self.create_new_path = True

            while self.RecodingState:
                if self.create_new_path:
                    self.check_result_path = self.generate_save_path()

                    scan_interval = 0.2  # 扫描间隔时间（秒）
                    # model_weights = "/home/usai/auto_RUSS/R_13_TI_RAIDS/yolov8/train4_tiny/weights/best.pt"  # YOLOv5模型权重文件路径
                    model_weights = "/home/usai/auto_RUSS/R_13_TI_RAIDS/yolo11/yolo11_best_TNS_detection.pt"  # YOLOv5模型权重文件路径
                    self.TI_RADS_thread = TI_ARDSThread(self.check_result_path, scan_interval, model_weights)
                    if not self.start_analysis:
                        self.TI_RADS_thread.start()
                        print('开始分析超声数据')
                        self.start_analysis = True
                    self.create_new_path = False
                # print(self.RecodingState)

                start_time = time.time()
                self.rate.sleep()
                
                
                rgb_image_dict = self.load_thread.get_latest_rgb_image()
                depth_image_dict = self.load_thread.get_latest_depth_image()
                seg_original_dict = self.load_thread.get_latest_seg_original()
                seg_pred_dict = self.load_thread.get_latest_seg_pred()
                franka_state_dict = self.load_thread.get_latest_state_data()
                force_data_dict = self.load_thread.get_latest_force_data()

                
                rgb_save_path = os.path.join(self.check_result_path, 'rgb_img')
                depth_save_path = os.path.join(self.check_result_path, 'depth_img')
                seg_original_save_path = os.path.join(self.check_result_path, 'seg_original_img')
                seg_pred_save_path = os.path.join(self.check_result_path, 'seg_pred_img')
                force_save_path = os.path.join(self.check_result_path, 'force_data')
                franka_save_path = os.path.join(self.check_result_path, 'franka_data')

                if not os.path.exists(rgb_save_path):
                    os.makedirs(rgb_save_path)
                if not os.path.exists(depth_save_path):
                    os.makedirs(depth_save_path)
                if not os.path.exists(seg_original_save_path):
                    os.makedirs(seg_original_save_path)
                if not os.path.exists(seg_pred_save_path):
                    os.makedirs(seg_pred_save_path)
                if not os.path.exists(force_save_path):
                    os.makedirs(force_save_path)
                    with open(os.path.join(force_save_path, 'force.csv'), 'w') as f:
                        f.write('sec,nsec,Fx,Fy,Fz,Mx,My,Mz\n')
                if not os.path.exists(franka_save_path):
                    os.makedirs(franka_save_path)
                    with open(os.path.join(franka_save_path, 'franka_state.csv'), 'w') as f:
                        f.write('sec,nsec,franka_state\n')
                
                if rgb_image_dict is not None:
                    rgb_img_list_num = len(os.listdir(rgb_save_path))
                    if rgb_img_list_num<=300:
                        cv2.imwrite(os.path.join(rgb_save_path, '{}_{:0>11}.png'.format(rgb_image_dict['secs'], rgb_image_dict['nsecs'])),
                                                    rgb_image_dict['data'])
                if depth_image_dict is not None:
                    depth_img_list_num = len(os.listdir(depth_save_path))
                    if depth_img_list_num<=300:
                        cv2.imwrite(os.path.join(depth_save_path, '{}_{:0>11}.png'.format(depth_image_dict['secs'], depth_image_dict['nsecs'])),
                                                    depth_image_dict['data'])
                if seg_original_dict is not None:
                    cv2.imwrite(os.path.join(seg_original_save_path, '{}_{:0>11}.png'.format(seg_original_dict['secs'], seg_original_dict['nsecs'])),
                                                seg_original_dict['data'])
                if seg_pred_dict is not None:
                    cv2.imwrite(os.path.join(seg_pred_save_path, '{}_{:0>11}.png'.format(seg_pred_dict['secs'], seg_pred_dict['nsecs'])),
                                                seg_pred_dict['data'])
                if force_data_dict is not None:
                    with open(os.path.join(force_save_path, 'force.csv'), 'a') as f:
                        f.write('{},{:0>11},{}\n'.format(force_data_dict['secs'], force_data_dict['nsecs'], force_data_dict['data']))
                if franka_state_dict is not None:
                    with open(os.path.join(franka_save_path, 'franka_state.csv'), 'a') as f:
                        f.write('{},{:0>11},{}\n'.format(franka_state_dict['secs'], franka_state_dict['nsecs'],  franka_state_dict['data']))

                end_time = time.time()
                print('fps: {:.2f}'.format(1/(end_time-start_time)))

            
if __name__ == '__main__':
    recorder = RecordingController()
    recorder.wait_for_recording()