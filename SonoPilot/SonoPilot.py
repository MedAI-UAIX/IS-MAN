import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(1, '/home/usai/IS-MAN')
sys.path.insert(2, '/home/usai/IS-MAN/SonoMind')
# sys.path.insert(2, '/home/uax/cv_bridge_ws/devel/lib/python3/dist-packages')
# sys.path.insert(3, '/home/usai/auto_RUSS/real_implement/tools/start_command')
# sys.path.insert(4, '/home/usai/auto_RUSS/R_UI_Agent')
# sys.path.insert(5, '/home/usai/auto_RUSS/R_03_v2_keypoint_RGB_HuJB/keypoint_ZhangHR/keypoint')
# sys.path.insert(6, '/home/usai/auto_RUSS/R_13_TI_RAIDS')
# sys.path.insert(7, '/home/usai/auto_RUSS/R_UI_Agent')

# from R_UI_Agent.keypoint_detection.KeyPoint_Thread import KeypointDetectionThread
# from R_UI_Agent.Start_command import start_command
# from R_10_ROS_PositionForce.Thyroid_scanning_v1_2_multiThread_multiagent  import UltrasoundServo
# from R_UI_Agent.save_dataset.dataset import Dataset
# from RobotControll_gif_v1_1 import Ui_Form
from RobotServer.keypoint_detection.KeyPoint_Thread import KeypointDetectionThread
from RobotServer.Start_command import start_command
from RobotServer.Thyroid_scanning_v1_2_multiThread_multiagent  import UltrasoundServo
from RobotServer.save_dataset.dataset import Dataset
from RobotServer.main_adapted_portable_minwidth import Ui_Form
from cv_bridge import CvBridge, CvBridgeError
import os
from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget, QAbstractItemView, QTableWidgetItem, QLabel
from PyQt5.QtGui import QImage, QPixmap, QTransform, QMovie
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5  import QtWidgets
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from franka_example_controllers.msg import EndForceTorque
from robotiq_ft_sensor.msg import ft_sensor  
from pyqtgraph import PlotWidget
# import open3d as o3d   # 注意！！！用这个库的时候，最好放到   from pyqtgraph import PlotWidget   后面
# import threading
import copy
import time
import cv2
import numpy as np
import pyqtgraph.opengl as gl
from glob import glob
import pandas as pd
import datetime
from std_msgs.msg import Float64, Float64MultiArray
import traceback
from PIL import Image as PILImage
from franka_msgs.msg import FrankaState
from flask import Flask, jsonify, request
import requests
import json
import subprocess


def kill_port(port=5006):
    """运行前强制释放端口"""
    try:
        print(f'try to kill port: {port}')
        result = subprocess.run(
            ['fuser', '-k', f'{port}/tcp'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f'[INFO] 已释放端口 {port}')
        else:
            # fuser 没发现进程时也会返回非0，属于正常情况
            pass
    except FileNotFoundError:
        print(f'[WARN] 未找到 fuser 命令，请手动确保端口 {port} 未被占用')
    except Exception as e:
        print(f'[WARN] 清理端口失败: {e}')

class LoadThread(QThread):#线程类
    color_signal = pyqtSignal(list)  #彩图，QImage格式, 只是用来看的,
    rgb_cv_img_signal = pyqtSignal(list)  #彩图，opencv格式
    depth_cv_img_signal = pyqtSignal(list)  #深度图， opencv格式
    seg_original_signal = pyqtSignal(list)  #自定义信号对象。参数str就代表这个信号可以传一个字符串
    seg_pred_signal = pyqtSignal(list)
    force_franka_signal = pyqtSignal(list)
    force_sensor_signal = pyqtSignal(list)
    franka_state = pyqtSignal(list)  #  Franka末端姿态
    llm_cmd = pyqtSignal(str)
    
    def __init__(self):
        super(LoadThread, self).__init__()
        self.init_video_node()
        # # 从JSON文件中读取参数信息
        with open('/home/usai/IS-MAN/SonoPilot/RobotServer/force/parameters.json', 'r') as json_file:
            loaded_parameters = json.load(json_file)

        # 打印读取的参数信息
        print(loaded_parameters)
        Lx = loaded_parameters['calibration']['Lx']
        Ly = loaded_parameters['calibration']['Ly']
        Lz = loaded_parameters['calibration']['Lz']
        Fx0 = loaded_parameters['calibration']['Fx0']
        Fy0 = loaded_parameters['calibration']['Fy0']
        Fz0 = loaded_parameters['calibration']['Fz0']
        self.G = np.array([[Lx, Ly, Lz]]).T
        self.F0 = np.array([[Fx0, Fy0, Fz0]]).T

    def run(self): #线程执行函数
        rospy.spin()
        
    # 初始化节点
    def init_video_node(self):
        self.bridge = CvBridge()

        # 订阅话题
        # rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        # rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        rospy.Subscriber('/gemini/depth/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/gemini/color/image_raw', Image, self.rgb_callback)

        rospy.Subscriber('/Unet/original_add_pred', Image, self.seg_original)
        rospy.Subscriber('/Unet/pred', Image, self.seg_pred)

        # rospy.Subscriber('/position_force_hybird_controller/EndForceTorque', EndForceTorque, self.ForceCallback)
        rospy.Subscriber('/robotiq_ft_sensor', ft_sensor, self.force_sensor_Callback)
        # rospy.Subscriber('/robotiq_ft_sensor', ft_sensor, self.force_sensor_Callback)
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.FrankaState_Callback)

        rospy.Subscriber('llm_command', String, self.LLMCmd_callback)


    # ----- 话题回调函数 -------
    def rgb_callback(self, data):
        rgb_cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.rgb_cv_img_signal.emit(list([rgb_cv_img]))
        frame = cv2.cvtColor(rgb_cv_img, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (110, 95))  #w,h
        frame = QImage(frame.data, # 数据源
                        frame.shape[1],  # 宽度
                        frame.shape[0],	# 高度
                        frame.shape[1] * 3, # 行字节数
                        QImage.Format_RGB888)
        image = QPixmap.fromImage(frame)
        image = image.scaled(QSize(328,243), 
                             aspectRatioMode=Qt.KeepAspectRatio,   #保持纵横比不变
                             transformMode=Qt.SmoothTransformation)
        self.color_signal.emit(list([image]))
        # time.sleep(0.01)

    def depth_callback(self, data):
        depth_cv_img = self.bridge.imgmsg_to_cv2(data, "16UC1")
        self.depth_cv_img_signal.emit(list([depth_cv_img]))
        # print('depth')
        
    def seg_original(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = QImage(frame.data, # 数据源
                        frame.shape[1],  # 宽度
                        frame.shape[0],	# 高度
                        frame.shape[1] * 3, # 行字节数
                        QImage.Format_RGB888)
        image = QPixmap.fromImage(frame)
        image = image.scaled(QSize(328,243), 
                             aspectRatioMode=Qt.KeepAspectRatio,   #保持纵横比不变
                             transformMode=Qt.SmoothTransformation)
        self.seg_original_signal.emit(list([image]))
        # time.sleep(0.01)

    def seg_pred(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")  #8UC1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = QImage(frame.data, # 数据源
                        frame.shape[1],  # 宽度
                        frame.shape[0],	# 高度
                        frame.shape[1] * 3, # 行字节数
                        QImage.Format_RGB888)
        self.seg_pred_signal.emit(list([frame]))
        # time.sleep(0.01)

    def FrankaState_Callback(self, msg):
        self.T_base2wrist = np.array(msg.O_T_EE).reshape((4, 4)).T
        self.franka_state.emit(list([self.T_base2wrist]))

    def LLMCmd_callback(self, msg):
        # rospy.loginfo("I heard %s", msg.data)
        self.llm_cmd.emit(msg.data)

    
    def force_sensor_Callback(self, msg):   
        Fx = msg.Fx 
        Fy = msg.Fy 
        Fz = msg.Fz
        fm = np.array([[Fx, Fy, Fz]]).T
        R = self.T_base2wrist[:3, :3]
        Fx, Fy, Fz = self.get_Fe(fm, R)

        # 从array拿出数值
        Fx = Fx[0]
        Fy = Fy[0]
        Fz = -Fz[0]
        external_force = pow((pow(Fx,2) + pow(Fy,2) + pow(Fz,2)), 0.5)
        Mx = msg.Mx
        My = msg.My
        Mz = msg.Mz
        self.force_sensor_signal.emit(list([Fx, Fy, Fz, external_force, Mx, My, Mz]))
    
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
        
class VTKThread(QThread):#线程类
    vtk_3D_signal = pyqtSignal(list)  #3D arr格式
    progress_signal = pyqtSignal(int) 

    def __init__(self, save_path):
        super(VTKThread, self).__init__()
        # self.path = '/home/uax/LiMD_example/Robot_arm/R_10_ROS_PositionForce/3D/examlpe_06_phantom'
        self.path = save_path
        

    
class AutoScanningThread(QThread):   #线程类
    """自动扫查的线程类"""
    # progress_signal = pyqtSignal(int) 

    def __init__(self, client_cmd, goal_force=1.5, save_path=''):
        super(AutoScanningThread, self).__init__()
        self.goal_force = goal_force
        self.save_path = save_path
        self.client_cmd = client_cmd
        self.stop_flag = False
        self.class_init_flag = False
        self.cmd_flag = ''    # 用于判断当前指令应该执行哪一个
        
    def run(self): #线程执行函数 
        if not self.class_init_flag:
            print('目标压力 {}'.format(self.goal_force))
            self.US_servo = UltrasoundServo(self.client_cmd, self.goal_force, init_node=False, 
                                                                            save_path=self.save_path)
            self.recover_stop_process()
            self.class_init_flag = True

        if self.cmd_flag == 'start_recording':
            self.US_servo.start_recording()
        elif self.cmd_flag == 'go_home':
            self.US_servo.go_home()
        elif self.cmd_flag == 'scanning_left':
            self.US_servo.scanning_left()
        elif self.cmd_flag == 'scanning_left_longitudinal':
            self.US_servo.scanning_left_longitudinal()
        elif self.cmd_flag == 'left_2_right':
            self.US_servo.left_2_right()
        elif self.cmd_flag == 'scanning_right':
            self.US_servo.scanning_right()
        elif self.cmd_flag == 'scanning_right_longitudinal':
            self.US_servo.scanning_right_longitudinal()
        elif self.cmd_flag == 'scanning_isthmus':
            self.US_servo.scanning_mid()
        elif self.cmd_flag == 'stop_recording':
            self.US_servo.stop_recording()

        elif self.cmd_flag == 'start_for_scanning':
            self.US_servo.start_for_scanning()
        elif self.cmd_flag == 'stop_for_scanning':
            self.US_servo.stop_for_scanning()
        elif self.cmd_flag == 'force_up':
            self.US_servo.force_up()
        elif self.cmd_flag == 'force_down':
            self.US_servo.force_down()
        elif self.cmd_flag == 'pause':
            if self.stop_flag:
                self.recover_stop_process()
                self.client_cmd.post(data='pause for stopping')
            else:
                self.stop_scanning_process()
                self.client_cmd.post(data='pause for restartting')
        elif self.cmd_flag == 'speed_up':
            self.US_servo.speed_up()
        elif self.cmd_flag == 'speed_down':
            self.US_servo.speed_down()
        elif self.cmd_flag == 'impedance_control':
            self.US_servo.change_control_mode('impedance_control')
        elif self.cmd_flag == 'admittance_control':
            self.US_servo.change_control_mode('admittance_control')
        elif self.cmd_flag == 'hybrid_controller':
            self.US_servo.change_control_mode('hybrid_controller')
        elif self.cmd_flag == 'cartesian_control':
            self.US_servo.change_control_mode('cartesian_control')
        else:
            print('运动指令不存在：{}'.format(self.cmd_flag))

        # self.client_cmd.post(data='{}执行完毕'.format(self.cmd_flag))

        # US_servo.for_testing()  #开始扫查
        # if not self.stop_flag:
        #     self.US_servo.start_scanning()
        # if not self.stop_flag:
        #     self.US_servo.franka.translation_xyz(z=0.1, motion_time=5, fix_frame=True)
        # if not self.stop_flag:
        #     self.US_servo.franka.go_home()

    def switch_cmd(self, cmd):
        self.cmd_flag = cmd

    def stop_scanning_process(self):
        """急停"""
        self.stop_flag = True
        self.US_servo.stop_scanning()

    def recover_stop_process(self):
        """恢复运动"""
        self.stop_flag = False
        self.US_servo.recover_stop_state()

    def update_goal_force(self, new_goal_force):
        self.goal_force = new_goal_force

    def update_save_path(self, new_save_path):
        self.save_path = new_save_path

    def update_robot_param(self):
        self.franka.update_param_safety(is_scaning=True, ext_force=self.goal_force)   # 进入力位混合控制

class FlaskServer(QThread):
    force_calibration = pyqtSignal(str)
    get_6_keypoint_signal = pyqtSignal(str)
    cmd_message = pyqtSignal(str)

    def __init__(self, port=5004):
        super().__init__()
        kill_port(port)
        self.app = Flask(__name__)
        self.port = port

        self.app.route('/franka/get_6_keypoint', methods=['GET'])(self.get_6_keypoint)
        self.app.route('/franka/get_abdomen_keypoint', methods=['GET'])(self.get_abdomen_keypoint)

        self.app.route('/franka/force_calibration', methods=['GET'])(self.get_force_calibration)
        self.app.route('/franka/force_up', methods=['GET'])(self.get_force_up)
        self.app.route('/franka/force_down', methods=['GET'])(self.get_force_down)

        self.app.route('/franka/pause', methods=['GET'])(self.get_pause)
        self.app.route('/franka/speed_up', methods=['GET'])(self.get_speed_up)
        self.app.route('/franka/speed_down', methods=['GET'])(self.get_speed_down)

        self.app.route('/franka/impedance_control', methods=['GET'])(self.get_impedance_control)
        self.app.route('/franka/hybrid_controller', methods=['GET'])(self.get_hybrid_controller)
        self.app.route('/franka/cartesian_control', methods=['GET'])(self.get_cartesian_control)
        self.app.route('/franka/admittance_control', methods=['GET'])(self.get_admittance_control)

        self.app.route('/franka/start_recording', methods=['GET'])(self.get_start_recording)
        self.app.route('/franka/go_home', methods=['GET'])(self.get_go_home)
        self.app.route('/franka/scanning_left', methods=['GET'])(self.get_scanning_left)
        self.app.route('/franka/scanning_left_longitudinal', methods=['GET'])(self.get_scanning_left_longitudinal)
        self.app.route('/franka/left_2_right', methods=['GET'])(self.get_left_2_right)
        self.app.route('/franka/scanning_right', methods=['GET'])(self.get_scanning_right)
        self.app.route('/franka/scanning_right_longitudinal', methods=['GET'])(self.get_scanning_right_longitudinal)
        self.app.route('/franka/scanning_isthmus', methods=['GET'])(self.get_scanning_isthmus)
        self.app.route('/franka/stop_recording', methods=['GET'])(self.get_stop_recording)
        self.app.route('/franka/start_for_scanning', methods=['GET'])(self.start_for_scanning)
        self.app.route('/franka/stop_for_scanning', methods=['GET'])(self.stop_for_scanning)
        self.app.route('/franka/recieve_info', methods=['POST'])(self.recieve_info)
        print('loading flask')

    def run(self):
        print('准备运行flask')
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
        print('运行flask')
    
    def get_6_keypoint(self):
        self.cmd_message.emit('get_6_keypoint')
        print('get_6_keypoint')
        return jsonify({'message': 'ok'})
    
    def get_abdomen_keypoint(self):
        self.cmd_message.emit('get_abdomen_keypoint')
        print('get_abdomen_keypoint')
        return jsonify({'message': 'ok'})
    
    def get_force_calibration(self):
        self.force_calibration.emit('force_calibration')
        print('force_calibration')
        return jsonify({'message': 'ok'})

    def get_force_up(self):
        self.cmd_message.emit('force_up')
        print('force_up')
        return jsonify({'message': 'ok'})
    
    def get_force_down(self):
        self.cmd_message.emit('force_down')
        print('force_down')
        return jsonify({'message': 'ok'})

    def get_pause(self):
        self.cmd_message.emit('pause')
        print('pause')
        return jsonify({'message': 'ok'})
    def get_speed_up(self):
        self.cmd_message.emit('speed_up')
        print('speed_up')
        return jsonify({'message': 'ok'})
    def get_speed_down(self):
        self.cmd_message.emit('speed_down')
        print('speed_down')
        return jsonify({'message': 'ok'})
    #==== controller ====
    def get_impedance_control(self):
        self.cmd_message.emit('impedance_control')
        print('impedance_control')
        return jsonify({'message': 'ok'})
    def get_hybrid_controller(self):
        self.cmd_message.emit('hybrid_controller')
        print('hybrid_controller')
        return jsonify({'message': 'ok'})
    def get_cartesian_control(self):
        self.cmd_message.emit('cartesian_control')
        print('cartesian_control')
        return jsonify({'message': 'ok'})
    def get_admittance_control(self):
        self.cmd_message.emit('admittance_control')
        print('admittance_control')
        return jsonify({'message': 'ok'})
    
    def get_start_recording(self):
        self.cmd_message.emit('start_recording')
        print('start_recording')
        return jsonify({'message': 'ok'})
    
    def get_go_home(self):
        self.cmd_message.emit('go_home')
        print('go_home')
        return jsonify({'message': 'success'})
    
    def get_scanning_left(self):
        self.cmd_message.emit('scanning_left')
        print('scanning_left')
        return jsonify({'message': 'ok'})
    
    def get_scanning_left_longitudinal(self):
        self.cmd_message.emit('scanning_left_longitudinal')
        print('scanning_left_longitudinal')
        return jsonify({'message': 'ok'})
    
    def get_left_2_right(self):
        self.cmd_message.emit('left_2_right')
        print('left_2_right')
        return jsonify({'message': 'ok'})
    
    def get_scanning_right(self):
        self.cmd_message.emit('scanning_right')
        print('scanning_right')
        return jsonify({'message': 'ok'})
    
    def get_scanning_right_longitudinal(self):
        self.cmd_message.emit('scanning_right_longitudinal')
        print('scanning_right_longitudinal')
        return jsonify({'message': 'ok'})
    
    def get_scanning_isthmus(self):
        self.cmd_message.emit('scanning_isthmus')
        print('scanning_isthmus')
        return jsonify({'message': 'ok'})
    
    def get_stop_recording(self):
        self.cmd_message.emit('stop_recording')
        print('stop_recording')
        return jsonify({'message': 'ok'})
    
    def start_for_scanning(self):
        self.cmd_message.emit('start_for_scanning')
        print('start_for_scanning')
        return jsonify({'message': 'ok'})
    
    def stop_for_scanning(self):
        self.cmd_message.emit('stop_for_scanning')
        print('stop_for_scanning')
        return jsonify({'message': 'ok'})
    
    def recieve_info(self):
        data = request.get_json()  # 获取请求中的 json 数据
        return jsonify({'received_data': data}), 200



class RestClient:
    def __init__(self, base_url):
        """
                base_url = 'http://127.0.0.1:5000'
                client = RestClient(base_url)

                # 测试 GET 方法
                get_response = client.get('/')
                print('GET response:', get_response)

                # 测试 POST 方法
                post_data = {
                    'key1': 'value1',
                    'key2': 'value2'
                }
                post_response = client.post('/post', post_data)
                print('POST response:', post_response)
        """
        self.base_url = base_url

    def get(self, endpoint, params=None):
        url = f'{self.base_url}{endpoint}'
        response = requests.get(url, params=params)
        return response.json()

    def post(self, endpoint='/agent/recieve_info_state', data=''):
        post_data = {
                    'message': data,
                }
        url = f'{self.base_url}{endpoint}'
        headers = {'Content-Type': 'application/json'}
        print(post_data)
        response = requests.post(url, data=json.dumps(post_data), headers=headers)
        # return response.json()
        print(response.text)
    
    def post_message(self, endpoint='/agent/recieve_info_message', data=''):
        post_data = {
                    'message': data,
                }
        url = f'{self.base_url}{endpoint}'
        headers = {'Content-Type': 'application/json'}
        print(post_data)
        response = requests.post(url, data=json.dumps(post_data), headers=headers)
        return response.json()
    
    def post_node(self, endpoint='/agent/recieve_info_node', data=''):
        """发送是否存在结节"""
        post_data = {
                    'message': data,
                }
        url = f'{self.base_url}{endpoint}'
        headers = {'Content-Type': 'application/json'}
        print(post_data)
        response = requests.post(url, data=json.dumps(post_data), headers=headers)
        return response.json()
        

class MyMainWindow(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.ret = start_command()  #启动

        self.setupUi(self)

        # ---- Agent 智能体通讯 ----
        self.flask_server = FlaskServer()
        self.flask_server.force_calibration.connect(self.force_calibration)
        self.flask_server.cmd_message.connect(self.agent_cmd_thread)
        self.flask_server.get_6_keypoint_signal.connect(self.init_scanning_position)
        # self.flask_server.go_home.connect()
        # self.flask_server.scanning_left.connect()
        # self.flask_server.scanning_left_longitudinal.connect()
        # self.flask_server.left_2_right.connect()
        # self.flask_server.scanning_right.connect()
        # self.flask_server.scanning_right_longitudinal.connect()
        # self.flask_server.scanning_mid.connect()
        # self.flask_server.stop_recording.connect()
        self.flask_server.start()

        base_url = 'http://127.0.0.1:5006'
        self.client = RestClient(base_url)

        # ---- ros 初始化
        rospy.init_node('control_panel', anonymous=True)
        self.bridge = CvBridge()

        

        #  --- 变量 -----
        self.save_path = ''
        self.init_num = 500  #用于校准的数量
        self.force_init_x = 0
        self.force_init_y = 0
        self.force_init_z = 0
        self.force_init_mx = 0
        self.force_init_my = 0
        self.force_init_mz = 0
        self.force_init_list_x = []
        self.force_init_list_y = []
        self.force_init_list_z = []
        self.force_init_list_mx = []
        self.force_init_list_my = []
        self.force_init_list_mz = []

        self.all_num = 1000   #最多显示折线图x轴的个数
        self.force_sensor_x = []
        self.force_sensor_y = []
        self.force_sensor_z = []
        self.force_sensor_external_force = []
        self.force_sensor_mx = []
        self.force_sensor_my = []
        self.force_sensor_mz = []

        self.rgb_cv_img = None
        self.history_action = ''

        # 创建隐藏的绘图设备
        # self.hidden_RGB_pixmap = QPixmap(self.label_RGB.size())
        self.label_RGB.setAlignment(Qt.AlignCenter)  # 居中显示图像

        self.set_ui_and_function()   # 各类线程，信号槽
        # self.listener_image()

        try:
            df = pd.read_csv('/home/usai/IS-MAN/SonoPilot/RobotServer/force/force_zero.csv')
            self.force_init_x = df['x'].values[0]
            self.force_init_y = df['y'].values[0]
            self.force_init_z = df['z'].values[0]
            self.force_init_mx = df['mx'].values[0]
            self.force_init_my = df['my'].values[0]
            self.force_init_mz = df['mz'].values[0]
        except:
            pass

        self.dataset = Dataset(root_path='/home/usai/IS-MAN/SonoMind/patient')



    def set_ui_and_function(self):
        # ========== 布局嵌入 =========
        # # ----- VTK 布局
        # self.vtkWidget = QVTKRenderWindowInteractor()
        # self.horizontalLayout_13.addWidget(self.vtkWidget)

        self.label_11.setStyleSheet("background-color:blue")

        # ========= 信号
        # 进行一些布局的修改, 信号的连接
        self.loading_thread = LoadThread()        
        self.loading_thread.color_signal.connect(self.loading_rgb_frame)  #自定义的信号，改与主函数的哪个槽函数进行联动
        self.loading_thread.rgb_cv_img_signal.connect(self.get_rgb_cv_img)
        self.loading_thread.depth_cv_img_signal.connect(self.get_depth_cv_img)
        self.loading_thread.seg_original_signal.connect(self.loading_seg_original_frame)
        self.loading_thread.seg_pred_signal.connect(self.loading_seg_pred_frame)
        self.loading_thread.force_sensor_signal.connect(self.plotting_sensor_force)
        self.loading_thread.franka_state.connect(self.get_T_base2wrist)
        # self.loading_thread.llm_cmd.connect(self.agent_llm)
        self.loading_thread.start()#启动线程

        # self.loading_thread_RGB = LoadRGBThread()
        # self.loading_thread_RGB.color_signal.connect(self.loading_rgb_frame)
        # self.loading_thread_RGB.start()

        #----- 3D重建线程
        # # self.vtk_thread = VTKThread(save_path=self.save_path)
        # self.vtk_thread.vtk_3D_signal.connect(self.loading_VTK)
        # self.vtk_thread.progress_signal.connect(self.show_vtk_progress)

        #----- 甲状腺TI-RADS分析线程
        # self.TI_RADS_thread = TI_ARDSThread()
        # self.TI_RADS_thread.progress_signal.connect(self.show_report_process)

        #----- 自动扫查线程
        self.AutoScanning_thread = AutoScanningThread(self.client, goal_force=2)

        #----- 关键点检测线程
        self.KeypointDetection_thread = KeypointDetectionThread()
        self.KeypointDetection_thread.keypoint_img_signal.connect(self.loading_keypoint_img)
        self.KeypointDetection_thread.start()

        

        self.plotting_line()
        # self.init_lesion_img()
        # self.init_report()
        # self.init_VTK()

        self.pushButton.clicked.connect(self.start_and_check)
        self.pushButton_2.clicked.connect(self.auto_scanning)
        # self.pushButton_3.clicked.connect(self.OpenVTK)
        self.pushButton_4.clicked.connect(self.StartAnalysis) #report_pdf
        # self.pushButton_5.clicked.connect(self.init_PointCloud)
        self.pushButton_5.clicked.connect(self.init_scanning_position)
        self.pushButton_6.clicked.connect(self.force_calibration)
        self.pushButton_7.clicked.connect(self.stop_scanning_thread)
        self.pushButton_8.clicked.connect(self.pub_keypoint)
        self.pushButton_9.clicked.connect(self.del_keypoint)
        self.dial.valueChanged.connect(self.force_value)
        self.horizontalSlider.valueChanged.connect(self.get_P)
        self.horizontalSlider_2.valueChanged.connect(self.get_I)
        self.horizontalSlider_3.valueChanged.connect(self.get_D)
        self.radioButton.toggled.connect(self.select_keypoint_way)

    # ------ Agent 扫查的方法
    def agent_cmd_thread(self, cmd):
        print('当前cmd指令: {}'.format(cmd))
        if cmd == "force_calibration":
            self.force_calibration(return_state=True)

        elif cmd == "get_6_keypoint":
            self.init_scanning_position(return_state=True)
        else:
            self.AutoScanning_thread.switch_cmd(cmd)
            self.AutoScanning_thread.start()


        # elif cmd == "force_up":
        #     self.AutoScanning_thread.switch_cmd(cmd)
        #     self.AutoScanning_thread.start()
        # elif cmd == "force_down":
        #     self.AutoScanning_thread.switch_cmd(cmd)
        #     self.AutoScanning_thread.start()
        # elif cmd == 'pause':
        #     self.AutoScanning_thread.switch_cmd(cmd)
        #     self.AutoScanning_thread.start()
        # elif cmd == 'speed_up':
        #     self.AutoScanning_thread.switch_cmd(cmd)
        #     self.AutoScanning_thread.start()
        # elif cmd == "speed_down":
        #     self.AutoScanning_thread.switch_cmd(cmd)
        #     self.AutoScanning_thread.start()
        # elif cmd == "impedance_control":
        #     self.AutoScanning_thread.switch_cmd(cmd)
        #     self.AutoScanning_thread.start()
        # elif cmd == "hybrid_controller":
        #     self.AutoScanning_thread.switch_cmd(cmd)
        #     self.AutoScanning_thread.start()
        # elif cmd == "cartesian_control":
        #     self.AutoScanning_thread.switch_cmd(cmd)
        #     self.AutoScanning_thread.start()
        # elif cmd == "admittance_control":
        #     self.AutoScanning_thread.switch_cmd(cmd)
        #     self.AutoScanning_thread.start()

        # if cmd == 'start_for_scanning':
        #     self.init_scanning_position(return_state=False)
        #     self.force_calibration(return_state=False)
        # self.AutoScanning_thread.switch_cmd(cmd)
        # self.AutoScanning_thread.start()

    # -------- 多线程的槽函数
    def loading_keypoint_img(self, sigle):
        image = sigle[0]
        self.label_20.setPixmap(image)

    def loading_rgb_frame(self, sigle):    
        image = sigle[0]
        self.label_RGB.setPixmap(image)
        

    def get_rgb_cv_img(self, data):
        self.rgb_cv_img = copy.deepcopy(data[0])
        self.KeypointDetection_thread.rgb_cv_img = copy.deepcopy(self.rgb_cv_img)  #更新多线程里面的图像
        # print(self.rgb_cv_img.shape)
    
    def get_depth_cv_img(self, data):
        # print(data)
        self.depth_cv_img = copy.deepcopy(data[0])
        self.KeypointDetection_thread.depth_cv_img = copy.deepcopy(self.depth_cv_img)  #更新多线程里面的图像
    
    def get_T_base2wrist(self, data):
        self.T_base2wrist = copy.deepcopy(data[0])
        self.KeypointDetection_thread.T_base2wrist = self.T_base2wrist

    def loading_seg_original_frame(self, sigle):    
        time_now = time.time()
        image = sigle[0]
        # image = QPixmap.fromImage(image)
        # image = image.scaled(self.label_US.size(), 
        #                     aspectRatioMode=Qt.KeepAspectRatio,   #保持纵横比不变
        #                     transformMode=Qt.SmoothTransformation)
        self.label_US.setPixmap(image)
        self.label_US.setAlignment(Qt.AlignCenter)  # 居中显示图像
        # print('thread fps: {}'.format(1/(time.time()-time_now)))

    def loading_seg_pred_frame(self, sigle):    
        image = sigle[0]
        image = QPixmap.fromImage(image)
        image = image.scaled(self.label_seg.size(), 
                             aspectRatioMode=Qt.KeepAspectRatio,   #保持纵横比不变
                             transformMode=Qt.SmoothTransformation)
        self.label_seg.setPixmap(image)
        self.label_seg.setAlignment(Qt.AlignCenter)  # 居中显示图像

    def agent_llm(self, msg):
        try:
            if msg== '查询':
                pass
            else:
                agent_plan_output = eval(msg)
                if agent_plan_output != self.history_action:
                    self.history_action = agent_plan_output
                    for each in agent_plan_output['function']: # 运行智能体规划编排的每个函数
                        print('开始执行动作', each)
                        eval(each)
        except:
            print('解析/执行 出现问题，原始数据如下：{}'.format(msg))

    def force_value(self, data):
        self.label_7.setText('{} N'.format(data/10))

    
    def get_P(self, data):
        self.label_8.setText('{}'.format(data/100))
    
    def get_I(self, data):
        self.label_9.setText('{}'.format(data/100))
    
    def get_D(self, data):
        self.label_10.setText('{}'.format(data))

    def insert_info(self, info_str):
        self.textEdit.append(' === {} === '.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.textEdit.append(info_str)

    def select_keypoint_way(self):
        """ 用哪种关键点测试方式进行推理 """
        if self.radioButton.isChecked():
            """方法1"""
            self.KeypointDetection_thread.chaneg_test_mode('1')
        elif self.radioButton_2.isChecked():
            """方法2"""
            self.KeypointDetection_thread.chaneg_test_mode('2')
        elif self.radioButton_3.isChecked():
            """方法3"""
            self.KeypointDetection_thread.chaneg_test_mode('3')

    def pub_keypoint(self):
        """ 发布关键点 """
        self.KeypointDetection_thread.publish_keypoint()


    def del_keypoint(self):
        """ 清除关键点 """
        self.KeypointDetection_thread.clear_keypoint()

    def check_scanning(self):
        try:
            seg_original = rospy.wait_for_message('/left_thyroid', Float64MultiArray, timeout=1.0)
            seg_original = rospy.wait_for_message('/right_thyroid', Float64MultiArray, timeout=1.0)
            seg_original = rospy.wait_for_message('/mid_thyroid', Float64MultiArray, timeout=1.0)
            self.insert_info('已检测到3个扫查初始点, 准备开始扫查')
            return True
        except:
            self.insert_info('扫查目标初始化未完成, 请检查！')
            return False

    def auto_scanning(self):
        # ===== 存储数据 ========
        patient_id = self.lineEdit_2.text()
        name = self.lineEdit.text()
        gender = self.comboBox.currentText()
        age = self.lineEdit_3.text()
        diagnosis = self.lineEdit_4.text()
        description = self.lineEdit_5.text()
        other = '第xxx次扫查'
    
        #创建存储文件夹
        self.dataset.create_subject(subject_id=self.lineEdit_2.text(),
                                                                name=name, gender=gender)
        self.save_path = self.dataset.generate_save_path(subject_id=self.lineEdit_2.text())
        self.dataset.save_patient_data(self.save_path, patient_id, name, gender, age, diagnosis, description, other)
        self.insert_info('已创建检查者信息')

        # ===== 设定压力参数+数据存储路径 ======
        new_goal_force = float(self.label_7.text().strip('N'))
        self.AutoScanning_thread.update_goal_force(new_goal_force)
        self.AutoScanning_thread.update_save_path(self.save_path)

        # ===== 开始扫查 =====
        check_result = self.check_scanning()
        if check_result:
            self.AutoScanning_thread.start()
    
    def stop_scanning_thread(self):
        """  停止扫查 """
        self.AutoScanning_thread.stop_scanning_process()

    def StartAnalysis(self):
        if self.save_path!='':
            video_path = os.path.join(self.save_path, 'video_US.avi')
            result_path = os.path.join(self.save_path, 'TI_RADS_result')
            file_state = self.is_file_complete(video_path)
            if file_state:
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                
                self.TI_RADS_thread.VIDEO_PATH = video_path
                self.TI_RADS_thread.RESULT_PATH = result_path
                self.TI_RADS_thread.start()
                self.insert_info('开始检测TI-RADS')
            else:
                self.insert_info('US视频不存在or不完整, 请检查！')
        else:
            self.insert_info('请确认患者信息输入正确')

    def is_file_complete(self, file_path):
        """判断文件是否已经存在  并且完整"""
        if not os.path.exists(file_path):
            return False
        else:
            cap = cv2.VideoCapture(file_path)
            all_frame_num  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if all_frame_num>0:
                return True
            else:
                return False

    # def show_vtk_progress(self, data):
    #     self.progressBar.setProperty("value", data)

    def show_report_process(self, data):
        self.progressBar_2.setProperty("value", data)
        if data == 100:
            self.report_pdf()

    def start_and_check(self):
        self.textEdit.setPlainText('正在开启，系统自检中...')
        if self.ret:
            self.insert_info('系统已经启动')
        else:
            self.insert_info('系统启动失败，请联系开发人员')

    def plotting_sensor_force(self, data):
        # print(data)
        Fx, Fy, Fz, external_force, Mx, My, Mz = data 
        self.force_sensor_x.append(Fx-self.force_init_x)
        self.force_sensor_y.append(Fy-self.force_init_y)
        self.force_sensor_z.append(Fz-self.force_init_z)
        # self.force_sensor_external_force.append(external_force)
        self.force_sensor_mx.append(Mx-self.force_init_mx)
        self.force_sensor_my.append(My-self.force_init_my)
        self.force_sensor_mz.append(Mz-self.force_init_mz)
        if len(self.force_sensor_x) > self.all_num:
            self.force_sensor_x = self.force_sensor_x[-self.all_num:]
            self.force_sensor_y = self.force_sensor_y[-self.all_num:]
            self.force_sensor_z = self.force_sensor_z[-self.all_num:]
            # self.force_sensor_external_force = self.force_sensor_external_force[-self.all_num:]
            self.force_sensor_mx = self.force_sensor_mx[-self.all_num:]
            self.force_sensor_my = self.force_sensor_my[-self.all_num:]
            self.force_sensor_mz = self.force_sensor_mz[-self.all_num:]
        self.update_plot_data()

        self.force_init_list_x.append(Fx)
        self.force_init_list_y.append(Fy)
        self.force_init_list_z.append(Fz)
        self.force_init_list_mx.append(Mx)
        self.force_init_list_my.append(My)
        self.force_init_list_mz.append(Mz)
        if len(self.force_init_list_x) > self.init_num:
            self.force_init_list_x = self.force_init_list_x[-self.init_num:]
            self.force_init_list_y = self.force_init_list_y[-self.init_num:]
            self.force_init_list_z = self.force_init_list_z[-self.init_num:]
            self.force_init_list_mx = self.force_init_list_mx[-self.init_num:]
            self.force_init_list_my = self.force_init_list_my[-self.init_num:]
            self.force_init_list_mz = self.force_init_list_mz[-self.init_num:]

    def force_calibration(self, return_state=True):
        self.force_init_x = np.mean(self.force_init_list_x)
        self.force_init_y = np.mean(self.force_init_list_y)
        self.force_init_z = np.mean(self.force_init_list_z)
        self.force_init_mx = np.mean(self.force_init_list_mx)
        self.force_init_my = np.mean(self.force_init_list_my)
        self.force_init_mz = np.mean(self.force_init_list_mz)
        # #np.mean([0.1, 0.2, 2,3,4,5,6,7,8])

        df = pd.DataFrame([{'x': self.force_init_x,
                                                'y': self.force_init_y,
                                                'z': self.force_init_z,
                                                'mx': self.force_init_mx,
                                                'my': self.force_init_my,
                                                'mz': self.force_init_mz}])
        df.to_csv('/home/usai/IS-MAN/SonoPilot/RobotServer/force/force_zero.csv', index=False)
        print('Finished.')
        if return_state:
            self.client.post(data='sucess')
        

    def plotting_line(self):
        self.plotWidget_ted = PlotWidget(self)
        # 创建垂直布局，并将绘图区域添加到布局中
        self.horizontalLayout_10.addWidget(self.plotWidget_ted)
        self.plotWidget_ted.setYRange(-8, 8, padding=0)   #Y轴范围

        self.plotWidget_M = PlotWidget(self)
        self.horizontalLayout_11.addWidget(self.plotWidget_M)
        self.plotWidget_M.setYRange(-2, 2, padding=0)   #Y轴范围

        self.curvel_Fx = self.plotWidget_ted.plot(np.array([]), pen=(255,0,0), name="x red")
        self.curvel_Fy = self.plotWidget_ted.plot(np.array([]), pen=(0,255,0), name="y green")
        self.curvel_Fz = self.plotWidget_ted.plot(np.array([]), pen=(0,0,255), name="z blue")
        self.curvel_external_force = self.plotWidget_ted.plot(np.array([]), pen=(0,255,255), name="external_force")

        self.curvel_Mx = self.plotWidget_M.plot(np.array([]), pen=(255,0,0), name="x red")
        self.curvel_My = self.plotWidget_M.plot(np.array([]), pen=(0,255,0), name="y green")
        self.curvel_Mz = self.plotWidget_M.plot(np.array([]), pen=(0,0,255), name="z blue")

    def update_plot_data(self):
        self.curvel_Fx.setData(np.array(self.force_sensor_x))
        self.curvel_Fy.setData(np.array(self.force_sensor_y))
        self.curvel_Fz.setData(np.array(self.force_sensor_z))
        # self.curvel_external_force.setData(np.array(self.force_sensor_external_force))
        self.curvel_Mx.setData(np.array(self.force_sensor_mx))
        self.curvel_My.setData(np.array(self.force_sensor_my))
        self.curvel_Mz.setData(np.array(self.force_sensor_mz))

    def init_scanning_position(self, return_state=True):
        print('keypoint init start')
        if self.rgb_cv_img is not None:
            self.KeypointDetection_thread.rgb_cv_img = copy.deepcopy(self.rgb_cv_img)
            time.sleep(0.5)
            self.KeypointDetection_thread.pre_view()
            time.sleep(2)
            self.KeypointDetection_thread.publish_keypoint()
            if return_state:
                self.client.post(data='sucess')
        else:
            print('error')
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    main_window.show()
    print('退出')
    app.exec_()
