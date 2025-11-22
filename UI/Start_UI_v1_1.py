import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(3, '/home/usai/auto_RUSS')
sys.path.insert(4, '/home/usai/auto_RUSS/R_UI')
sys.path.insert(5, '/home/usai/auto_RUSS/R_03_v2_keypoint_RGB_HuJB/keypoint_ZhangHR/keypoint')
from cv_bridge import CvBridge, CvBridgeError
from RobotControll_gif_v1_1 import Ui_Form
from Threading_report_image import TI_ARDSThread  # 一边扫查，一边检测结节
from TIRADS_analysis_3D import TIRADS_Analysis
from R_20_Recording_Status.config import rocord_save_path
import os
from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget, QAbstractItemView, QTableWidgetItem, QLabel
from PyQt5.QtGui import QImage, QPixmap, QTransform, QMovie
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5  import QtWidgets
import rospy
from sensor_msgs.msg import Image
from franka_example_controllers.msg import EndForceTorque
from robotiq_ft_sensor.msg import ft_sensor  
from pyqtgraph import PlotWidget
import open3d as o3d
from R_UI.keypoint_detection.KeyPoint_Thread import KeypointDetectionThread
import copy
import threading
import time
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph.opengl as gl
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util.vtkImageImportFromArray import vtkImageImportFromArray
import vtk
import fitz
from glob import glob
from R_04_3D_Reconstruction.Reconstruction_v2 import caculate_3D, get_T_probe2img, decode_string2TransMatrix
import pandas as pd
from R_UI.Start_command import start_command
from R_10_ROS_PositionForce.Thyroid_scanning_v1_2_multiThread import UltrasoundServo
from R_UI.save_dataset.dataset import Dataset
import datetime
from std_msgs.msg import Float64, Float64MultiArray
import traceback
from PIL import Image as PILImage
from franka_msgs.msg import FrankaState
import json




class LoadThread(QThread):#线程类
    color_signal = pyqtSignal(list)  #彩图，QImage格式, 只是用来看的,
    rgb_cv_img_signal = pyqtSignal(list)  #彩图，opencv格式
    depth_cv_img_signal = pyqtSignal(list)  #深度图， opencv格式
    seg_original_signal = pyqtSignal(list)  #自定义信号对象。参数str就代表这个信号可以传一个字符串
    seg_pred_signal = pyqtSignal(list)
    force_franka_signal = pyqtSignal(list)
    force_sensor_signal = pyqtSignal(list)
    franka_state = pyqtSignal(list)  #  Franka末端姿态
    
    def __init__(self):
        super(LoadThread, self).__init__()
        # # 从JSON文件中读取参数信息
        with open('/home/usai/auto_RUSS/R_16_ForceCalibration/parameters.json', 'r') as json_file:
            loaded_parameters = json.load(json_file)

        # 打印读取的参数信息
        print(loaded_parameters)
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

        self.seg_pred_count = 0
        self.seg_pred_break = 2  # 每几张显示一下，避免帧率过快造成UI不稳定
        self.force_count = 0
        self.force_break = 15

        self.init_video_node()
        time.sleep(1)

    def run(self): #线程执行函数
        rospy.spin()
        
    # 初始化节点
    def init_video_node(self):
        self.bridge = CvBridge()

        # 订阅话题
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)

        rospy.Subscriber('/Unet/original_add_pred', Image, self.seg_original)
        rospy.Subscriber('/Unet/pred', Image, self.seg_pred)

        # rospy.Subscriber('/position_force_hybird_controller/EndForceTorque', EndForceTorque, self.ForceCallback)
        rospy.Subscriber('/robotiq_ft_sensor', ft_sensor, self.force_sensor_Callback)
        # rospy.Subscriber('/robotiq_ft_sensor', ft_sensor, self.force_sensor_Callback)
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.FrankaState_Callback)


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
        image = image.scaled(QSize(346,261), 
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
        image = QPixmap.fromImage(frame)
        image = image.scaled(QSize(346,261), 
                             aspectRatioMode=Qt.KeepAspectRatio,   #保持纵横比不变
                             transformMode=Qt.SmoothTransformation)
        self.seg_pred_count += 1
        if self.seg_pred_count == self.seg_pred_break:
            self.seg_pred_signal.emit(list([frame]))
            self.seg_pred_count = 0
            # time.sleep(0.01)

    def FrankaState_Callback(self, msg):
        self.T_base2wrist = np.array(msg.O_T_EE).reshape((4, 4)).T
        self.franka_state.emit(list([self.T_base2wrist]))

    
    def force_sensor_Callback(self, msg):   
        if self.force_count == self.force_break:
            Fx = msg.Fx 
            Fy = msg.Fy 
            Fz = msg.Fz
            Mx = msg.Mx
            My = msg.My
            Mz = msg.Mz
            fm = np.array([[Fx, Fy, Fz]]).T
            M = np.array([[Mx, My, Mz]]).T
            R = self.T_base2wrist[:3, :3]
            Fx, Fy, Fz = self.get_Fe(fm, R)
            Mx, My, Mz = self.get_Me(M, R)

            # 从array拿出数值
            Fx = Fx[0]
            Fy = Fy[0]
            Fz = -Fz[0]
            Mx = Mx[0]
            My = My[0]
            Mz = Mz[0]
            external_force = pow((pow(Fx,2) + pow(Fy,2) + pow(Fz,2)), 0.5)
            self.force_sensor_signal.emit(list([Fx, Fy, Fz, external_force, Mx, My, Mz]))
            self.force_count = 0
        else:
            self.force_count += 1
    
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


    
class AutoScanningThread(QThread):   #线程类
    """自动扫查的线程类"""
    # progress_signal = pyqtSignal(int) 

    def __init__(self, goal_force=1.5, save_path=''):
        super(AutoScanningThread, self).__init__()
        self.goal_force = goal_force
        self.save_path = save_path
        self.stop_flag = False
        
    def run(self): #线程执行函数 
        self.US_servo = UltrasoundServo(self.goal_force, init_node=False, 
                                                                        save_path=self.save_path)
        self.recover_stop_process()
        if not self.stop_flag:
            self.US_servo.start_scanning()
        if not self.stop_flag:
            self.US_servo.franka.go_home()

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


class MyMainWindow(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.ret = start_command()  #启动

        self.setupUi(self)
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

        # 创建隐藏的绘图设备
        # self.hidden_RGB_pixmap = QPixmap(self.label_RGB.size())
        self.label_RGB.setAlignment(Qt.AlignCenter)  # 居中显示图像

        self.set_ui_and_function()
        # self.listener_image()

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

        self.dataset = Dataset(root_path='/home/usai/Data/Project_Dataset')
 


    def set_ui_and_function(self):
        # ========== 布局嵌入 =========
        # ----- VTK 布局
        self.vtkWidget = QVTKRenderWindowInteractor()
        self.horizontalLayout_13.addWidget(self.vtkWidget)

        self.label_11.setStyleSheet("background-color:blue")

        # ========= 信号
        self.loading_thread = LoadThread()        
        self.loading_thread.color_signal.connect(self.loading_rgb_frame)  
        self.loading_thread.rgb_cv_img_signal.connect(self.get_rgb_cv_img)
        self.loading_thread.depth_cv_img_signal.connect(self.get_depth_cv_img)
        self.loading_thread.seg_original_signal.connect(self.loading_seg_original_frame)
        self.loading_thread.seg_pred_signal.connect(self.loading_seg_pred_frame)
        self.loading_thread.force_sensor_signal.connect(self.plotting_sensor_force)
        self.loading_thread.franka_state.connect(self.get_T_base2wrist)
        self.loading_thread.start()

        self.TIRADS_analysis = TIRADS_Analysis()   

        self.AutoScanning_thread = AutoScanningThread(goal_force=2.5)

        self.KeypointDetection_thread = KeypointDetectionThread()
        self.KeypointDetection_thread.keypoint_img_signal.connect(self.loading_keypoint_img)
        self.KeypointDetection_thread.start()

    
        self.plotting_line()
        self.init_lesion_img()
        self.init_report()
        self.init_VTK()

        self.pushButton.clicked.connect(self.start_and_check)
        self.pushButton_2.clicked.connect(self.auto_scanning)
        self.pushButton_3.clicked.connect(self.OpenVTK)
        self.pushButton_4.clicked.connect(self.StartAnalysis) 
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

    # -------- 多线程的槽函数
    def loading_keypoint_img(self, sigle):
        image = sigle[0]
        self.label_20.setPixmap(image)

    def loading_rgb_frame(self, sigle):    
        image = sigle[0]
        self.label_RGB.setPixmap(image)
        

    def get_rgb_cv_img(self, data):
        self.rgb_cv_img = copy.deepcopy(data[0])
        self.KeypointDetection_thread.rgb_cv_img = copy.deepcopy(self.rgb_cv_img)  
        # print(self.rgb_cv_img.shape)
    
    def get_depth_cv_img(self, data):
        # print(data)
        self.depth_cv_img = copy.deepcopy(data[0])
        self.KeypointDetection_thread.depth_cv_img = copy.deepcopy(self.depth_cv_img)
    
    def get_T_base2wrist(self, data):
        self.T_base2wrist = copy.deepcopy(data[0])
        self.KeypointDetection_thread.T_base2wrist = self.T_base2wrist

    def loading_seg_original_frame(self, sigle):    
        time_now = time.time()
        image = sigle[0]
        self.label_US.setPixmap(image)
        self.label_US.setAlignment(Qt.AlignCenter) 
        # print('thread fps: {}'.format(1/(time.time()-time_now)))

    def loading_seg_pred_frame(self, sigle):    
        image = sigle[0]
        image = QPixmap.fromImage(image)
        image = image.scaled(self.label_seg.size(), 
                             aspectRatioMode=Qt.KeepAspectRatio,   
                             transformMode=Qt.SmoothTransformation)
        self.label_seg.setPixmap(image)
        self.label_seg.setAlignment(Qt.AlignCenter) 

    def force_value(self, data):
        self.label_7.setText('{} N'.format(data/10))

    
    def get_P(self, data):
        self.label_8.setText('{}'.format(data/100))
    
    def get_I(self, data):
        self.label_9.setText('{}'.format(data/100))
    
    def get_D(self, data):
        self.label_10.setText('{}'.format(data))

    def insert_info(self, info_str):
        self.textEdit.append(' === {} === '.format(datetime.datetime.now().strftime('%Y-%m-%d%H-%M-%S')))
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
        self.AutoScanning_thread.update_save_path(self.save_path)

        # ===== 开始扫查 =====
        check_result = self.check_scanning()
        if check_result:
            self.AutoScanning_thread.start()
    
    def stop_scanning_thread(self):
        """  停止扫查 """
        self.AutoScanning_thread.stop_scanning_process()

    def init_VTK(self):
        arr3d = np.load('/home/usai/auto_RUSS/R_UI/image/example_3D.npy')
        image = vtkImageImportFromArray()        
        image.SetArray(arr3d)                           # 加载三维矩阵
        image.Update()
        self.show_VTK(image)

    def OpenVTK(self):
        if self.save_path!='':
            self.insert_info('开始进行3D重建')
            self.vtk_thread.path = os.path.join(self.save_path, '3D')   #更新存储路径

            # 尝试打开渲染已有文件
            arr3d_path = os.path.join(self.save_path, '3D', '3Darray.npy')
            if os.path.exists(arr3d_path):
                arr3d = np.load(arr3d_path)
                image = vtkImageImportFromArray()        
                image.SetArray(arr3d)                           # 加载三维矩阵
                image.Update()
                self.show_VTK(image)

            #针对新的文件进行处理+渲染
            self.vtk_thread.start()
        else:
            self.insert_info('请确认患者信息输入正确')

    def StartAnalysis(self):
        # --- 配合边扫查边分析
        # 获取当前检查日期的子文件夹
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        check_date_path = os.path.join(rocord_save_path, date_str)

        # 获取该日期已有的检查次数，以便命名新的检查结果文件夹
        last_check_num = 0
        for folder_name in os.listdir(check_date_path):
            check_folder_path = os.path.join(check_date_path, folder_name)
            if os.path.isdir(check_folder_path) and folder_name.isdigit():
                last_check_num = max(last_check_num, int(folder_name))
        new_check_num = f"{last_check_num:03d}"

        # 创建检查结果文件夹
        check_result_path = os.path.join(check_date_path, new_check_num)

        self.TIRADS_analysis.RESULT_PATH = check_result_path
        self.TIRADS_analysis.run()
        self.report_pdf(pdf_path=os.path.join(self.TIRADS_analysis.RESULT_PATH, 'report_tmp', 'report.pdf'))
        # self.report_pdf(pdf_path=os.path.join('/home/usai/Data/For_paper/2024-09-22/007/report_tmp', 'report.pdf'))
        

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

    def show_vtk_progress(self, data):
        self.progressBar.setProperty("value", data)

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

    def force_calibration(self):
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
        df.to_csv('/home/usai/auto_RUSS/R_UI/force/force_zero.csv', index=False)
        print('Finished.')
        

    def plotting_line(self):
        self.plotWidget_ted = PlotWidget(self)
        # 创建垂直布局，并将绘图区域添加到布局中
        self.horizontalLayout_10.addWidget(self.plotWidget_ted)
        self.plotWidget_ted.setYRange(-6, 6, padding=0)   #Y轴范围

        self.plotWidget_M = PlotWidget(self)
        self.horizontalLayout_11.addWidget(self.plotWidget_M)
        self.plotWidget_M.setYRange(-1, 1, padding=0)   #Y轴范围

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

    

    def get_pointcloud(self, seg=True):
        # 创建Visualizer对象并添加点云到场景中
        self.point_cloud = o3d.geometry.PointCloud()

        to_reset = True
        # 定义ROI分割区域
        if seg:
            rect_roi = [100, 80, 510, 450]
            x1, y1, x2, y2 = rect_roi
        else:
            rect_roi = None
        
        # while True:
        # 拍摄彩图跟深度图
        color_image = copy.deepcopy(self.rgb_cv_img)
        depth_image = copy.deepcopy(self.depth_cv_img).astype(float)

        if rect_roi:
            color_image = color_image[y1:y2, x1:x2,]
            depth_image = depth_image[y1:y2, x1:x2]

        # #------ 相机内参（ros驱动之后可以看到）
        camera_matrix = np.loadtxt("R_UI/cameraMatrix.txt", delimiter=",")
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]
        # 根据相机内参矩阵计算3D坐标
        height, width, _ = color_image.shape
        py, px = np.indices((height, width))
        
        # 提取
        z = depth_image
        # 计算相机坐标系下的三维坐标
        x = (px - cx) * z / fx
        y = (py - cy) * z / fy
        points = np.stack([x/1000, y/1000, z/1000], axis=-1)
        # 将3D坐标转换为点云对象
        self.points = points.reshape(-1, 3)

        # 根据深度图生成点云        
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points)
        # print(points.shape)
        self.colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).reshape(-1,3)/255
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.colors)

        # 体素下采样的尺寸 单位m
        DOWN_SAMPLE_VOXEL_SIZE = 0.010

        # 下采样
        self.point_cloud =  self.point_cloud.voxel_down_sample(voxel_size=DOWN_SAMPLE_VOXEL_SIZE)

    def loading_VTK(self, data):
        image = data[0]
        self.show_VTK(image)

    def show_VTK(self, image):
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()#GPU
        volume_mapper.SetInputData(image.GetOutput())

        volume_color = vtk.vtkColorTransferFunction()
        volume_color.AddRGBPoint(0, 0.0, 0, 0.0)
        volume_color.AddRGBPoint(500, 240.0 / 255.0, 184.0 / 255.0, 160.0 / 255.0)
        volume_color.AddRGBPoint(1000, 240.0 / 255.0, 184.0 / 255.0, 160.0 / 255.0)
        volume_color.AddRGBPoint(1150, 1.0, 1.0, 240.0 / 255.0)

        volume_scalar_opacity = vtk.vtkPiecewiseFunction()#不透明度
        volume_scalar_opacity.AddPoint(0, 0.00)
        volume_scalar_opacity.AddPoint(500, 0.15)
        volume_scalar_opacity.AddPoint(1000, 0.15)
        volume_scalar_opacity.AddPoint(1150, 0.85)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(volume_color)
        volume_property.SetScalarOpacity(volume_scalar_opacity)
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOn()
        volume_property.SetAmbient(0.4)#控制渲染的图像的亮度，从而调节渲染的图像的明暗程度
        volume_property.SetDiffuse(0.6)#控制渲染的图像的漫反射程度，从而调节渲染的图像的颜色和纹理
        volume_property.SetSpecular(0.2)#控制渲染的图像的镜面反射程度，从而调节渲染的图像的高光和反光效果

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        ren = vtk.vtkRenderer()#渲染器
        ren.AddViewProp(volume)
        
        
        camera = ren.GetActiveCamera()#摄像机
        c = volume.GetCenter()
        camera.SetViewUp(0, 0, -1)
        camera.SetPosition(c[0], c[1] - 400, c[2])
        camera.SetFocalPoint(c[0], c[1], c[2])
        camera.Azimuth(30.0)
        camera.Elevation(30.0)

        self.vtkWidget.GetRenderWindow().AddRenderer(ren)
        self.vtkWidget.Initialize()
        

        # 取消相机旋转
        interactor = self.vtkWidget.GetRenderWindow().GetInteractor()  
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera()) 


        self.vtkWidget.Start()#启动VTK


    def init_report(self):
        movie = QMovie('/home/usai/auto_RUSS/R_UI/image/demo.gif')
        movie.setScaledSize(self.label_PDF.size()) 
        self.label_PDF.setMovie(movie)
        movie.start()


    def report_pdf(self, pdf_path=None):  
        self.update_lesion_img()

        # # 打开文件
        if pdf_path is not None:
            doc = fitz.open(pdf_path)
        else:
            doc = fitz.open(os.path.join(self.save_path, 'TI_RADS_result', 'report.pdf'))
        # doc = fitz.open('/home/uax/LiMD_example/Robot_arm/R_13_TI_RAIDS/result/report.pdf')
        self.allpages=int(doc.page_count)
        pages = 0
        page_one = doc.load_page(pages)
        page_pixmap = page_one.get_pixmap()
        #将Pixmap转换为QImage
        image_format = QImage.Format_RGBA8888 if page_pixmap.alpha else QImage.Format_RGB888
        page_image = QImage(page_pixmap.samples, page_pixmap.width,
                            page_pixmap.height, page_pixmap.stride, image_format)
        width = page_image.width()
        height = page_image.height()
        pix = QPixmap.fromImage(page_image)
        trans = QTransform()
        trans.rotate(0)  
        new = pix.transformed(trans)
        self.label_PDF.setScaledContents(True)   #自适应调整内容
        self.label_PDF.setPixmap(new)

       


    def init_lesion_img(self):         
        #     #隐藏表格线
        self.tableWidget.setShowGrid(False)
        self.tableWidget.setRowCount(1)   #重新设置行数
        self.tableWidget.setColumnCount(1)   

        self.tableWidget.setRowHeight(0, 636)  #设置单元格高度
        self.tableWidget.setColumnWidth(0, 256)


        # Insert QLabel into table cell
        label_frame = QLabel()
        label_frame.setFixedSize(256, 636)
        
        #图片（label控件）居中：
        #  1.实例化一个新布局
        hLayout = QtWidgets.QHBoxLayout()
        #  2.在布局里添加checkBox
        hLayout.addWidget(label_frame)
        #  3.在布局里居中放置checkbox1
        hLayout.setAlignment(label_frame, Qt.AlignCenter)
        #  4.实例化一个QWidget（控件）
        widget = QtWidgets.QWidget()
        #  5.在QWidget放置布局
        widget.setLayout(hLayout)
        #  6.在tableWidget1放置widget     
        self.tableWidget.setCellWidget(0, 0, widget)  
        
        movie = QMovie('/home/usai/auto_RUSS/R_UI/image/demo2.gif')
        movie.setScaledSize(label_frame.size()) 
        label_frame.setMovie(movie)
        movie.start()

       
    def update_lesion_img(self):
        #线程的槽函数
        result_path = os.path.join(self.save_path, 'TI_RADS_result')
        # result_path = '/home/uax/LiMD_example/Robot_arm/R_13_TI_RAIDS'
        img_list = sorted(glob(os.path.join(result_path, '*plot.png')))
        num = len(img_list)
        if num >0 :
            # 有结果才更新
            self.tableWidget.setRowCount(0)   #先清空
            self.tableWidget.setRowCount(num)   #重新设置行数
            self.tableWidget.setColumnCount(2)   
            self.tableWidget.setIconSize(QSize(220,133))#设置图像的大小w,h
            
            self.tableWidget.setColumnWidth(0, 20) #设置第1列宽度(图片序号)
            self.tableWidget.setColumnWidth(1, 220) #设置第2列宽度(图片缩略图)
            # self.tableWidget.setColumnWidth(2, 25) #设置第3列宽度(后期补充，用于显示是否标记)
            for num, img_path in enumerate(img_list):
                self.tableWidget.setRowHeight(num, 143)  #设置单元格高度
                frame = cv2.imread(img_path)
                # frame = cv2.resize(frame, (100, 90))  #w,h
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = frame = QImage(frame.data, # 数据源
                                        frame.shape[1],  # 宽度
                                        frame.shape[0],	# 高度
                                        frame.shape[1] * 3, # 行字节数
                                        QImage.Format_RGB888)


                    
                image = QPixmap.fromImage(image)
                image = image.scaled(QSize(190,110), 
                                aspectRatioMode=Qt.KeepAspectRatio,   #保持纵横比不变
                                transformMode=Qt.SmoothTransformation)
                
                label_frame = QLabel()
                # self.label_frame.setMinimumSize(300, 200)
                label_frame.setPixmap(image)
                #图片（label控件）居中：
                #  1.实例化一个新布局
                hLayout = QtWidgets.QHBoxLayout()
                #  2.在布局里添加checkBox
                hLayout.addWidget(label_frame)
                #  3.在布局里居中放置checkbox1
                hLayout.setAlignment(label_frame, Qt.AlignCenter)
                #  4.实例化一个QWidget（控件）
                widget = QtWidgets.QWidget()
                #  5.在QWidget放置布局
                widget.setLayout(hLayout)
                #  6.在tableWidget1放置widget     
                self.tableWidget.setCellWidget(num, 1, widget)  
                
                #---设置每张图的帧数名称
                index_Item = QTableWidgetItem(str(num+1))
                index_Item.setTextAlignment(Qt.AlignCenter)  #Qt.AlignRight右对齐、居中AlignCenter
                self.tableWidget.setItem(num, 0, index_Item) 

    def init_scanning_position(self):
        if self.rgb_cv_img is not None:
            self.KeypointDetection_thread.rgb_cv_img = copy.deepcopy(self.rgb_cv_img)
            self.KeypointDetection_thread.pre_view()
        else:
            print('初始化关键点失败, 缺少RGB图像')
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    main_window.show()
    app.exec_()
