import sys
# 将keypoint文件夹加入环境变量，否则会找不到包
sys.path.append('/home/uax/LiMD_example/Robot_arm/R_03_keypoint/NeckNet')
from R_03_keypoint.yolov8.predict_6_point import get_thy_keypoint
from ultralytics import YOLO

from cv_bridge import CvBridge
import os

import rospy
import copy
import open3d as o3d
import cv2
import numpy as np


from glob import glob
from std_msgs.msg import Float64, Float64MultiArray
import traceback
# from yolo import YOLO
# from hrnet import Hrnet
from PIL import Image as PILImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QTransform, QMovie
import copy
import scipy.linalg as la
from math import pi

class KeypointDetectionThread(QThread):   #线程类
    """关键点检测模型的线程类"""
    keypoint_img_signal = pyqtSignal(list) 

    def __init__(self, goal_force=2, save_path='', test_mode='1'):
        super(KeypointDetectionThread, self).__init__()

        # === keypoint detection
        self.model_yolov8 = YOLO("/home/usai/auto_RUSS/R_03_keypoint/yolo11/yolo11m_pose_best_thy_ketpoint.pt")  # load an official model
        # Predict with the model
        rgb_img_original = cv2.imread("/home/usai/auto_RUSS/R_03_keypoint/yolo11/yolo11m_pose_best_thy_ketpoint.pt")  #第一次推理需要加载，耗时
        results = self.model_yolov8(rgb_img_original)  # predict on an image

        self.test_mode = test_mode  # 推理的方式，有1/2/3种
        self.select_point = []  # 存储关键点

        self.bridge = CvBridge()
        self.init_publisher()  #初始化发布节点

        self.rgb_cv_img = None 
        self.depth_cv_img = None
        self.frame_rgb_for_calculate = None
        self.frame_depth_for_calculate = None
        self.T_base2wrist = None
        self.publish_flag = False
        self.need_caculate_normal = True

        #======= 相机在机器人基底坐标系下的变换 ============
        #------ 相机内参（ros驱动之后可以看到）
        camera_matrix = np.loadtxt("/home/uax/LiMD_example/Robot_arm/R_UI/keypoint_detection/cameraMatrix.txt", delimiter=",")
        self.fx = camera_matrix[0][0]
        self.fy = camera_matrix[1][1]
        self.cx = camera_matrix[0][2]
        self.cy = camera_matrix[1][2]

        #手眼标定获得，相机坐标系，相对于基底坐标系的变换矩阵T
        R_wrist2cam = np.load('/home/uax/LiMD_example/Robot_arm/R_UI/keypoint_detection/R_wrist2cam.npz.npy').astype(np.float32)
        t_wrist2cam = np.load('/home/uax/LiMD_example/Robot_arm/R_UI/keypoint_detection/t_wrist2cam.npz.npy').T.astype(np.float32)
        self.T_wrist2cam = np.eye(4).astype(np.float32)
        self.T_wrist2cam[:3,:3] = R_wrist2cam
        self.T_wrist2cam[:3, 3] = t_wrist2cam  #*1000.0
        
    def run(self):
        while True:
            self.rate.sleep()
            while self.publish_flag:
                if len(self.select_point) != 6:
                    print('关键点个数为: {}, 请检查'.format(len(self.select_point)))
                    self.publish_flag = False
                    self.need_caculate_normal = True
                else:
                    if self.need_caculate_normal:
                        left_up, left_down, right_up, right_down, mid_up, mid_down = self.select_point
                        self.left_Trans, left_cam2object = self.keypoint_normal(left_up[0], left_up[1], left_down[0], left_down[1])
                        self.right_Trans, right_cam2object = self.keypoint_normal(right_up[0], right_up[1], right_down[0], right_down[1])
                        self.mid_Trans, mid_cam2object = self.keypoint_normal(mid_up[0], mid_up[1], mid_down[0], mid_down[1])
                        self.need_caculate_normal = False

                    self.axis_publisher()    # 发布

    def pre_view(self):
        """预览检测结果"""
        if self.rgb_cv_img is not None:
            self.frame_rgb_for_calculate = copy.deepcopy(self.rgb_cv_img)
            self.frame_depth_for_calculate = copy.deepcopy(self.depth_cv_img).astype(float)

            frame, self.select_point = get_thy_keypoint(model=self.model_yolov8, rgb_img_original=self.frame_rgb_for_calculate, way=self.test_mode)
            self.need_caculate_normal = True
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(image.data, # 数据源
                            image.shape[1],  # 宽度
                            image.shape[0],	# 高度
                            image.shape[1] * 3, # 行字节数
                            QImage.Format_RGB888)
            image = QPixmap.fromImage(frame)
            image = image.scaled(QSize(328,243), 
                                aspectRatioMode=Qt.KeepAspectRatio,   #保持纵横比不变
                                transformMode=Qt.SmoothTransformation)
            self.keypoint_img_signal.emit(list([image]))

            #重置None，防止中间被意外调用
            self.rgb_cv_img = None
        else:
            print('未获取可用于关键点检测的RGB图片, 请检查信号来源')


    def publish_keypoint(self):
        """ 发布 """
        self.publish_flag = True
        self.need_caculate_normal = True

    def chaneg_test_mode(self, mode:str):
        """ 切换测试的方法 """
        self.test_mode = mode

    def clear_keypoint(self):
        """ 清除已经存储的关键点 """
        self.select_point = []
        self.publish_flag = False
        self.need_caculate_normal = True

    def init_publisher(self):
        """ 初始化话题发布器 """
        self.left_thyroid_pub = rospy.Publisher('/left_thyroid', Float64MultiArray, queue_size=10)
        self.right_thyroid_pub = rospy.Publisher('/right_thyroid', Float64MultiArray, queue_size=10)
        self.mid_thyroid_pub = rospy.Publisher('/mid_thyroid', Float64MultiArray, queue_size=10)

        #设置循环的频率
        self.rate = rospy.Rate(10)

    def axis_publisher(self):
            """ 发布数据 """
            left_msg = Float64MultiArray()
            right_msg = Float64MultiArray()
            mid_msg = Float64MultiArray()

            right_msg.data = [self.left_Trans[0][0], self.left_Trans[0][1], self.left_Trans[0][2], self.left_Trans[0][3],
                                              self.left_Trans[1][0], self.left_Trans[1][1], self.left_Trans[1][2], self.left_Trans[1][3],
                                              self.left_Trans[2][0], self.left_Trans[2][1], self.left_Trans[2][2], self.left_Trans[2][3],
                                              self.left_Trans[3][0], self.left_Trans[3][1], self.left_Trans[3][2], self.left_Trans[3][3]]
            left_msg.data = [self.right_Trans[0][0], self.right_Trans[0][1], self.right_Trans[0][2], self.right_Trans[0][3],
                                                self.right_Trans[1][0], self.right_Trans[1][1], self.right_Trans[1][2], self.right_Trans[1][3],
                                                self.right_Trans[2][0], self.right_Trans[2][1], self.right_Trans[2][2], self.right_Trans[2][3],
                                                self.right_Trans[3][0], self.right_Trans[3][1], self.right_Trans[3][2], self.right_Trans[3][3]]
            mid_msg.data = [self.mid_Trans[0][0], self.mid_Trans[0][1], self.mid_Trans[0][2], self.mid_Trans[0][3],
                                                self.mid_Trans[1][0], self.mid_Trans[1][1], self.mid_Trans[1][2], self.mid_Trans[1][3],
                                                self.mid_Trans[2][0], self.mid_Trans[2][1], self.mid_Trans[2][2], self.mid_Trans[2][3],
                                                self.mid_Trans[3][0], self.mid_Trans[3][1], self.mid_Trans[3][2], self.mid_Trans[3][3]]

            # 发布消息
            self.left_thyroid_pub.publish(left_msg)
            self.right_thyroid_pub.publish(right_msg)
            self.mid_thyroid_pub.publish(mid_msg)

            self.rate.sleep()

    def vector_normal(self, vector):
        """向量归一化"""
        x, y, z = vector
        w = pow((pow(x, 2) + pow(y, 2) + pow(z, 2)), 0.5)
        normal = np.array([x/w, y/w, z/w])
        return normal

    def keypoint_normal(self, x_center, y_center, x_direction, y_direction):
        # 创建Visualizer对象并添加点云到场景中
        point_cloud = o3d.geometry.PointCloud()
        
        # 拍摄彩图跟深度图
        # color_image = copy.deepcopy(self.rgb_cv_img)
        # depth_image = copy.deepcopy(self.depth_cv_img).astype(float)
        color_image = copy.deepcopy(self.frame_rgb_for_calculate)
        depth_image = copy.deepcopy(self.frame_depth_for_calculate)

        # 根据相机内参矩阵计算3D坐标
        height, width, _ = color_image.shape
        py, px = np.indices((height, width))
        
        # 提取
        z = depth_image
        # 计算相机坐标系下的三维坐标
        x = (px - self.cx) * z / self.fx
        y = (py - self.cy) * z / self.fy
        points = np.stack([x/1000, y/1000, z/1000], axis=-1)
        # 将3D坐标转换为点云对象
        points = points.reshape(-1, 3)

        # 根据深度图生成点云        
        point_cloud.points = o3d.utility.Vector3dVector(points)
        # print(points.shape)
        colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).reshape(-1,3)/255
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # 先将x坐标和y坐标转换为图像上的索引，假设x和y是整数
        index = int(y_center) * width + int(x_center)
        index_direction = int(y_direction) * width + int(x_direction)       
        center_point =  point_cloud.points[index]

        # 创建盒子上表面点云的KDTree
        panel_kdtree = o3d.geometry.KDTreeFlann(point_cloud)

        #方法一：根据体素 数量进行搜索
        # neighbor_num = 40 # 邻居个数
        # print(f"寻找距离 中心点{center_point} \n最近的{neighbor_num}个邻居，并将其绘制为蓝色")
        # k, idx, _ = panel_kdtree.search_knn_vector_3d(center_point, neighbor_num)

        #方法二：根据半径进行搜索
        # 检索半径, 单位mm
        radius = 0.005
        print(f"寻找距离 中心点{center_point} \n半径为{radius}的邻居，并将其绘制为蓝色")
        k, idx, _ = panel_kdtree.search_radius_vector_3d(center_point, radius)

        print(f"实际找到的邻居个数: {k}")

        # 法向量估计-配置参数
        ESTIMATE_NORMALS_RADIUS = 0.005 # 法向量检索半径，单位mm
        ESTIMATE_NORMALS_MAX_NN = 20   # 法向量估计最大邻居数

        # 法向量估计
        point_cloud.estimate_normals(search_param=\
                    o3d.geometry.KDTreeSearchParamHybrid(radius=ESTIMATE_NORMALS_RADIUS,
                    max_nn=ESTIMATE_NORMALS_MAX_NN))
        # 法向量重定向
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(\
                                point_cloud, camera_location=[0,0,0])
   
        # 计算质心的法向量
        # 法向量均值
        # 注：因为这些点都是被投影到一个平面上的， 因此他们的法向量是同一个值
        neighbor_normal_vector = np.asarray(point_cloud.normals)[idx]
        # print(f"neighbor_normal_vector: \n{neighbor_normal_vector}")
        center_normal_vector = np.mean(neighbor_normal_vector, axis=0)
        center_normal_vector /= np.linalg.norm(center_normal_vector) # 归一化
        center_normal_vector = center_normal_vector.reshape((3, 1))
        print(f"邻居节点法向量均值:\n {center_normal_vector}")

        #--- 法向量可视化
        # 计算向量之间的夹角
        z0 = np.float64([0, 0, 1]).reshape((3, 1))
        cos_theta = z0.T.dot(center_normal_vector)
        theta = np.arccos(cos_theta)
        print(f"cos_theta: {cos_theta} theta={np.degrees(theta)}")
        # 向量叉乘得到旋转轴
        rot_vect = np.cross(z0.reshape(-1), center_normal_vector.reshape(-1))
        rot_vect /= np.linalg.norm(rot_vect) # 归一化
        print(f"旋转向量: {rot_vect}")
        # 构造旋转矩阵
        rot_mat = cv2.Rodrigues(rot_vect*theta)[0]
        print(f"旋转矩阵:\n  {rot_mat}")
     
        z_normal = self.vector_normal(center_normal_vector.squeeze())
        #计算前进方向
        direction_point =  point_cloud.points[index_direction]
        vector_direction = direction_point - center_point
        print('z_normal',z_normal)
        print('vector_direction',vector_direction)
        y_temp = np.cross(z_normal ,  vector_direction)
        x_temp = np.cross(y_temp, z_normal)

        x_normal = self.vector_normal(x_temp)
        y_normal = self.vector_normal(y_temp)
        
        print('z_normal', z_normal)
        # 该法向量在相机下的4*4矩阵
        Trans = np.array([[x_normal[0], y_normal[0], z_normal[0], center_point[0]],
                                            [x_normal[1], y_normal[1], z_normal[1], center_point[1]],
                                            [x_normal[2], y_normal[2], z_normal[2], center_point[2]],
                                            [0,0,0,1]])
        
        # 测试的时候，发现z轴相反，需要沿着体表向内，故进行绕自身的x轴旋转180
        #绕x轴旋转gamma角度，返回旋转矩阵
        gamma = 180
        cos = np.cos(gamma *pi/180)
        sin = np.sin(gamma *pi/180)
        Rx = np.array([[1,  0,  0, 0],
                                    [0, cos, -sin, 0],
                                    [0, sin, cos, 0],
                                    [0, 0, 0, 1]])
        Trans = np.dot(Trans, Rx)
        
        print('Trans', Trans)
     
        P = np.dot(self.T_wrist2cam, Trans)
        P = np.dot(self.T_base2wrist, P)
        print('robot', P)
        print('================')
        return P, Trans

if __name__ == '__main__':
    #----- 关键点检测线程
    KeypointDetection_thread = KeypointDetectionThread()

