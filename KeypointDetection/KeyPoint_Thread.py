import sys
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
from PIL import Image as PILImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QTransform, QMovie
import copy
import scipy.linalg as la
from math import pi
import datetime



class KeypointDetectionThread(QThread): 
    keypoint_img_signal = pyqtSignal(list) 

    def __init__(self, goal_force=2, save_path='', test_mode='1'):
        super(KeypointDetectionThread, self).__init__()

        self.model_yolov8 = YOLO("/home/usai/auto_RUSS/R_03_keypoint/yolo11/yolo11m_pose_best_thy_ketpoint.pt") 
        # Predict with the model
        rgb_img_original = cv2.imread("person.png")  
        results = self.model_yolov8(rgb_img_original) 

        self.test_mode = test_mode  
        self.select_point = []  

        self.bridge = CvBridge()
        self.init_publisher() 

        self.rgb_cv_img = None 
        self.depth_cv_img = None
        self.T_base2wrist = None
        self.publish_flag = False
        self.need_caculate_normal = True


        camera_matrix = np.loadtxt("/home/usai/auto_RUSS/R_UI/keypoint_detection/cameraMatrix.txt", delimiter=",")
        self.fx = camera_matrix[0][0]
        self.fy = camera_matrix[1][1]
        self.cx = camera_matrix[0][2]
        self.cy = camera_matrix[1][2]

        R_wrist2cam = np.load('/home/usai/auto_RUSS/R_UI/keypoint_detection/R_wrist2cam.npz.npy').astype(np.float32)
        t_wrist2cam = np.load('/home/usai/auto_RUSS/R_UI/keypoint_detection/t_wrist2cam.npz.npy').T.astype(np.float32)
        self.T_wrist2cam = np.eye(4).astype(np.float32)
        self.T_wrist2cam[:3,:3] = R_wrist2cam
        self.T_wrist2cam[:3, 3] = t_wrist2cam  #*1000.0
        
    def run(self):
        while True:
            self.rate.sleep()
            while self.publish_flag:
                if len(self.select_point) != 6:
                    self.publish_flag = False
                    self.need_caculate_normal = True
                else:
                    if self.need_caculate_normal:
                        left_up, left_down, right_up, right_down, mid_up, mid_down = self.select_point
                        self.left_Trans, left_cam2object = self.keypoint_normal(left_up[0], left_up[1], left_down[0], left_down[1])
                        self.right_Trans, right_cam2object = self.keypoint_normal(right_up[0], right_up[1], right_down[0], right_down[1])
                        self.mid_Trans, mid_cam2object = self.keypoint_normal(mid_up[0], mid_up[1], mid_down[0], mid_down[1])
                        self.need_caculate_normal = False

                    self.axis_publisher()   

    def pre_view(self):
        if self.rgb_cv_img is not None:
            frame = copy.deepcopy(self.rgb_cv_img)
            frame, self.select_point = get_thy_keypoint(model=self.model_yolov8, rgb_img_original=frame, way=self.test_mode)
            self.need_caculate_normal = True
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(image.data, 
                            image.shape[1],  
                            image.shape[0],	
                            image.shape[1] * 3, 
                            QImage.Format_RGB888)
            image = QPixmap.fromImage(frame)
            image = image.scaled(QSize(328,243), 
                                aspectRatioMode=Qt.KeepAspectRatio,  
                                transformMode=Qt.SmoothTransformation)
            self.keypoint_img_signal.emit(list([image]))

        else:
            print('none')


    def publish_keypoint(self):
        self.publish_flag = True
        self.need_caculate_normal = True

    def chaneg_test_mode(self, mode:str):
        self.test_mode = mode

    def clear_keypoint(self):
        self.select_point = []
        self.publish_flag = False
        self.need_caculate_normal = True

    def init_publisher(self):
        self.left_thyroid_pub = rospy.Publisher('/left_thyroid', Float64MultiArray, queue_size=10)
        self.right_thyroid_pub = rospy.Publisher('/right_thyroid', Float64MultiArray, queue_size=10)
        self.mid_thyroid_pub = rospy.Publisher('/mid_thyroid', Float64MultiArray, queue_size=10)

        #设置循环的频率
        self.rate = rospy.Rate(10)

    def save_data(self, save_path='/home/usai/Data'):
        strftime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        rgb_img = copy.deepcopy(self.rgb_cv_img)
        depth_img = copy.deepcopy(self.depth_cv_img)
        T_base2wrist = copy.deepcopy(self.T_base2wrist)
        if not os.path.exists(os.path.join(save_path, strftime)):
            os.makedirs(os.path.join(save_path, strftime))
        cv2.imwrite(os.path.join(save_path, strftime,  'rgb.png'), rgb_img)
        np.save(os.path.join(save_path, strftime, 'depth.npy'), depth_img)
        np.save(os.path.join(save_path, strftime, 'franka.npy'), T_base2wrist)

    def axis_publisher(self):
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
        x, y, z = vector
        w = pow((pow(x, 2) + pow(y, 2) + pow(z, 2)), 0.5)
        normal = np.array([x/w, y/w, z/w])
        return normal

    def keypoint_normal(self, x_center, y_center, x_direction, y_direction):
        point_cloud = o3d.geometry.PointCloud()
        
        color_image = copy.deepcopy(self.rgb_cv_img)
        depth_image = copy.deepcopy(self.depth_cv_img).astype(float)

        height, width, _ = color_image.shape
        py, px = np.indices((height, width))
        
        # 提取
        z = depth_image
        x = (px - self.cx) * z / self.fx
        y = (py - self.cy) * z / self.fy
        points = np.stack([x/1000, y/1000, z/1000], axis=-1)
        points = points.reshape(-1, 3)
      
        point_cloud.points = o3d.utility.Vector3dVector(points)
        colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).reshape(-1,3)/255
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        index = int(y_center) * width + int(x_center)
        index_direction = int(y_direction) * width + int(x_direction)       
        center_point =  point_cloud.points[index]

        panel_kdtree = o3d.geometry.KDTreeFlann(point_cloud)

        radius = 0.005
        k, idx, _ = panel_kdtree.search_radius_vector_3d(center_point, radius)

        ESTIMATE_NORMALS_RADIUS = 0.005 
        ESTIMATE_NORMALS_MAX_NN = 20

        point_cloud.estimate_normals(search_param=\
                    o3d.geometry.KDTreeSearchParamHybrid(radius=ESTIMATE_NORMALS_RADIUS,
                    max_nn=ESTIMATE_NORMALS_MAX_NN))
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(\
                                point_cloud, camera_location=[0,0,0])
   
        neighbor_normal_vector = np.asarray(point_cloud.normals)[idx]
        # print(f"neighbor_normal_vector: \n{neighbor_normal_vector}")
        center_normal_vector = np.mean(neighbor_normal_vector, axis=0)
        center_normal_vector /= np.linalg.norm(center_normal_vector) # 归一化
        center_normal_vector = center_normal_vector.reshape((3, 1))



        z0 = np.float64([0, 0, 1]).reshape((3, 1))
        cos_theta = z0.T.dot(center_normal_vector)
        theta = np.arccos(cos_theta)
        rot_vect = np.cross(z0.reshape(-1), center_normal_vector.reshape(-1))
        rot_vect /= np.linalg.norm(rot_vect) # 归一化
        rot_mat = cv2.Rodrigues(rot_vect*theta)[0]
     
        z_normal = self.vector_normal(center_normal_vector.squeeze())
        direction_point =  point_cloud.points[index_direction]
        vector_direction = direction_point - center_point
        y_temp = np.cross(z_normal ,  vector_direction)
        x_temp = np.cross(y_temp, z_normal)

        x_normal = self.vector_normal(x_temp)
        y_normal = self.vector_normal(y_temp)
        
        Trans = np.array([[x_normal[0], y_normal[0], z_normal[0], center_point[0]],
                                            [x_normal[1], y_normal[1], z_normal[1], center_point[1]],
                                            [x_normal[2], y_normal[2], z_normal[2], center_point[2]],
                                            [0,0,0,1]])
        gamma = 180
        cos = np.cos(gamma *pi/180)
        sin = np.sin(gamma *pi/180)
        Rx = np.array([[1,  0,  0, 0],
                                    [0, cos, -sin, 0],
                                    [0, sin, cos, 0],
                                    [0, 0, 0, 1]])
        Trans = np.dot(Trans, Rx)
        
     
        P = np.dot(self.T_wrist2cam, Trans)
        P = np.dot(self.T_base2wrist, P)
        print('robot', P)
        print('================')
        return P, Trans

if __name__ == '__main__':
    KeypointDetection_thread = KeypointDetectionThread()

