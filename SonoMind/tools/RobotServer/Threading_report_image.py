"""
Threading_report.py
是针对扫查完成之后的文件 mp4 视频进行整体分析和报告生成

Threading_report_image.py
边扫边分析。针对扫查过程中的实时 图像进行分析
"""

import sys
sys.path.append('/home/usai/auto_RUSS/R_13_TI_RAIDS')
sys.path.append('/home/usai/auto_RUSS')
import os
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from glob import glob
import copy
import threading
import time
import cv2
# from R_13_TI_RAIDS.yolov7.yolo import YOLO
# from R_13_TI_RAIDS import objtracker
# from R_13_TI_RAIDS.report import generate_report
# from R_13_TI_RAIDS.objtracker import plot_bboxes
# from R_13_TI_RAIDS.TI_RADS.train_single import test_one_img
from ultralytics import YOLO
import numpy as np
import cv2
import pandas as pd
import os
import threading
import time
import csv
# import torch
from pathlib import Path
from PIL import Image
import pandas as pd
import logging
# 设置日志级别为 WARNING 或更高，这样就不会显示 INFO 级别的速度提示
logging.getLogger('ultralytics').setLevel(logging.WARNING)
import traceback

class TI_ARDSThread(threading.Thread):#线程类
    def __init__(self, folder_path,
                    scan_interval=0.1, 
                    model_weights="/home/usai/auto_RUSS/R_13_TI_RAIDS/yolov8/train4_tiny/weights/best.pt"
                    ):
        super(TI_ARDSThread, self).__init__()
        # self.VIDEO_PATH = '/home/uax/LiMD_example/Robot_arm/R_13_TI_RAIDS/video/WirelessUSG2023-07-12-16-19-46.mp4'
        # self.RESULT_PATH =  '/home/uax/LiMD_example/Robot_arm/R_13_TI_RAIDS/result'

        self.model_yolov8 = YOLO(model_weights)  # load an official model
        self.folder_path = folder_path
        self.csv_file = os.path.join(folder_path, 'thyroid_result.csv')
        self.scan_interval = scan_interval
        self.model_weights = model_weights
        self.stop_signal = False
        # self.thyroid_node_frame = 3  # 连续3帧都有，才认为存在结节
        self.thyroid_node_conf = 0.3  # 结节的概率阈值
        self.left_node_num = 0
        self.right_node_num = 0

        with open(os.path.join(self.csv_file), 'a') as f:
            f.write('US_img_name,sec,nsec,seg_pred_img_name,node_num,top_boxes,conf_boxes,box_in_thyroid\n')

    def run(self): #线程执行函数
        # 视频写入设置
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # original_images = cv2.imread('/home/uax/LiMD_example/Robot_arm/Dataset/Result_for_paper/2024-07-02/001/seg_original_img/1719885842_00554127693.png')
        # self.out = cv2.VideoWriter('/home/usai/Data/output_video2.mp4', fourcc, 20.0, (850, 580))

        processed_files = set()
        while not self.stop_signal:
            try:
                new_files = self.scan_folder()
                new_files = [f for f in new_files if f not in processed_files]
                print('new_file', new_files)
                if new_files:
                    for file in new_files:
                        self.detect_objects(file)
                    processed_files.update(new_files)
                    # print('释放')
                    # self.out.release()
                    # print('结束')
                time.sleep(self.scan_interval)      
            except:
                print(traceback.format_exc())
                pass
    
    def scan_folder(self):
        tmp_list = []
        for f in sorted(os.listdir(os.path.join(self.folder_path, 'seg_original_img'))):
            if f.endswith('.jpg') or f.endswith('.png'):
                tmp_list.append(f)
        return tmp_list
    
    def stop(self):
        self.stop_signal.set()

    def encode_array2string(self, array):
        """numpy矩阵转成空格的字符串"""
        # 验证输入是否为列表
        if not isinstance(array, list):
            raise ValueError("Input must be a list of numpy arrays.")

        # 将每个np.array对象转换为字符串形式
        string_list = []
        for arr in array:
            # 验证每个元素是否为np.array对象
            if  isinstance(arr, np.ndarray):
                # 将每个np.array对象转换为字符串，并添加到字符串列表中
                string_list.append(' '.join(['{}'.format(value) for value in arr]))
            else:
                string_list.append(str(arr))

        # 使用换行符连接字符串列表
        return '_'.join(string_list)


    def detect_objects(self, file):
        img_path = os.path.join(self.folder_path, 'seg_original_img', file)
        rgb_img_original = cv2.imread(img_path)
        results = self.model_yolov8(rgb_img_original)  # predict on an image

        img_name = os.path.basename(img_path)
        sec = img_name.strip('.png').split('_')[0]
        nsec = img_name.strip('.png').split('_')[1]
        match_seg_name = self.find_nearest_time(img_path=os.path.join(self.folder_path, 'seg_pred_img'),
                                                target_sec=sec, target_nsec=nsec)
        # 读取分割结果
        segmentation = cv2.imread(os.path.join(self.folder_path, 'seg_pred_img', match_seg_name), cv2.IMREAD_GRAYSCALE)
        segmentation_bgr = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR)

        org_mask_bgr = cv2.addWeighted(rgb_img_original,0.7, segmentation_bgr, 0.3, 0)

        # Process results list
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Boxes object for bounding box outputs
            conf  = result.boxes.conf.cpu().numpy()

        top_boxes = []
        conf_boxes = []
        box_in_thyroid = []  # 1表示在甲状腺内部，0表示在甲状腺外部
        for i in range(len(boxes)):
            # print('conf: {}'.format(conf[i]))
            if conf[i] > self.thyroid_node_conf :   # 置信度大于0.5
                # 这里不去预设阈值，把最原始的数据结果记录下来
                top_boxes.append(boxes[i])
                conf_boxes.append(conf[i])

                # 判断中心点是否位于目标上
                x1, y1, x2, y2 = boxes[i]
                # 计算检测框的中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                if segmentation[center_y, center_x] == 255:
                    box_in_thyroid.append(1)
                    cv2.rectangle(org_mask_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # BGR
                else:
                    box_in_thyroid.append(0)
            # cv2.rectangle(org_mask_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # BGR
        # self.out.write(org_mask_bgr)
        cv2.imshow('yolo+seg', org_mask_bgr)
        cv2.waitKey(1)

        with open(os.path.join(self.csv_file), 'a') as f:
            # print('记录')
            f.write('{},{},{},{},{},{},{},{}\n'.format(img_name, sec, nsec, match_seg_name, len(top_boxes), self.encode_array2string(top_boxes), self.encode_array2string(conf_boxes), self.encode_array2string(box_in_thyroid)))

    # 定义一个函数来查找最接近的时间点
    def find_nearest_time(self, img_path, target_sec, target_nsec):
        """根据当前照片的sec 与nsec, 查找距离其时间距离最近的图像, 返回其图像名"""

        target_sec = int(target_sec)
        target_nsec = int(target_nsec)
        candidate_image_list = glob(os.path.join(img_path, '{}_*.png'.format(target_sec)))
        if len(candidate_image_list)==0:
            # 防止找不到
            candidate_image_list = glob(os.path.join(img_path, '{}_*.png'.format(target_sec-1)))

        # Initialize variables to find the nearest time
        nearest_file_name = None
        min_diff = float('inf')

        for img in candidate_image_list:
            # Extract the nanosecond part from the filename
            img_name = os.path.basename(img)
            sec, nsec = img_name.strip('.png').split('_')
            nsec = int(nsec)

            # Calculate the absolute difference in nanoseconds
            diff = abs(target_nsec - nsec)
            
            # Update the nearest file if this one is closer
            if diff < min_diff:
                min_diff = diff
                nearest_file_name = img_name

        return nearest_file_name



if __name__ == '__main__':
    folder_path = "/home/usai/Data/For_paper/2024-10-16/002"
    # csv_file = "/home/uax/LiMD_example/Robot_arm/Dataset/Result_for_paper/2024-07-02/001/results.csv"
    scan_interval = 0.2  # 扫描间隔时间（秒）
    # model_weights = "/home/usai/auto_RUSS/R_13_TI_RAIDS/yolov8/train4_tiny/weights/best.pt"  # YOLOv5模型权重文件路径
    model_weights = "/home/usai/auto_RUSS/R_13_TI_RAIDS/yolo11/yolo11_best_TNS_detection.pt"  # YOLOv5模型权重文件路径

    thread = TI_ARDSThread(folder_path, scan_interval, model_weights)
    thread.start()

    # 停止扫描的示例：主线程在某个条件下发送停止信号给子线程
    thread.join()  # 等待子线程结束


