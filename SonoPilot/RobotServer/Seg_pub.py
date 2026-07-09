# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:33:32 2021

@author: LMD

用途: phantom
训练数据: 掌超phantom
模型:SwinUnet_Unet
标签:2分类( 0背景，1甲状腺) 
视频源: 采集卡-topic
"""
import sys
# sys.path.insert(1, '/home/uax/cv_bridge_ws/install/lib/python3/dist-packages')
# sys.path.insert(2, '/home/uax/cv_bridge_ws/devel/lib/python3/dist-packages')
sys.path.append('..') # 添加上一级目录
# sys.path.append('/home/usai/auto_RUSS/R_02_Unet/Thy_inception_seg')
import torch.nn as nn
import torch
torch.cuda.is_available()
from torchsummary import summary
import torch.nn.functional as F
import os
import torchvision.transforms.functional as ff
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
# from Thy_inception_seg.model.my_seg import Liver_full_Seg
# from v2_canon_mindray.model.my_seg import Thyriod_Seg
# from SwinUnet.swin_unet import SwinUnet
# from Unet.unet_and_variant import U_Net
from Segmentation.model.UNet_with_ReparamBlock import U_Net
##-------基于视频，根据Unet分割的面积，挑选最大切面的帧    
import glob
import numpy as np
import torch
import os
import cv2 
# import SimpleITK as sitk
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import rospy
from std_msgs.msg import Float64, Float64MultiArray
import cv2
import imagezmq
import traceback
import time
import simplejpeg
import copy
from cv_bridge import CvBridge, CvBridgeError
# 运行rostopic type查看对应的话题，rostopic type /depth_to_rgb/image_raw
# 返回 sensor_msgs/Image 
# 因此直接 from sensor_msgs.msg import Image，得到该话题的信息格式，用于接受该话题
from sensor_msgs.msg import Image
import copy
import threading
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import threading


# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------
def load_model(checkpoint_path, device="cuda"):
    """Load trained UNet model from checkpoint."""
    model = U_Net(out_ch=5)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model

def color_map(prediction):
    # 定义颜色映射表
    colormap = np.asarray([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0,255,255]], dtype=np.uint8)
    # 将像素值映射到对应RGB色彩值
    visual_map = np.zeros(shape=(prediction.shape[0],prediction.shape[1],3),dtype=np.uint8)
    for i in range(4):
        indices = prediction == i
        visual_map[indices,:] = colormap[i]
    return visual_map


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = load_model('/home/usai/ISMANtest/20260521/IS-MAN/SonoPilot/Segmentation/checkpoint/best.pth', device)



frame_count = 1
time1 = 0

# 建立节点，发布分割面积
# #原文链接：https://blog.csdn.net/weixin_43956732/article/details/105951288
rospy.init_node("unet_node", anonymous=True)
#建立话题
area_pub = rospy.Publisher("unet_area", Float64MultiArray, queue_size=4)
original_img_pub = rospy.Publisher('/Unet/original', Image, queue_size=4) 
original_add_pred_pub = rospy.Publisher('/Unet/original_add_pred', Image, queue_size=1) 
pred_img_pub = rospy.Publisher('/Unet/pred', Image, queue_size=4)
thyroid_trachea_pub = rospy.Publisher('/Unet/thyroid_trachea_gray', Image, queue_size=4)


bridge = CvBridge()
num_img = 0

valid_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])



image_global = None

def us_array_callback(data):
        global image_global
        image_global = bridge.imgmsg_to_cv2(data, "bgr8")
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

rospy.Subscriber('/Unet/capture_us', Image, us_array_callback)

# 定义一个线程执行的函数
def thread_function(name):
    rospy.spin()

# 创建线程对象
thread = threading.Thread(target=thread_function, args=("Thread1", ))

# 启动线程
thread.start()


while True:
    if image_global is not None:
        try:
            num_img = num_img + 1
            time1 = time.time() #if frame_count == 1 else time1


            frame = copy.deepcopy(image_global)
            original_img_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))


            w,h,c = frame.shape
            # print( (int(w),int(h)))

            frame_tensor = valid_transform(image=frame)['image'].unsqueeze(0)

            frame_tensor = frame_tensor.to(device=device, dtype=torch.float32)
            frame_tensor.shape
            # 预测
            pred = net(frame_tensor)
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            pred = pred.squeeze().astype('uint8')
            pred = copy.deepcopy(cv2.resize(pred, (h,w), interpolation=cv2.INTER_NEAREST))

            #0背景，1甲状腺，2气管，3颈动脉
            pred_gray = copy.deepcopy(pred)
            pred_gray[np.where(pred_gray != 1)] = 0
            pred_gray[pred==1] = 255

            #mask 改成3通道，便于后面可视化 
            image_array = color_map(pred)  
            image_array.shape
            
            #寻找角点与轮廓
            cont = copy.deepcopy(pred)
            cont[np.where(cont!=1)] =0
            # print(np.where(cont==2))
            cnts, hierarchy = cv2.findContours(cont.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            # len(cnts)
            area = {}  #用来记录每个目标的像素面积
            for index, c in enumerate(cnts):
                # len(cnts)
                # print(cv2.contourArea(c))
                area[index] = cv2.contourArea(c) #cv2.contourArea(c) 计算的是不包含轮廓边界的一圈的像素点
                
            sorted_result = sorted(area.items(), key=lambda x: x[1],reverse=True) ## 按照字典的值，从大到小，进行排序

            #气管
            cont_2 = copy.deepcopy(pred)
            cont_2[np.where(cont_2 != 2)] = 0
            cnts_2, hierarchy_2 = cv2.findContours(cont_2.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_2 = {}  # 用来记录每个目标的像素面积
            for index, c_2 in enumerate(cnts_2):
                # len(cnts)
                # print(cv2.contourArea(c))
                area_2[index] = cv2.contourArea(c_2)  # cv2.contourArea(c) 计算的是不包含轮廓边界的一圈的像素点

            sorted_result_2 = sorted(area_2.items(), key=lambda x: x[1], reverse=True)  ## 按照字典的值，从大到小，进行排序
        
            if sorted_result :
                max_area_index = sorted_result[0][0]
                max_area = sorted_result[0][1]
                (x, y, w, h) = cv2.boundingRect(cnts[max_area_index])  # 寻找最小矩形
                center_x = x + w // 2
                if sorted_result_2:
                    max_area_index = sorted_result[0][0]
                    max_area = sorted_result[0][1]
                    # 为了便于可视化，将最大的区域绘制出来
                    (x, y, w, h) = cv2.boundingRect(cnts[max_area_index])  # 寻找最小矩形
                    max_area_index_2 = sorted_result_2[0][0]
                    max_area_2 = sorted_result_2[0][1]
                    # 为了便于可视化，将最大的区域绘制出来
                    (x2, y2, w2, h2) = cv2.boundingRect(cnts_2[max_area_index_2])  # 寻找最小矩形
                else:
                    x=0
                    w=0
                    x2=0
                    y2=0
                    w2=0
                    h2=0
                    max_area=-2   # 特殊情况：没有气管，但是却有甲状腺，返回一个特殊值。那就让其一直前进扫查
            else:
                x=0
                w=0
                x2=0
                y2=0
                w2=0
                h2=0
                max_area=0
         
            #将面积发布出去
            send_area = Float64MultiArray()


            send_area.data = [x, w, x2, w2, 35, 820, max_area]   #  canno 高清
            area_pub.publish(send_area)

            #将分割图像发布出去
            pred_gray = cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2BGR)
            pred_img_pub.publish(bridge.cv2_to_imgmsg(pred_gray, "bgr8"))   #发送黑白的就行

                        # ==================== 新增：生成甲状腺气管灰度图并发布 ====================
            # 创建灰度图像：背景0，甲状腺150，气管50
            thyroid_trachea_gray = np.zeros_like(pred, dtype=np.uint8)
            thyroid_trachea_gray[pred == 1] = 150  # 甲状腺赋值150
            thyroid_trachea_gray[pred == 2] = 50   # 气管赋值50
            # 发布单通道灰度图
            thyroid_trachea_pub.publish(bridge.cv2_to_imgmsg(thyroid_trachea_gray, "mono8"))


           
            image = cv2.addWeighted(frame,0.7, image_array, 0.3, 0).astype('uint8')
            original_add_pred_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))
            # # print(image)
            # cv2.imshow('frame1', cv2.resize(image, (int(image.shape[0]//2),int(image.shape[1]//2)), interpolation=cv2.INTER_CUBIC))
            # cv2.imshow('unet', cv2.resize(image_array, (int(image.shape[0]//2),int(image.shape[1]//2)), interpolation=cv2.INTER_CUBIC))
        
            # # #原尺寸
            # cv2.imshow('unet', image_array)
            cv2.imshow('origanl', image)
            

            time2 = time.time()
            print(device, int(1/(time2-time1)))
            time.sleep(0.01)
            # frame_count += 1

            
            if  cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()    
                # image_hub.close()
                break
        except:
            print(traceback.format_exc())
            print('==1===')
            # break


