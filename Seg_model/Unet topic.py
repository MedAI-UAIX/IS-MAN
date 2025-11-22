# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:33:32 2021

@author: LMD

视频源: 采集卡-topic
"""
import sys
# sys.path.insert(1, '/home/uax/cv_bridge_ws/install/lib/python3/dist-packages')
# sys.path.insert(2, '/home/uax/cv_bridge_ws/devel/lib/python3/dist-packages')
sys.path.append('/home/usai/auto_RUSS/R_02_Unet')
sys.path.append('/home/usai/auto_RUSS/R_02_Unet/Thy_inception_seg')
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
from v2_canon_mindray.model.my_seg import Thyriod_Seg
from SwinUnet.swin_unet import SwinUnet
from Unet.unet_and_variant import U_Net


# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = U_Net(in_ch=3, out_ch=4)
net.load_state_dict(torch.load('/home/usai/auto_RUSS/R_02_Unet/Unet/Unet_best_test0.pth',  map_location=device)['state_dict'])
net.to(device=device)
net.eval()

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
from sensor_msgs.msg import Image
import copy
import threading
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import threading


frame_count = 1
time1 = 0

# 建立节点，发布分割面积
rospy.init_node("unet_node", anonymous=True)
#建立话题
area_pub = rospy.Publisher("unet_area", Float64MultiArray, queue_size=4)
original_img_pub = rospy.Publisher('/Unet/original', Image, queue_size=4) 
original_add_pred_pub = rospy.Publisher('/Unet/original_add_pred', Image, queue_size=1) 
pred_img_pub = rospy.Publisher('/Unet/pred', Image, queue_size=4) 

bridge = CvBridge()
num_img = 0

valid_transform = A.Compose([
    #A.RandomCrop(320,320,p=1.0),
    A.Resize(224, 224,p=1),
    A.Normalize(),
    ToTensorV2()
])

def color_map(prediction):
    # 定义颜色映射表
    colormap = np.asarray([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    # 将像素值映射到对应RGB色彩值
    visual_map = np.zeros(shape=(prediction.shape[0],prediction.shape[1],3),dtype=np.uint8)
    for i in range(4):
        indices = prediction == i
        visual_map[indices,:] = colormap[i]
    return visual_map


image_global = None

def us_array_callback(data):
        global image_global
        image_global = bridge.imgmsg_to_cv2(data, "bgr8")

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

            # ===== ros话题接收
            
            frame = copy.deepcopy(image_global)
            #将图像发布出去
            original_img_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))

            #frame = frame[:1450,360:1830,:]
            frame.shape
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
                area[index] = cv2.contourArea(c) 
                
            sorted_result = sorted(area.items(), key=lambda x: x[1],reverse=True) ## 
            cont_2 = copy.deepcopy(pred)
            cont_2[np.where(cont_2 != 2)] = 0
            cnts_2, hierarchy_2 = cv2.findContours(cont_2.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_2 = {}  # 用来记录每个目标的像素面积
            for index, c_2 in enumerate(cnts_2):
                # len(cnts)
                # print(cv2.contourArea(c))
                area_2[index] = cv2.contourArea(c_2)  

            sorted_result_2 = sorted(area_2.items(), key=lambda x: x[1], reverse=True) 

            if sorted_result :
                max_area_index = sorted_result[0][0]
                max_area = sorted_result[0][1]
                (x, y, w, h) = cv2.boundingRect(cnts[max_area_index]) 
                center_x = x + w // 2
                if sorted_result_2:
                    max_area_index = sorted_result[0][0]
                    max_area = sorted_result[0][1]
                    # 为了便于可视化，将最大的区域绘制出来
                    (x, y, w, h) = cv2.boundingRect(cnts[max_area_index]) 
                    max_area_index_2 = sorted_result_2[0][0]
                    max_area_2 = sorted_result_2[0][1]
                    # 为了便于可视化，将最大的区域绘制出来
                    (x2, y2, w2, h2) = cv2.boundingRect(cnts_2[max_area_index_2]) 
                else:
                    x=0
                    w=0
                    x2=0
                    y2=0
                    w2=0
                    h2=0
                    max_area=0
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

            send_area.data = [x, w, x2, w2, 35, 820,max_area]  
            area_pub.publish(send_area)

            #将分割图像发布出去
            pred_gray = cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2BGR)
            pred_img_pub.publish(bridge.cv2_to_imgmsg(pred_gray, "bgr8"))  
           

            image = cv2.addWeighted(frame,0.7, image_array, 0.3, 0).astype('uint8')
            original_add_pred_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))

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


