# -*- coding: utf-8 -*-
"""
@author: LMD

利用V4L2采集超声图像, 并以话题的形式发送出去, 需要US图像的节点, 直接订阅
"""

import sys
# sys.path.insert(1, '/home/uax/cv_bridge_ws/install/lib/python3/dist-packages')
# sys.path.insert(2, '/home/uax/cv_bridge_ws/devel/lib/python3/dist-packages')
import cv2
import fcntl                                                                                               
import numpy as np
import time
import rospy
# from std_msgs.msg import Float64, Float64MultiArray
# from cv_bridge import CvBridge
from cv_bridge import CvBridge, CvBridgeError
# 运行rostopic type查看对应的话题，rostopic type /depth_to_rgb/image_raw
# 返回 sensor_msgs/Image 
# 因此直接 from sensor_msgs.msg import Image，得到该话题的信息格式，用于接受该话题
from sensor_msgs.msg import Image


# 初始化节点
rospy.init_node("CaptureCard_node", anonymous=True)

#建立话题
US_pub = rospy.Publisher("/Unet/capture_us", Image, queue_size=1)
bridge = CvBridge()

# 打开视频设备
video_device = '/dev/video0'  # 你的视频设备文件路径
# width, height = 1920, 1080  #迈瑞   视频分辨率
# width, height = 1280, 1024     # 东芝  视频分辨率



# def is_image_black(frame, threshold=10):
#     """
#     判断图像是否为全黑（或接近全黑）
#     threshold: 像素值阈值，小于该值认为是黑的
#     """
#     # 计算图像的平均亮度
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     mean_brightness = cv2.mean(gray)[0]
#     # 如果平均亮度很低，认为是黑屏
#     return mean_brightness < threshold

# # --- 智能查找摄像头 ---
# found_id = -1
# for i in range(10): # 只需要查 0-9 就够了
#     print(f"正在测试摄像头 ID: {i} ...")
    
#     # 尝试打开
#     cap_test = cv2.VideoCapture(i, cv2.CAP_V4L2)
    
#     if not cap_test.isOpened():
#         print(f"  ID {i}: 无法打开")
#         continue
        
#     # 读取一帧
#     ret, frame = cap_test.read()
#     if not ret:
#         print(f"  ID {i}: 读取失败")
#         cap_test.release()
#         continue
        
#     # 检查是否全黑
#     if is_image_black(frame):
#         print(f"  ID {i}: 打开成功，但画面是黑的 (可能是笔记本摄像头或无效设备)")
#     else:
#         print(f"🎉 ID {i}: 打开成功，且画面有内容！这很可能就是你的超声探头！")
#         found_id = i
#         cap_test.release() # 释放测试用的句柄
#         break

# # --- 结果处理 ---
# if found_id == -1:
#     print(" 错误：在 0-9 范围内未找到输出有效画面的摄像头。")
#     print("    请检查：1. 超声机是否开机 2. USB 线是否插紧 3. 是否需要驱动")
#     exit()
# else:
#     # 这里的 found_id 就是你要用的 ID
#     print(f"✅ 最终选定摄像头 ID: {found_id}")
    
    # --- 下面开始正式使用这个 ID ---
cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
ret, frame = cap.read()

# cap.open(0, apiPreference=cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #：设置帧的宽度。
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)  #：设置帧的高度。
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #：设置帧的宽度。
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  #：设置帧的高度。


# 设置参数（先检查是否支持）
properties = {
    cv2.CAP_PROP_BRIGHTNESS: 40,
    cv2.CAP_PROP_CONTRAST: 50,
    cv2.CAP_PROP_HUE: 50,
    cv2.CAP_PROP_SATURATION: 50,
}

for prop, value in properties.items():
    ret = cap.set(prop, value)
    if not ret:
        print(f"警告：属性 {prop} 设置失败（设备可能不支持）")

# 验证实际值（设备可能自动调整到最接近的有效值）
print("实际亮度:", cap.get(cv2.CAP_PROP_BRIGHTNESS))
print("实际对比度:", cap.get(cv2.CAP_PROP_CONTRAST))
print("实际色调:", cap.get(cv2.CAP_PROP_HUE))
print("实际饱和度:", cap.get(cv2.CAP_PROP_SATURATION))

while True:
    start_t = time.time()
    ret, frame = cap.read()

    if not ret:
        continue

    # bgr_image = bgr_image[60:960, 400:1400,:]   #迈锐
    # bgr_image = bgr_image[60:960, 500:1400,:]   #截图桌面
    # print(bgr_image.shape)
    frame = frame[150:730,350 :1200,:]   #canon桌面  HDMI
    # bgr_image = bgr_image[90:500, 110:620,:]   #canon桌面   #上下， 
    # print(frame.shape)
    #将图像发布出去
    US_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
    
    cv2.imshow('YUYV Video', frame)
    s = cv2.waitKey(20)
    if s != -1:
        print(s)
        break

    end_t = time.time()
    # print('fps: {:.1f}'.format(1/(end_t-start_t)))
    # print(bgr_image.shape)

# 关闭视频设备和窗口
cv2.destroyAllWindows()