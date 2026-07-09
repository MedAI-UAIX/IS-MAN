from ultralytics import YOLO
import numpy as np
import cv2
import math
from PIL import ImageDraw, Image
  



# 返回p0到直线p1 p2的距离和直线的斜率
def get_d(p0, p1, p2):
    A = p2[1] - p1[1]
    B = (p1[0] - p2[0]) if (p1[0] - p2[0]) != 0 else 1e-5
    C = p2[0]*p1[1] - p1[0]*p2[1]
    d = abs(A*p0[0] + B*p0[1] + C) / math.sqrt(A**2+B**2)
    k = -A/B
    # print(f"A={A}  B={B}  C={C}  d={d} k={k}" )
    return d, k

def get_end_point(s0, d, k0, side):
    """
    过s0斜率为k0这条直线,垂直s0距离d的点的坐标
    :param side: =1|2 表示左右
    """
    k = -1/k0  # 垂线斜率
    theta = math.atan(k)
    dx = d*math.cos(theta)
    dy = d*math.sin(theta)
    if k < 0:
        if side == 2:
            x = s0[0] + dx
            y = s0[1] + dy
        else:
            x = s0[0] - dx
            y = s0[1] - dy
    else:
        if side == 1:
            x = s0[0] - dx
            y = s0[1] - dy
        else:
            x = s0[0] + dx
            y = s0[1] + dy
    return np.array([x,y])


def draw_points(img, keypoints, color=(0, 255, 255)):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for point in keypoints:
        draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2],fill=color,outline=color)
    return np.array(img)

def draw_points2(img, keypoints, color=(0, 255, 255)):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for k,v in keypoints.items():
        point = v
        draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2],fill=color,outline=color)
    return np.array(img)


def get_final_point(img, keypoints):
    '''喉结明显，锁骨上切凹陷明显 '''
    def point2edge(p0,p1,p2, e=1e-4):
        k = (p2[1]-p1[1])/(p2[0]-p1[0] + e)
        x = (k**2 * p1[0] + p0[0] + k*(p0[1]-p1[1]))/(k**2 + 1)
        y = (x - p1[0]) * k + p1[1]
        return (np.array([x,y]) + p0)/2

    # p1, p2 = keypoints[3], keypoints[7]#喉结，锁骨上切
    # e11, e12 = keypoints[20], keypoints[21] #颈部边缘右
    # e21, e22 = keypoints[17], keypoints[18]#颈部边缘左

    # print(keypoints)

    p1, p2 = keypoints[1], keypoints[2]#喉结，锁骨上切
    e11, e12 = keypoints[6], keypoints[8] #颈部边缘右
    e21, e22 = keypoints[3], keypoints[5]#颈部边缘左
    # 喉结——锁骨下切记上3/4处
    s0 = (3*p1 + p2)/4
    pm = (7*p1 + p2)/8
    # p3与盼p3到边缘的垂点的中点
    s1 = point2edge(pm, e11, e12)
    s2 = point2edge(pm, e21, e22)

    # 方向为锁骨两侧胸锁关节连线中点
    # zhr:没有考虑到 喉结和锁骨上切 正好在一列(20230329_Name_tengyan.png)
    # divide by zero encountered in scalar divide
    if(p2[0]-p1[0] == 0):
        k_d = 10000
    else:
        k_d = (p2[1]-p1[1])/(p2[0]-p1[0])
    
    derta = 5
    d0 = [(s0[0]*k_d+derta)/k_d, derta+s0[1]]
    d1 = [(s1[0]*k_d+derta)/k_d, derta+s1[1]]
    d2 = [(s2[0]*k_d+derta)/k_d, derta+s2[1]]
    # print(f'起始点1:{s0}, 起始点2:{s1}, 起始点3:{s2}')
    # print(f' 方向1:{d0}, 方向2:{d1}, 方向3:{d2}')
    # img = draw_points(img, [s0], color=(255,0,0))
    # img = draw_points(img, [s1], color=(0,255,0))
    # img = draw_points(img, [s2], color=(0,0,255))
    # img = draw_points(img, [d0,d1,d2], color=(255,255,255))
    final_point_1 = [s0, s1, s2]
    return img, final_point_1

def get_final_point2(img, keypoints):
    # '''喉结明显，胸锁关节（左右）明显'''
    '''喉结不明显或者喉结不存在(女性)，锁骨上切明显'''
    def point2edge(p0,p1,p2, e=1e-4):
        k = (p2[1]-p1[1])/(p2[0]-p1[0] + e)
        x = (k**2 * p1[0] + p0[0] + k*(p0[1]-p1[1]))/(k**2 + 1)
        y = (x - p1[0]) * k + p1[1]
        return (np.array([x,y]) + p0)/2
    #原2
    # p1 = keypoints[3] #喉结
    # p2, p3 = keypoints[8], keypoints[9]#胸锁关节左右
    # e11, e12 = keypoints[20], keypoints[21] #颈部边缘右
    # e21, e22 = keypoints[17], keypoints[18]#颈部边缘左
    

    p1 = keypoints[0] #下颚
    p2 = keypoints[2] #锁骨上切
    e11, e12 = keypoints[6], keypoints[8] #颈部边缘右
    e21, e22 = keypoints[3], keypoints[5]#颈部边缘左

    p3 = (p1 + p2)/2 #下颚——锁骨上切中心点
    s0 = (3*p3 + p2)/4 #中心点与锁骨上切上1/4处
    pm = (7*p3 + p2)/8  #中心点——锁骨上切中点上1/8处
    s1 = point2edge(pm, e11, e12)
    s2 = point2edge(pm, e21, e22)
    # # 喉结——胸锁关节中点上3/4处
    # p2 = (p2+p3)/2
    # s0 = (3*p1 + p2)/4
    # pm = (7*p1 + p2)/8
    # # p3与盼p3到边缘的垂点的中点
    # s1 = point2edge(pm, e11, e12)
    # s2 = point2edge(pm, e21, e22)

    # 方向为下颚——>锁骨上切连线
    if(p2[0]-p1[0] == 0):
        k_d = 10000
    else:
        k_d = (p2[1]-p1[1])/(p2[0]-p1[0])
    
    derta = 5
    d0 = [(s0[0]*k_d+derta)/k_d, derta+s0[1]]
    d1 = [(s1[0]*k_d+derta)/k_d, derta+s1[1]]
    d2 = [(s2[0]*k_d+derta)/k_d, derta+s2[1]]
    # print(f'起始点1:{s0}, 起始点2:{s1}, 起始点3:{s2}')
    # print(f' 方向1:{d0}, 方向2:{d1}, 方向3:{d2}')
    # img = draw_points(img, [s0], color=(255,0,0))
    # img = draw_points(img, [s1], color=(0,255,0))
    # img = draw_points(img, [s2], color=(0,0,255))
    # img = draw_points(img, [d0,d1,d2], color=(255,255,255))
    final_point_2 = [s0, s1, s2]
    return img, final_point_2


def get_final_point3(img, keypoints):
    '''啥都不明显，头摆正，取脖子下半部'''
    def point2edge(p0,p1,p2, e=1e-4):
        k = (p2[1]-p1[1])/(p2[0]-p1[0] + e)
        x = (k**2 * p1[0] + p0[0] + k*(p0[1]-p1[1]))/(k**2 + 1)
        y = (x - p1[0]) * k + p1[1]
        return (np.array([x,y]) + p0)/2

    e11, e12 = keypoints[6], keypoints[8] #颈部边缘右
    e21, e22 = keypoints[3], keypoints[5]#颈部边缘左
    # 左右连线中点
    p1 = (e11+e21)/2
    p2 = (e12+e22)/2
    # p2上1/4处为s1
    s0 = (3*p2 + p1)/4
    # s2s3为左右区域中心
    s1 = (p1+e12)/2
    s2 = (p1+e22)/2

    # 方向为锁骨两侧胸锁关节连线中点
    # zhr:没有考虑到 喉结和锁骨上切 正好在一列(20230329_Name_tengyan.png)
    # divide by zero encountered in scalar divide
    if(p2[0]-p1[0] == 0):
        k_d = 10000
    else:
        k_d = (p2[1]-p1[1])/(p2[0]-p1[0])
    
    derta = 5
    d0 = [(s0[0]*k_d+derta)/k_d, derta+s0[1]]
    d1 = [(s1[0]*k_d+derta)/k_d, derta+s1[1]]
    d2 = [(s2[0]*k_d+derta)/k_d, derta+s2[1]]
    # print(f'起始点1:{s0}, 起始点2:{s1}, 起始点3:{s2}')
    # print(f' 方向1:{d0}, 方向2:{d1}, 方向3:{d2}')
    # img = draw_points(img, [s0], color=(255,0,0))
    # img = draw_points(img, [s1], color=(0,255,0))
    # img = draw_points(img, [s2], color=(0,0,255))
    # img = draw_points(img, [d0,d1,d2], color=(255,255,255))
    final_point3 = [s0, s1, s2]
    return img, final_point3



# 甲状腺关键点检测
def get_thy_keypoint(model, rgb_img_original, way = '1'):
    """
    甲状腺关键点检测
    :param image:输入的图片,需为cv2的numpy数组
    :param box: 颈部框 [x1,y1,x2,y2]
    :param way: =1|2|3 表示三种方法
    :return: 关键点的的坐标
    """
    
    height, width = rgb_img_original.shape[:2]

    # Predict with the model
    # 推理的图是正向
    rgb_img = np.flipud(rgb_img_original)
    rgb_img = np.fliplr(rgb_img)

    # Predict with the model
    results = model(rgb_img)  # predict on an image

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        # img =
        # result.save(filename="/home/uax/LiMD_example/Robot_arm/R_03_keypoint/yolov8/99result.jpg")  # save to disk
        ann = results[0].plot()
        


    keypoints_list = keypoints.data.cpu().numpy()[0]
    print('shape', keypoints_list.shape)
    print('keypoints_list')
    print(keypoints_list)
    print('------------------')
    # print(keypoints_list.shape[0])
    if keypoints_list.shape[0] != 0:
        keypoints_predict = {}
        for num, point in enumerate(keypoints_list):
            # print('point', point)
            keypoints_predict[num] = np.array([point[0], point[1]])
            # print('point 2', point[2])
           
            score = int(point[2])
        print(keypoints_predict)
        if way == '1':
            img2, final_point = get_final_point(rgb_img_original, keypoints_predict)
            # cv2.imwrite("./00test/方法1.png", img2)
            # print(final_point)
            # [array([ 78.989525, 100.08432 ], dtype=float32), 
            # array([47.89653666, 88.55749274]), 
            # array([104.23434128,  88.80537231])]
        elif way =='2':
            img2, final_point = get_final_point2(rgb_img_original, keypoints_predict)
            # cv2.imwrite("./00test/方法2.png", img2)
        elif way == '3':
            img2, final_point = get_final_point3(rgb_img_original, keypoints_predict)
            # cv2.imwrite("./00test/方法3.png", img2)


        # 计算其余3个点
        p1, p2 = keypoints_predict[1], keypoints_predict[2]#喉结，锁骨上切
        # 喉结——锁骨下切记下3/4处
        s0 = (p1 + 3*p2)/4
        # img_test = draw_points(img2, [s0], color=(255,255,255))
        # cv2.imwrite("./00test/中间点.png", img_test)
        d_left, k = get_d(final_point[1], p1, p2)
        s1 = get_end_point(s0, d_left*1.2, k, 1)

        d_right, k =get_d(final_point[2], p1, p2)
        s2 = get_end_point(s0, d_right*1.2, k, 2)

        #test 
        # img3 = draw_points(img2, [s0], color=(255,255,255))
        # img3 = draw_points(img3, [s1], color=(255,255,255))
        # img3 = draw_points(img3, [s2], color=(255,255,255))
        # cv2.imwrite("./00test/最后结果.png", img3)
        # print(s0)
        # print(s1)
        # print(s2)

        # 整理结果，对应要求顺序
        #           图片为正向时，顺序如下：
        #               final_point顺序为：上方三个点：中，左，右
        #               s0, s1, s2分别为 ：下方三个点：中，左，右
        

        mid_up = int(width-final_point[0][0]), int(height-final_point[0][1])
        left_up = int(width-final_point[1][0]), int(height-final_point[1][1])
        right_up = int(width-final_point[2][0]), int(height-final_point[2][1])

        # right_up = int(width-final_point[1][0]), int(height-final_point[1][1])
        # left_up = int(width-final_point[2][0]), int(height-final_point[2][1])

        mid_down = int(width-s0[0]), int(height-s0[1])
        left_down = int(width-s1[0]), int(height-s1[1])
        right_down = int(width-s2[0]), int(height-s2[1])

        # res = np.array([final_point[2], s2, final_point[1], s1, final_point[0], s0])
        res = [left_up, left_down, right_up, right_down, mid_up, mid_down]
        # print(res)

            

        cv2.circle(rgb_img_original, mid_up, 3, (255, 0, 0), -1)  # 中上
        cv2.circle(rgb_img_original, mid_down, 3, (125, 0, 0), -1)  #中下
        cv2.circle(rgb_img_original, left_up, 3, (0, 255, 0), -1)  # 左上
        # cv2.circle(rgb_img_original, left_down, 3, (0, 125, 0), -1)  #左下
        cv2.circle(rgb_img_original, right_up, 3, (0, 0, 255), -1)  # 右上
        # cv2.circle(rgb_img_original, right_down, 3, (0, 0, 125), -1)  #右下

        # 设置线的粗细
        thickness = 4
        # 绘制箭头
        cv2.arrowedLine(rgb_img_original, mid_up, mid_down, (200, 0, 0), thickness)
        cv2.arrowedLine(rgb_img_original, left_up, left_down, (0, 200, 0), thickness)
        cv2.arrowedLine(rgb_img_original, right_up, right_down, (0, 0, 200), thickness)

        # cv2.imshow('rgb_img_original', rgb_img_original)
        cv2.imshow('ann', ann)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return rgb_img_original, res
    return rgb_img_original, None



if __name__ == '__main__':
    # Load a model
    model = YOLO("/home/uax/LiMD_example/Robot_arm/R_03_keypoint/yolov8/train3/weights/best.pt")  # load an official model
    # model = YOLO("path/to/best.pt")  # load a custom model

    rgb_img_original = cv2.imread("/home/uax/LiMD_example/Robot_arm/R_03_keypoint/yolov8/Person1_13-19-48_off_0116.png")
    res = get_thy_keypoint(model, rgb_img_original, way='1')

