#!/usr/bin/env python2.7
#encoding:utf-8 
import numpy as np
from tf.transformations import quaternion_slerp, quaternion_from_matrix, quaternion_matrix, rotation_matrix
import rospy
from math import acos, sin, cos

def fifth_intP( parameter, start,  end,  timelen):
	parameter[0] = start
	parameter[1] = 0
	parameter[2] = 0
	parameter[3] = 10 * (end - start) / pow(timelen, 3)
	parameter[4] = -15 * (end - start) / pow(timelen, 4)
	parameter[5] = 6 * (end - start) / pow(timelen, 5)
	return parameter

def getintP( parameter,  time):

	return parameter[0] + parameter[3] * pow(time, 3) + parameter[4] * pow(time, 4) + parameter[5] * pow(time, 5)

def Funtion_t( x):
	return -20 * pow(x, 7) + 70 * pow(x, 6) - 84 * pow(x, 5) + 35 * pow(x, 4)

def quat_to_pos_matrix_hm(quat):
        w,x,y,z = quat
        # 四元数  转  旋转矩阵
        # 创建位姿矩阵，写入位置
        R = np.zeros((3,3))
        R[0, 0] = 1 - 2 * pow(y, 2) - 2 * pow(z, 2)
        R[0, 1] = 2 * (x * y - w * z)
        R[0, 2] = 2 * (x * z + w * y)

        R[1, 0] = 2 * (x * y + w * z)
        R[1, 1] = 1 - 2 * pow(x, 2) - 2 * pow(z, 2)
        R[1, 2] = 2 * (y * z - w * x)

        R[2, 0] = 2 * (x * z - w * y)
        R[2, 1] = 2 * (y * z + w * x)
        R[2, 2] = 1 - 2 * pow(x, 2) - 2 * pow(y, 2)
        return R
# def pos_matrix_to_quat_hm(R):
    #出现部分情况为inf
#         # 旋转矩阵  转  四元数
#         r11 = R[0, 0]
#         r12 = R[0, 1]
#         r13 = R[0, 2]
#         r21 = R[1, 0]
#         r22 = R[1, 1]
#         r23 = R[1, 2]
#         r31 = R[2, 0]
#         r32 = R[2, 1]
#         r33 = R[2, 2]
#         w = 0.5 * pow((1 + r11 + r22 + r33), 0.5)
#         x = (r32 - r23) / (4 * w)
#         y = (r13 - r31) / (4 * w)
#         z = (r21 - r12) / (4 * w)
#         return (w, x, y, z)

# import numpy as np

def pos_matrix_to_quat_hm(R):
    # 旋转矩阵转换为四元数
    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]
    
    trace = r11 + r22 + r33
    if trace > 0:
        s = 0.5 / np.sqrt(trace+1.0)
        w = 0.25 / s
        x = (r32 - r23) * s
        y = (r13 - r31) * s
        z = (r21 - r12) * s
    else:
        if r11 > r22 and r11 > r33:
            s = 2.0 * np.sqrt(1.0 + r11 - r22 - r33)
            w = (r32 - r23) / s
            x = 0.25 * s
            y = (r12 + r21) / s
            z = (r13 + r31) / s
        elif r22 > r33:
            s = 2.0 * np.sqrt(1.0 + r22 -r11 - r33)
            w = (r13 - r31) / s
            x = (r12 + r21) / s
            y = 0.25 * s
            z = (r23 + r32) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r33 - r11 -r22)
            w = (r21 - r12) / s
            x = (r13 + r31) / s
            y = (r23 + r32) / s
            z = 0.25 * s
            
    return (w, x, y, z)


# home_pose = np.array([[1, 0.000438773, -0.000247279, 0.306846],
#                                                                   [0.000438702, -1, -0.000285702, 0.0000519128],
#                                                                   [-0.000247404, 0.000285593, -1, 0.590191],
#                                                                   [0, 0, 0, 1]])
# (w, x, y, z) = pos_matrix_to_quat_hm(home_pose[:3,:3])

def slerp(q1, q2, t):
    # print(q1, q2)
    dot = q1[3]*q2[3] + q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2]
    
    # 如果点乘小于0，则反转旋转轴，避免路径上的奇点问题
    if dot < 0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot
    
    # dot的取值区间只能在[-1, 1]之间，由于精度问题，可能出现1.0000000048的情况，进行判断避免
    if dot >1:
          dot = 1.0
    # print('dot', dot)
    theta_0 = acos(dot)  # 夹角
    sin_theta = sin(theta_0)
    
    # 当夹角过小时直接线性插值
    if sin_theta == 0:
        w = q1[0]*(1-t) + q2[0]*t
        x =  q1[1]*(1-t) + q2[1]*t
        y = q1[2]*(1-t) + q2[2]*t
        z = q1[3]*(1-t) + q2[3]*t
        return (w,x,y,z)
    
    # 计算两个四元数的插值比例
    ratio_a = sin((1-t)*theta_0)/sin_theta
    ratio_b = sin(t*theta_0)/sin_theta
    
    # 插值操作
    w = q1[0]*ratio_a + q2[0]*ratio_b
    x = q1[1]*ratio_a + q2[1]*ratio_b
    y = q1[2]*ratio_a + q2[2]*ratio_b
    z = q1[3]*ratio_a + q2[3]*ratio_b
    return (w,x,y,z)


def cartesian_pose_callback_impedance ( initial_pose, end_Pose, motion_time, delta_time = 0.005):
    new_pose_list = []
    parax = np.zeros(6)
    paray = np.zeros(6)
    paraz = np.zeros(6)
    time = 0.0
    period = rospy.Duration(delta_time)

    while time < motion_time:
        if (time == 0.0):
            # inital_R = initial_pose[:3,:3]                                                      
            # intial_Quat = pos_matrix_to_quat_hm( inital_R)
            intial_Quat = quaternion_from_matrix(initial_pose)  #4*4矩阵转 四元数xyzw，不受xyz的影响

            parax = fifth_intP(parax, initial_pose[0,3], end_Pose[0, 3], motion_time)
            paray = fifth_intP(paray, initial_pose[1,3], end_Pose[1, 3], motion_time)
            paraz = fifth_intP(paraz, initial_pose[2,3], end_Pose[2, 3], motion_time)
                
        time += period.to_sec()
        nowx = getintP(parax, time)
        nowy = getintP(paray, time)
        nowz = getintP(paraz, time)

        # final_R = end_Pose[:3,:3]
        # final_Quat = pos_matrix_to_quat_hm(final_R) 
        final_Quat = quaternion_from_matrix(end_Pose)  #4*4矩阵转 四元数xyzw，不受xyz的影响

        now_Quat = quaternion_slerp(intial_Quat, final_Quat, Funtion_t(time / motion_time))
        # now_R = quat_to_pos_matrix_hm(now_Quat)

        # new_pose = np.zeros_like(  initial_pose)
        # new_pose[:3,:3] = now_R
        # new_pose[0,3] = nowx
        # new_pose[1,3] = nowy
        # new_pose[2,3] = nowz
        # new_pose[3,3] = 1
        new_pose_list.append([nowx, nowy, nowz, now_Quat])

    return new_pose_list


if __name__ == '__main__':
    import numpy as np
    from tf.transformations import quaternion_from_matrix, rotation_matrix

    # 定义一个旋转矩阵
    rot_matrix = np.array([[ 0.99298467, -0.09729164,  0.06719953,  0.       ],
                                                    [ 0.0994502 ,  0.99460359, -0.02955246,  0.      ],
                                                    [-0.06396169,  0.03602815,  0.9973018 ,  0.      ],
                                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # 将旋转矩阵转成四元数形式
    quat = quaternion_from_matrix(rot_matrix)

    inital_R = rot_matrix[:3,:3]                                                      
    intial_Quat = pos_matrix_to_quat_hm( inital_R)
    # parameter = np.zeros(6)
    # start_pose = np.array([
	#      [ 9.51229269e-01  ,2.03446422e-01 , 2.31840140e-01 , 4.47586345e-01],
    #     [-3.09385863e-02, -6.84730224e-01,  7.28128412e-01, -2.42169184e-01],
    #     [ 3.06892183e-01 ,-6.99835514e-01, -6.45048635e-01  ,2.19027132e-01],
    #     [ 0.00000000e+00  ,0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]
    # ])
    # end_pose = np.array(
	#      [[ 9.55525000e-01,  1.91365000e-01 , 2.24390000e-01,  4.56888000e-01],
    #     [-2.67380000e-02 ,-7.01527000e-01 , 7.12141000e-01 ,-2.40716000e-01],
    #     [ 2.93695000e-01 ,-6.86468000e-01, -6.65210000e-01 , 2.18517000e-01],
    #     [ 0.00000000e+00,  0.00000000e+00, 0.00000000e+00 , 1.00000000e+00]]
    # )
    # timelen = 5.

    # res = cartesian_pose_callback_impedance ( start_pose, end_pose, timelen, delta_time = 0.5)
    # print(len(res))
    # print(res)
    x = quat_to_pos_matrix_hm()