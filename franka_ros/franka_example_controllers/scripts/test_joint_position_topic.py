#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import time

def get_joint_state(joint_angle):
    # 创建 JointState 消息
    joint_msg = JointState()

    # 设置消息的头部（包括时间戳，用来指定插值时间）
    joint_msg .header.frame_id  = 'panda_link0'   #参考坐标系
    joint_msg .header.stamp = rospy.Time(5)

    joint_msg.name = ['panda_joint1', 
                        'panda_joint2',
                        'panda_joint3',
                        'panda_joint4',
                        'panda_joint5',
                        'panda_joint6',
                        'panda_joint7']

    # 设置目标关节角度（这里我们随机给定一些角度，实际可以根据你的需求修改）
    joint_msg.position = joint_angle
    return joint_msg

def send_joint_angles():
    # 初始化ROS节点
    rospy.init_node('joint_angle_publisher', anonymous=True)

    # 创建Publisher，发布到 joint_angle 话题
    pub = rospy.Publisher('/joint_position_example_topic_controller/joint_angle_lmd', JointState, queue_size=10)

    # 设定发布频率（1 Hz）
    rate = rospy.Rate(1)  # 1 Hz

    home_joint = [-0.0009458172778441211, -0.7855118522911229, 2.5122755325406737e-05, -2.356234683883146, 0.0015263833141605882, 1.5700401894656144, 0.7844268487662455]
    # target_joint = [-1.0103377171025745, 0.26317800103756744, 0.0005051850757810236, -2.0774567122295626, 0.6455340316373558, 1.0724318313361523, -0.5138071714546109]
    target_joint = [-0.0009458172778441211, -0.7855118522911229, 2.5122755325406737e-05, -1.956234683883146, 0.0015263833141605882, 1.5700401894656144, 0.7844268487662455]

    target_joint_2 = [-0.11577593202105473, 0.522686329395703, -0.3088637143912743, -1.4275084628842671, 0.4533333231133918, 2.5233812518148966, -0.4186975049829597]
    while not rospy.is_shutdown():
        joint_msg = get_joint_state(home_joint)
        # 发布消息
        rospy.loginfo("Publishing joint angles: %s" % joint_msg.position)
        pub.publish(joint_msg)

        print(joint_msg)
        # 按照设定的频率休眠
        time.sleep(5+0.5)


        joint_msg = get_joint_state(target_joint)
        # 发布消息
        rospy.loginfo("Publishing joint angles: %s" % joint_msg.position)
        pub.publish(joint_msg)

        print(joint_msg)
        # 按照设定的频率休眠
        time.sleep(5+0.5)


        joint_msg = get_joint_state(target_joint_2)
        # 发布消息
        rospy.loginfo("Publishing joint angles: %s" % joint_msg.position)
        pub.publish(joint_msg)

        print(joint_msg)
        # 按照设定的频率休眠
        time.sleep(5+0.5)

        # # 创建 JointState 消息
        # joint_msg = JointState()

        # # 设置消息的头部（包括时间戳，用来指定插值时间）
        # joint_msg .header.frame_id  = 'panda_link0'   #参考坐标系
        # joint_msg .header.stamp = rospy.Time(5)

        # joint_msg.name = ['panda_joint1', 
        #                   'panda_joint2',
        #                   'panda_joint3',
        #                   'panda_joint4',
        #                   'panda_joint5',
        #                   'panda_joint6',
        #                   'panda_joint1']

        # # 设置目标关节角度（这里我们随机给定一些角度，实际可以根据你的需求修改）
        # joint_msg.position = target_joint
        # # 发布消息
        # rospy.loginfo("Publishing joint angles: %s" % joint_msg.position)
        # pub.publish(joint_msg)

        # print(joint_msg)
        # # 按照设定的频率休眠
        # time.sleep(5+0.5)

if __name__ == '__main__':
    try:
        send_joint_angles()
    except rospy.ROSInterruptException:
        pass
