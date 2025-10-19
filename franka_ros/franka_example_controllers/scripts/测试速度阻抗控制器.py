#!/usr/bin/env python

import rospy
import time
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_matrix
import numpy as np

def create_pose_msg(position, rotation_matrix):
    """构造 PoseStamped 消息"""
    msg = PoseStamped()
    msg.header.stamp = 5.0
    msg.header.frame_id = "panda_link0"

    msg.pose.position.x = position[0]
    msg.pose.position.y = position[1]
    msg.pose.position.z = position[2]

    quat = quaternion_from_matrix(np.vstack((np.hstack((rotation_matrix, [[0], [0], [0]])), [0, 0, 0, 1])))
    msg.pose.orientation.x = 0.916983
    msg.pose.orientation.y = -0.156005
    msg.pose.orientation.z = 0.0374482
    msg.pose.orientation.w = -0.365242

    return msg

def main():
    rospy.init_node("test_impedance_target_pose_sender")
    # pub = rospy.Publisher('/cartesian_velocity_Z_impedance_LiMD/target_pose', PoseStamped, queue_size=1)
    pub = rospy.Publisher('/cartesian_pose_ZJK_controller/equilibrium_pose', PoseStamped, queue_size=1)
    rospy.sleep(1.0)

    # 固定的初始朝向（可来自 rviz 或仿真）
    q_init = [0.916983, -0.156005, 0.0374482, -0.365242]  # [w, x, y, z]
    rot = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  1.0,  0.0],
        [0.0,  0.0,  1.0]
    ])

    # 位姿列表（与之前测试用位置一致）
    poses = [
        np.array([0.4,  0.0, 0.4]),
        np.array([0.4,  0.1, 0.5]),
        np.array([0.4, -0.1, 0.55]),
    ]

    rate = rospy.Rate(0.1)  # 每 10 秒一个
    while not rospy.is_shutdown():
        for pos in poses:
            msg = create_pose_msg(pos, rot)
            pub.publish(msg)
            rospy.loginfo(f"[Python] Sent pose to: {pos}")
            time.sleep(8)  # 等待控制器插值完毕

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
