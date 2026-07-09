#!/usr/bin/env python
import sys
sys.path.append('/home/usai/auto_RUSS')
from R_20_Recording_Status.config import rocord_save_path
import rospy
from std_msgs.msg import Bool
import time
import datetime
import os



class CheckPoint:
    def __init__(self, root_path=rocord_save_path) :
        """
        记录不同时间点发生的事件，便于后续根据时间点进行事件的对齐
        """
        self.save_path = root_path
        # self.check_new_path()
        # with open(os.path.join(self.last_path, 'check_point.csv'), 'w') as f:
        #     f.write('check_point,sec,nsec\n')

    def check_new_path(self):
        # 创建以检查日期的子文件夹
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        check_date_path = os.path.join(self.save_path, date_str)
        if not os.path.exists(check_date_path):
            os.makedirs(check_date_path)

        # 获取该日期已有的检查次数，以便命名新的检查结果文件夹
        last_check_num = 1
        for folder_name in os.listdir(check_date_path):
            check_folder_path = os.path.join(check_date_path, folder_name)
            if os.path.isdir(check_folder_path) and folder_name.isdigit():
                last_check_num = max(last_check_num, int(folder_name))
       
        # 最新的文件夹
        self.last_path = os.path.join(check_date_path, '{:0>3}'.format(last_check_num))
        if not os.path.exists(self.last_path):
            os.makedirs(self.last_path)

    def write_check_point(self, text):
        self.check_new_path()
        current_time = rospy.get_rostime()
        with open(os.path.join(self.last_path, 'check_point.csv'), 'a') as f:
            f.write('{},{},{}\n'.format(text, current_time.secs, current_time.nsecs))



class  ContinuousPublisher:
    def __init__(self, need_init_node=False):
        """
        用于告诉数据记录节点，是否进行数据记录
        """
        if need_init_node:
            rospy.init_node('continuous_publisher')  # 初始化ROS节点

        self.pub = rospy.Publisher('/start_recording', Bool, queue_size=10)  # 创建一个发布者，指定话题和消息类型
        self.rate = rospy.Rate(30)  # 设置发布频率为1Hz（每秒发布一次）
       
    def start_recording(self):
        for i in range(30):
            # 创建一个布尔类型的消息，设置为True
            msg = Bool()
            msg.data = True

            # 发布消息
            self.pub.publish(msg)
            self.rate.sleep()

    def stop_recording(self):
        for i in range(30):
            # 创建一个布尔类型的消息，设置为True
            msg = Bool()
            msg.data = False

            # 发布消息
            self.pub.publish(msg)
            self.rate.sleep()
    

if __name__ == '__main__':
    recorder_state = ContinuousPublisher(need_init_node=True)
    check_point = CheckPoint()

    recorder_state.start_recording()
    # time.sleep(2)
    # check_point.write_check_point('Start')
    # time.sleep(2)
    # check_point.write_check_point('目标检测')
    # # time.sleep(2)
    # # recorder_state.stop_recording()
    # time.sleep(2)
    # # recorder_state.start_recording()
    # check_point.write_check_point('到达初始点')
    # time.sleep(2)
    # # recorder_state.stop_recording()
    # # time.sleep(2)
    # # recorder_state.start_recording()
    # check_point.write_check_point('Left')
    # time.sleep(2)
    # check_point.write_check_point('End')
    # recorder_state.stop_recording()
   

        
        
