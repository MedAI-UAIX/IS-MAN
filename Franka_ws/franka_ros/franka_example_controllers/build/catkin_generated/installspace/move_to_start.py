#!/usr/bin/env python2

import rospy
import sys
sys.path.insert(0,'/opt/ros/melodic/lib/python2.7/dist-packages')
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray

if __name__ == '__main__':
    rospy.init_node('move_to_start')
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    commander = MoveGroupCommander('panda_arm')
    commander.set_named_target('ready')
    commander.go()
