#!/usr/bin/env python2.7
#encoding:utf-8 
import rospy, sys
# sys.path.append('/home/uax/LiMD_example/Robot_arm')
from franka_msgs.msg import FrankaState
from math import pi
from robotiq_ft_sensor.msg import ft_sensor  
#线程
import threading
import numpy as np
import time
import time
import copy
import scipy.linalg as la
from controller_manager_msgs.srv import LoadController, UnloadController, SwitchController, ListControllers, ListControllerTypes, ReloadControllerLibraries
from tf.transformations import quaternion_from_matrix, quaternion_matrix
from get_path_intP import cartesian_pose_callback_impedance, pos_matrix_to_quat_hm
from geometry_msgs.msg import PoseStamped, Pose
from franka_msgs.msg import FrankaState, ErrorRecoveryActionGoal
# from R_16_ForceCalibration.Start_command import start_command

class MoveItFranka:
    def __init__(self) :
        # ret = start_command()
        time.sleep(2)

        #初始化ROS节点
        rospy.init_node('Franka_current_state', anonymous=True)
        

        self.current_pose = None
        self.all_state = []
        self.force_sensor = []
        
        self.listener_thread()
        self.cartesian_pose_publisher = rospy.Publisher('/cartesian_velocity_Z_impedance_LiMD/target_pose', PoseStamped, queue_size=1)
        # self.cartesian_pose_publisher = rospy.Publisher('/cartesian_pose_ZJK_controller/equilibrium_pose', PoseStamped, queue_size=10) 
        self.error_publisher = rospy.Publisher('/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=10) 
        self.recovery_error()

        #设置终端link所参考的坐标系
        self.reference_frame = 'panda_link0'  
        # 不同控制器的名字 
        self.hybrid_force_position_controller_name = 'position_force_hybird_controller'
        self.cartesian_position_controller_name = 'cartesian_pose_ZJK_controller'        
        current_controller = self.list_controllers()
        current_controller_name = current_controller.keys()
        if not self.hybrid_force_position_controller_name in current_controller_name:
            _ = self.load_controller(self.hybrid_force_position_controller_name)
            print('加载力位控制器')
        if not self.cartesian_position_controller_name in current_controller_name:          
            _ = self.load_controller(self.cartesian_position_controller_name)  #预先加载位置控制器
            print('加载位置控制器')
        #print(current_controller)
        # 更新控制器信息
        current_controller = self.list_controllers()
        current_controller_name = current_controller.keys()

        force_controll_state = current_controller[self.hybrid_force_position_controller_name]
        position_controll_state = current_controller[self.cartesian_position_controller_name]
        print('位置控制器状态：{}'.format(position_controll_state))
        if force_controll_state == 'running':
            # 确保第一个控制器是位置控制
            state = self.switch_controllers([self.cartesian_position_controller_name], [self.hybrid_force_position_controller_name])
            print('初始化切换成 位置控制')

        time.sleep(1)
        self.init_pose = copy.deepcopy(self.current_pose)
        self.init_quater =  quaternion_from_matrix(self.init_pose)
        # probe是参考系，在T的左上角
        self.T_end2probe = np.array([[1,0,0, 0],
                                                                    [0,1,0, 0],
                                                                    [0,0,1, 0.22716229],
                                                                    [0,0,0,1]])
        self.inv_end2probe = np.linalg.inv(self.T_end2probe)


    # -------------- 重置错误状态 ----
    def recovery_error(self):
        recovery_error = ErrorRecoveryActionGoal()
        self.error_publisher.publish(recovery_error)

    def load_controller(self, controller_name):
        rospy.wait_for_service('/controller_manager/load_controller')
        try:
            load_controller_service = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
            response = load_controller_service(controller_name)
            if response.ok:
                print("Controller loaded successfully")
                return True
            else:
                print("Failed to load controller")
                return False
        except rospy.ServiceException as e:
            print("Service call failed:", str(e))
            return False
        
    def unload_controller(self, controller_name):
        rospy.wait_for_service('/controller_manager/unload_controller')
        try:
            unload_controller_service = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)
            response = unload_controller_service(controller_name)
            if response.ok:
                print("Controller unloaded successfully")
                return True
            else:
                print("Failed to unload controller")
                return False
        except rospy.ServiceException as e:
            print("Service call failed:", str(e))
            return False


    def switch_controllers(self, start_controllers, stop_controllers, strictness=2,  start_asap=True, timeout=1):
        """
        start_controllers：一个包含需要启动的控制器名称的列表。这些控制器将被启动并接管机器人的控制。
        stop_controllers：一个包含需要停止的控制器名称的列表。这些控制器将被停止，并不再控制机器人。
        strictness：严格性选项，指定控制器切换的行为。它可以是以下几个值之一：
            0：宽松模式，允许在启动新控制器之前，当前控制器可能继续运行。
            1：中等模式，要求在启动新控制器之前，所有要停止的控制器必须完全停止。
            2：严格模式，要求在启动新控制器之前，所有要停止的控制器必须完全停止，并且新控制器必须立即启动。
        start_asap：一个布尔值，指示是否尽快启动新控制器。如果为 True，则新控制器将尽快启动；如果为 False，则新控制器将等待当前控制器完全停止后再启动。
        timeout：控制器切换的超时时间，以秒为单位。如果在超时时间内无法完成控制器切换操作，则会报错。

        """
        rospy.wait_for_service('/controller_manager/switch_controller')
        try:
            switch_controller_service = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
            response = switch_controller_service(start_controllers, stop_controllers, strictness,  start_asap, timeout)
            time.sleep(0.5)
            if response.ok:
                print("Switching controllers successful")
                return True
            else:
                print("Failed to switch controllers")
                return  False
        except rospy.ServiceException as e:
            print("Service call failed:", str(e))
            return False
        
    def list_controllers(self):
        """返回当前所有的控制器，及其运行状态"""
        rospy.wait_for_service('/controller_manager/list_controllers')
        controller_dict = {}
        try:
            list_controllers_service = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
            response = list_controllers_service()
            for controller in response.controller:
                print("Controller name:", controller.name)
                print("Controller state:", controller.state)
                controller_dict[controller.name] = controller.state
                # print("Controller type:", controller.type)
                # print("Hardware interface:", controller)
                # print("Claimed resources:", controller.claimed_resources)
            return controller_dict
        except rospy.ServiceException as e:
            print("Service call failed:", str(e))
            return controller_dict

    #========= 坐标变换工具 ===========
    def operator_translation(self, x=0, y=0, z=0):
        #平移
        T = np.array([[1, 0, 0, x],
                                    [0, 1, 0, y],
                                    [0, 0, 1, z],
                                    [0, 0, 0, 1]])
        return T
    def operator_rotate_x(self, gamma):
        #绕x轴旋转gamma角度，返回旋转矩阵
        cos = np.cos(gamma *pi/180)
        sin = np.sin(gamma *pi/180)
        Rx = np.array([[1,  0,  0, 0],
                                      [0, cos, -sin, 0],
                                      [0, sin, cos, 0],
                                      [0, 0, 0, 1]])
        return Rx
    def operator_rotate_y(self, beta):
        #绕y轴旋转beta角度，返回旋转矩阵
        cos = np.cos(beta *pi/180)
        sin = np.sin(beta *pi/180)
        Ry = np.array([[cos,  0,  sin, 0],
                                      [0, 1, 0, 0],
                                      [-sin, 0, cos, 0],
                                      [0, 0, 0, 1]])
        return Ry
    def operator_rotate_z(self, alpha):
        #绕z轴旋转alpha角度，返回旋转矩阵
        cos = np.cos(alpha *pi/180)
        sin = np.sin(alpha *pi/180)
        Rz = np.array([[cos,  -sin, 0, 0],
                                      [sin, cos, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        return Rz
    
    # ===== 纯位置控制的平移, 旋转
    """
            力位混合控制模式的指令考虑了位置误差偏移，但是由于较软，可能误差较大
            为了防止在力位混合和位置控制之间切换的时候，位置偏差过大，纯位置控制会出现快速移动出现危险
            同时考虑到力位混合为主要的控制模式，纯位置控制只用于旋转操作
            因此，当使用纯位置控制的旋转指令时，就不考虑之前的误差，直接读取当前的位置+姿态，并只进行姿态的变换计算
            并且把计算后的姿态存入self.init_quater, 保证切回 力位混合控制 时姿态正确
    """
    def rotate_x_by_quater_cartesian(self, angle, fix_frame=True, motion_time=4.0):
        # 存储理论值，为力位混合控制更新全局的姿态变量
        R_global = quaternion_matrix(self.init_quater)

        current_pose = copy.deepcopy(self.current_pose)
        current_pose = np.dot(current_pose, self.T_end2probe)  #探头末端坐标系

        R = self.operator_rotate_x(gamma=angle)
        if fix_frame:
            R_global_end = np.dot(R, R_global)
            R_end = np.dot(R, current_pose)  #左乘，以基底坐标系为参考（参考系不会变）
        else:
            R_global_end = np.dot(R_global, R)
            R_end = np.dot(current_pose, R)  #右乘，以自身坐标系为参考（参考系会变）
        
        # 更新全局4元数位姿
        self.init_quater = quaternion_from_matrix(R_global_end)  #力位混合用这个旋转矩阵转 4元数，避免180翻转

        R_end = np.dot(R_end, self.inv_end2probe)
        x = R_end[0][3]
        y = R_end[1][3]
        z = R_end[2][3]
        cartesian_quater = pos_matrix_to_quat_hm(R_end[:3,:3])  #纯位置用这个旋转矩阵转 4元数，避免180翻转
        self.move_to_object_Cartesian(x, y, z, cartesian_quater, motion_time=motion_time)
    
    def rotate_y_by_quater_cartesian(self, angle, fix_frame=True, motion_time=4.0):
        # 存储理论值，为力位混合控制更新全局的姿态变量
        R_global = quaternion_matrix(self.init_quater)

        current_pose = copy.deepcopy(self.current_pose)
        current_pose = np.dot(current_pose, self.T_end2probe)  #探头末端坐标系

        R = self.operator_rotate_y(angle)
        if fix_frame:
            R_global_end = np.dot(R, R_global)
            R_end = np.dot(R, current_pose)  #左乘，以基底坐标系为参考（参考系不会变）
        else:
            R_global_end = np.dot(R_global, R)
            R_end = np.dot(current_pose, R)  #右乘，以自身坐标系为参考（参考系会变）
        
        # 更新全局4元数位姿
        self.init_quater = quaternion_from_matrix(R_global_end)  #力位混合用这个旋转矩阵转 4元数，避免180翻转

        R_end = np.dot(R_end, self.inv_end2probe)
        x = R_end[0][3]
        y = R_end[1][3]
        z = R_end[2][3]
        cartesian_quater = pos_matrix_to_quat_hm(R_end[:3,:3])  #纯位置用这个旋转矩阵转 4元数，避免180翻转
        self.move_to_object_Cartesian(x, y, z, cartesian_quater, motion_time=motion_time)

    def rotate_z_by_quater_cartesian(self, angle, fix_frame=True, motion_time=4.0):
        # 存储理论值，为力位混合控制更新全局的姿态变量
        R_global = quaternion_matrix(self.init_quater)

        current_pose = copy.deepcopy(self.current_pose)
        current_pose = np.dot(current_pose, self.T_end2probe)  #探头末端坐标系

        R = self.operator_rotate_z(angle)
        if fix_frame:
            R_global_end = np.dot(R, R_global)
            R_end = np.dot(R, current_pose)  #左乘，以基底坐标系为参考（参考系不会变）
        else:
            R_global_end = np.dot(R_global, R)
            R_end = np.dot(current_pose, R)  #右乘，以自身坐标系为参考（参考系会变）
        
        # 更新全局4元数位姿
        self.init_quater = quaternion_from_matrix(R_global_end)  #力位混合用这个旋转矩阵转 4元数，避免180翻转

        R_end = np.dot(R_end, self.inv_end2probe)
        x = R_end[0][3]
        y = R_end[1][3]
        z = R_end[2][3]
        cartesian_quater = pos_matrix_to_quat_hm(R_end[:3,:3])  #纯位置用这个旋转矩阵转 4元数，避免180翻转
        self.move_to_object_Cartesian(x, y, z, cartesian_quater, motion_time=motion_time)

    def move_to_object_Cartesian(self, x, y, z, quat, motion_time):
        """纯位置控制模式
            motion_time 为执行时间, 单位秒。此部分的运动插值直接由C++控制器完成，这里只需要发布最终的位置和时间即可
        """
        pose_goal = PoseStamped()
        pose_goal .header.frame_id  = self.reference_frame  #参考坐标系
        pose_goal .header.stamp = rospy.Time(motion_time)

        pose_goal.pose.position.x = x
        pose_goal.pose.position.y = y
        pose_goal.pose.position.z = z
        pose_goal.pose.orientation.x = quat[1]
        pose_goal.pose.orientation.y = quat[2]
        pose_goal.pose.orientation.z = quat[3]
        pose_goal.pose.orientation.w = quat[0]
        print('Object pose ', pose_goal)

        #执行
        self.cartesian_pose_publisher.publish(pose_goal)
        time.sleep(motion_time+0.5)

    def listener_thread(self):
        #订阅话题
        self.Subscriber_FrankaState()
        self.Subscriber_Force()

        t1 = threading.Thread(target=self.thread_spin)  # 末端位置订阅线程
        t1.start()

    def thread_spin(self):
        rospy.spin()

    def Subscriber_FrankaState(self):
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.FrankaState_Callback)
    def FrankaState_Callback(self, msg):
        self.current_pose = np.array(msg.O_T_EE).reshape((4, 4)).T
        # print(self.current_pose)

    def Subscriber_Force(self):
        rospy.Subscriber('/robotiq_ft_sensor', ft_sensor, self.force_sensor_Callback)
    def force_sensor_Callback(self, msg):       
        self.Fx = msg.Fx 
        self.Fy = msg.Fy 
        self.Fz = msg.Fz
        self.Mx = msg.Mx
        self.My = msg.My
        self.Mz = msg.Mz

    def encode_TransMatrix2string(self, TransMatrix):
        """numpy的4*4位姿矩阵转成空格的字符串"""
        vactor = TransMatrix.T.reshape(1, -1)
        string = ''
        for i in range(vactor.shape[1]):
            string = string + str(vactor[0, i]) + ' '
        string = string.strip()
        return string

    def record_state(self):
        pose_str = self.encode_TransMatrix2string(self.current_pose)
        self.all_state.append(pose_str)
        # print(self.all_state)
        with open('corners_FrankaState_.txt', 'w') as f:
            for state in self.all_state:
                # print(type(state))
                
                f.write(str(state) [1:-1]+ '\n')

    def record_force(self):
        force_sensor_x = []
        force_sensor_y = []
        force_sensor_z = []
        force_sensor_mx = []
        force_sensor_my = []
        force_sensor_mz = []
        for _ in range(50):
            force_sensor_x.append(self.Fx)
            force_sensor_y.append(self.Fy)
            force_sensor_z.append(self.Fz)

            force_sensor_mx.append(self.Mx)
            force_sensor_my.append(self.My)
            force_sensor_mz.append(self.Mz)
            time.sleep(0.02)

        force_init_x = np.mean(force_sensor_x)
        force_init_y = np.mean(force_sensor_y)
        force_init_z = np.mean(force_sensor_z)
        force_init_mx = np.mean(force_sensor_mx)
        force_init_my = np.mean(force_sensor_my)
        force_init_mz = np.mean(force_sensor_mz)

        self.force_sensor.append([force_init_x, force_init_y, force_init_z, force_init_mx, force_init_my, force_init_mz])
        # #np.mean([0.1, 0.2, 2,3,4,5,6,7,8])

        with open('force_.txt', 'w') as f:
            for force in self.force_sensor:
                # print(type(state))
                
                
                f.write(str(force) [1:-1]+ '\n')

    
if  __name__ == "__main__":
    franka =  MoveItFranka()
    
    for i in range(8):
        franka.rotate_x_by_quater_cartesian(-10, fix_frame=False, motion_time=1)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force()
    for i in range(6):
        franka.rotate_z_by_quater_cartesian(-10, fix_frame=False, motion_time=1)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force()
    franka.rotate_z_by_quater_cartesian(60, fix_frame=False, motion_time=7)
    for i in range(6):
        franka.rotate_z_by_quater_cartesian(10, fix_frame=False, motion_time=1)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force()
    franka.rotate_z_by_quater_cartesian(-60, fix_frame=False, motion_time=7)
    franka.rotate_x_by_quater_cartesian(80, fix_frame=False, motion_time=7)

    for i in range(8):
        franka.rotate_x_by_quater_cartesian(10, fix_frame=False, motion_time=2)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force() 
    for i in range(6):
        franka.rotate_z_by_quater_cartesian(-10, fix_frame=False, motion_time=1)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force()
    franka.rotate_z_by_quater_cartesian(60, fix_frame=False, motion_time=7)
    for i in range(6):
        franka.rotate_z_by_quater_cartesian(10, fix_frame=False, motion_time=1)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force()
    franka.rotate_z_by_quater_cartesian(-60, fix_frame=False, motion_time=7)
    franka.rotate_x_by_quater_cartesian(-80, fix_frame=False, motion_time=7)




    for i in range(3):
        franka.rotate_y_by_quater_cartesian(-10, fix_frame=False, motion_time=2)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force()
    for i in range(6):
        franka.rotate_z_by_quater_cartesian(-10, fix_frame=False, motion_time=1)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force()
    franka.rotate_z_by_quater_cartesian(60, fix_frame=False, motion_time=7)
    for i in range(6):
        franka.rotate_z_by_quater_cartesian(10, fix_frame=False, motion_time=1)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force()
    franka.rotate_z_by_quater_cartesian(-60, fix_frame=False, motion_time=7)
    franka.rotate_y_by_quater_cartesian(30, fix_frame=False, motion_time=7)

    for i in range(6):
        franka.rotate_y_by_quater_cartesian(5, fix_frame=False, motion_time=2)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force() 
    for i in range(6):
        franka.rotate_z_by_quater_cartesian(-10, fix_frame=False, motion_time=1)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force()
    franka.rotate_z_by_quater_cartesian(60, fix_frame=False, motion_time=7)
    for i in range(6):
        franka.rotate_z_by_quater_cartesian(10, fix_frame=False, motion_time=1)
        time.sleep(0.1)
        franka.record_state()
        # franka.record_force()
    franka.rotate_z_by_quater_cartesian(-60, fix_frame=False, motion_time=7)
    franka.rotate_y_by_quater_cartesian(-30, fix_frame=False, motion_time=7)
      



