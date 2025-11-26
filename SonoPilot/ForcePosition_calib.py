#!/usr/bin/env python2.7
#encoding:utf-8 
# LiMD
# MedAI
"""

"""
import rospy, sys
from geometry_msgs.msg import PoseStamped, Pose
from dynamic_reconfigure.msg import Config, ConfigDescription, BoolParameter, DoubleParameter, GroupState
from franka_msgs.msg import FrankaState, ErrorRecoveryActionGoal
from dynamic_reconfigure.srv import Reconfigure, ReconfigureRequest
from std_msgs.msg import Float64, Float64MultiArray
from nav_msgs.msg import Path
from math import pi
from tf.transformations import quaternion_from_matrix, quaternion_matrix
from robotiq_ft_sensor.msg import ft_sensor  
#线程
import threading
import numpy as np
from get_path_intP import cartesian_pose_callback_impedance, pos_matrix_to_quat_hm
import time
import copy
import scipy.linalg as la
from controller_manager_msgs.srv import LoadController, UnloadController, SwitchController, ListControllers, ListControllerTypes, ReloadControllerLibraries


class MoveItFranka:
    def __init__(self, init_node=True) :
        #初始化ROS节点
        if init_node:
            rospy.init_node('Franka_position_force_control', anonymous=True)
        self.update_param(is_scaning=False)   

        self.publisher()
        self.listener_thread()
        time.sleep(2)
        self.recovery_error() 
        
        self.stop_moving = False


        self.reference_frame = 'panda_link0'  
        self.hybrid_force_position_controller_name = 'position_force_hybird_controller'
        self.cartesian_position_controller_name = 'cartesian_pose_ZJK_controller'        
        current_controller = self.list_controllers()
        current_controller_name = current_controller.keys()
        if not self.hybrid_force_position_controller_name in current_controller_name:
            _ = self.load_controller(self.hybrid_force_position_controller_name)

        if not self.cartesian_position_controller_name in current_controller_name:          
            _ = self.load_controller(self.cartesian_position_controller_name) 

        # update
        current_controller = self.list_controllers()
        current_controller_name = current_controller.keys()

        force_controll_state = current_controller[self.hybrid_force_position_controller_name]
        position_controll_state = current_controller[self.cartesian_position_controller_name]
        if force_controll_state == 'stopped' and position_controll_state == 'running':
            state = self.switch_controllers([self.hybrid_force_position_controller_name], [self.cartesian_position_controller_name])

 
        self.T_end2probe = np.array([[1,0,0, 0],
                                                                    [0,1,0, 0],
                                                                    [0,0,1, 0.22416229],
                                                                    [0,0,0,1]])
        self.inv_end2probe = np.linalg.inv(self.T_end2probe)

        #home
        self.home_pose = np.array([
            [ 7.08827868e-01, -7.05380276e-01,  1.31154195e-03,  3.06476342e-01+0.2],
            [-7.05374815e-01, -7.08828021e-01, -3.03420462e-03, 4.27093176e-04],
            [ 3.06992578e-03,  1.22560013e-03, -9.99994537e-01,  5.90601219e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
        R = self.operator_rotate_z(-45)
        self.home_pose = np.dot(self.home_pose, R)
        R = self.operator_rotate_z(-90)  
        self.home_pose = np.dot(R, self.home_pose)

        self.update_param_safety()
        self.go_home(motion_time=4, delta_time=0.001)
        

    def update_command_pose(self, pose, is_robot_end=True):    
        self.delta_x = 0
        self.delta_y = 0
        self.delta_z = 0

        if is_robot_end:
            self.init_pose = np.dot(pose, self.T_end2probe)
        else:
            self.init_pose = pose
        self.init_position_x, self.init_position_y, self.init_position_z = self.init_pose[:3, 3]
        self.init_quater =  quaternion_from_matrix(self.init_pose)  # x,y,z,w
        # print('update', self.init_position_x, self.init_position_y, self.init_position_z)


    # ========== 话题发布 ======
    def publisher(self):
        self.error_publisher = rospy.Publisher('/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=10) 
        self.pose_publisher = rospy.Publisher('/position_force_hybird_controller/equilibrium_pose', PoseStamped, queue_size=10) 
        self.cartesian_pose_publisher = rospy.Publisher('/cartesian_pose_ZJK_controller/equilibrium_pose', PoseStamped, queue_size=10) 
        self.path_publisher = rospy.Publisher('/path', Path, queue_size=10)
  
    # =========== 话题订阅 ============
    def listener_thread(self):
        self.Subscriber_FrankaState()
        t1 = threading.Thread(target=self.thread_spin)  
        t1.start()
    def thread_spin(self):
        rospy.spin()

    def Subscriber_FrankaState(self):
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.FrankaState_Callback)
    def FrankaState_Callback(self, msg):
        self.current_pose = np.array(msg.O_T_EE).reshape((4, 4)).T
    

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


    # ========= 话题发布 ===============   
    # -------------- 重置错误状态 ----
    def recovery_error(self):
        recovery_error = ErrorRecoveryActionGoal()
        self.error_publisher.publish(recovery_error)

    # -------------- 发布位置 ---------
    def move_to_object(self, x, y, z, quat):
        # print('q',quat)
        pose_goal = PoseStamped()
        pose_goal .header.frame_id  = self.reference_frame  
        pose_goal .header.stamp = rospy.Time.now() 
          
        pose_goal.pose.position.x = x
        pose_goal.pose.position.y = y
        pose_goal.pose.position.z = z
        pose_goal.pose.orientation.x = quat[0]
        pose_goal.pose.orientation.y = quat[1]
        pose_goal.pose.orientation.z = quat[2]
        pose_goal.pose.orientation.w = quat[3]

        self.pose_publisher.publish(pose_goal)

    def move_to_object_Cartesian(self, x, y, z, quat, motion_time):
        if not self.stop_moving:
            pose_goal = PoseStamped()
            pose_goal .header.frame_id  = self.reference_frame 
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
        else:
            print('MoveItFranka stop moving.')


    # --------------- 发布参数 ---------------
    def update_param(self, is_scaning=False, translational_stiffness=300., rotational_stiffness=50., ext_force=0., force_p=0., force_I=0., force_D=0.):
        config_msg = ReconfigureRequest()
        config_msg.config.bools.append(BoolParameter(name='is_scaning', value=is_scaning))
        config_msg.config.doubles.append(DoubleParameter(name= 'translational_stiffness',value= translational_stiffness))
        config_msg.config.doubles.append(DoubleParameter(name= 'rotational_stiffness', value= rotational_stiffness))
        config_msg.config.doubles.append(DoubleParameter(name= 'ext_force', value=ext_force))
        config_msg.config.doubles.append(DoubleParameter(name= 'force_p', value=force_p))
        config_msg.config.doubles.append(DoubleParameter(name= 'force_I', value= force_I))
        config_msg.config.doubles.append(DoubleParameter(name= 'force_D', value= force_D))
        config_msg.config.groups.append(GroupState(name='Default', state=True, id=0, parent=0))
        
        rospy.wait_for_service('/position_force_hybird_controller/dynamic_reconfigure_position_force_param_node/set_parameters')
        server_parameter_call = rospy.ServiceProxy('/position_force_hybird_controller/dynamic_reconfigure_position_force_param_node/set_parameters', Reconfigure)
        respone = server_parameter_call(config_msg)

    def impedance_2_hybrid(self, is_scaning, translational_stiffness, rotational_stiffness, ext_force, force_p, force_I, force_D,  safety_threshold=1.5 ):
        if ext_force<safety_threshold:
            pass
        else:
            self.update_param(is_scaning=is_scaning, translational_stiffness=translational_stiffness, rotational_stiffness=rotational_stiffness, 
                                                    ext_force=safety_threshold, 
                                                    force_p=force_p, force_I=force_I, force_D=force_D)
            time.sleep(1)
            force = safety_threshold
            while force <ext_force:
                force = force+0.5
                self.update_param(is_scaning=is_scaning, translational_stiffness=translational_stiffness, rotational_stiffness=rotational_stiffness, 
                                                    ext_force=force, 
                                                    force_p=force_p, force_I=force_I, force_D=force_D)
                time.sleep(1)
        self.update_param(is_scaning=is_scaning, translational_stiffness=translational_stiffness, rotational_stiffness=rotational_stiffness, 
                                                    ext_force=ext_force, 
                                                    force_p=force_p, force_I=force_I, force_D=force_D)
            
    def hybrid_2_impedance(self):
        current_pose = copy.deepcopy(self.current_pose)
        self.go_pose(current_pose, motion_time=2, is_robot_end=True)

    def update_param_safety(self,  is_scaning=False, translational_stiffness=300., rotational_stiffness=50., ext_force=0., force_p=1.17, force_I=0.07, force_D=5.):
        if not is_scaning:    
            self.hybrid_2_impedance()
        else:
            self.impedance_2_hybrid(is_scaning, translational_stiffness, rotational_stiffness, ext_force, force_p, force_I, force_D, safety_threshold=2.0)
        self.update_param(is_scaning, translational_stiffness, rotational_stiffness, ext_force, force_p, force_I, force_D)

    # ========== 轨迹规划+执行 =============
    def plan_cartesian_path(self, start_pose, finish_pose, motion_time, delta_time = 0.001):
        """
        motion_time =5.   # 运动时间，决定速度
        delta_time = 0.005  # 决定平滑程度
        """
        res = cartesian_pose_callback_impedance ( start_pose, finish_pose, motion_time, delta_time = delta_time)
        rate = rospy.Rate(1/delta_time)
        return res, rate
    
    # ---------------- 执行 ---------------------
    def execute(self, res, rate):
        start_t = time.time()
        for r in res:
            if self.stop_moving:
                print('MoveItFranka stop moving.')
                break
            self.move_to_object(r[0], r[1], r[2], r[3]) 
            rate.sleep()

    # --------------- 运动到指定位置 -----------------
    def go(self, finish_pose, motion_time, delta_time = 0.001):
        if not self.stop_moving:
            start_pose = copy.deepcopy(self.current_pose)
            res, rate = self.plan_cartesian_path(start_pose, finish_pose, motion_time, delta_time)
            self.show_path(res) 
            self.execute(res, rate)
        else:
            print('MoveItFranka stop moving.')
    
    def go_home(self, motion_time=6, delta_time = 0.001):
        if not self.stop_moving:
            self.go(self.home_pose, motion_time, delta_time=delta_time)
            self.update_command_pose(self.home_pose, is_robot_end=True)  
        else:
            print('MoveItFranka stop moving.')

    def go_pose(self, finish_pose, motion_time=5.0, is_robot_end =False, delta_time = 0.001):
        if not self.stop_moving:
            if is_robot_end:
                end_pose = finish_pose 
            else:
                end_pose = np.dot(finish_pose, self.inv_end2probe)
            self.go(end_pose, motion_time, delta_time)
            self.update_command_pose(finish_pose, is_robot_end=is_robot_end)
            # time.sleep(2)
        else:
            print('MoveItFranka stop moving.')
       

    def show_path(self, res, frame_id='panda_link0'):     
        # Construct a new path message
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = frame_id

        # Generate random poses for each time step
        for r in res:
            pose = PoseStamped()
            # Generate random coordinates for each pose
            pose.pose.position.x = r[0]
            pose.pose.position.y = r[1]
            pose.pose.position.z = r[2]
            pose.pose.orientation.x = r[3][0]
            pose.pose.orientation.y = r[3][1]
            pose.pose.orientation.z = r[3][2]
            pose.pose.orientation.w = r[3][3]
            # Append this pose to the path
            path_msg.poses.append(pose)

        self.path_publisher.publish(path_msg)


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
    

    #========== 运动基本单位 ==========
    # --------------- 平移 -----------
    def operator_move(self, x, y, z):
        R_start = quaternion_matrix(self.init_quater) 

        qx = self.init_position_x + self.delta_x
        qy = self.init_position_y + self.delta_y
        qz = self.init_position_z + self.delta_z
        T_arm2probe = np.array([[R_start[0][0], R_start[0][1], R_start[0][2], qx],
                                                         [R_start[1][0], R_start[1][1], R_start[1][2], qy],
                                                         [R_start[2][0], R_start[2][1], R_start[2][2], qz],
                                                         [0, 0, 0, 1]])
        T_probe2point = np.array([[1.,0.,0.,x],
                                                                [0.,1.,0.,y],
                                                                [0.,0.,1.,z],
                                                                [0.,0.,0.,1]])
        T_arm = np.dot(T_arm2probe, T_probe2point)
        x = T_arm[0][3]
        y = T_arm[1][3]
        z = T_arm[2][3]
        return x, y, z

    def translation_xyz(self, x=0, y=0, z=0, fix_frame=True,  motion_time=5.0):   
        if x>0.5 or y>0.5 or z>0.5:
            raise('Danger')    
        if fix_frame:
            self.delta_x += x  
            self.delta_y += y
            self.delta_z += z
            R_end = quaternion_matrix(self.init_quater) 
            R_end[0, 3] = self.init_position_x + self.delta_x
            R_end[1, 3] = self.init_position_y + self.delta_y
            R_end[2, 3] = self.init_position_z + self.delta_z
            R_end = np.dot(R_end, self.inv_end2probe)

            self.go(R_end, motion_time=motion_time)
        else:
            x, y, z = self.operator_move(x=x, y=y, z=z)

            self.delta_x = round(x - self.init_position_x, 4)
            self.delta_y = round(y - self.init_position_y, 4)
            self.delta_z = round(z - self.init_position_z, 4)

            R_end = quaternion_matrix(self.init_quater) 
            R_end[0, 3] = x
            R_end[1, 3] = y
            R_end[2, 3] = z
            R_end = np.dot(R_end, self.inv_end2probe)
            self.go(R_end, motion_time=motion_time)
        time.sleep(0.5)

    # --------------- 旋转 -----------------
    def rotate_x_by_quater(self, angle, fix_frame=True, motion_time=4.0):
        R_start = quaternion_matrix(self.init_quater)

        R = self.operator_rotate_x(gamma=angle)
        if fix_frame:
            R_end = np.dot(R, R_start) 
        else:
            R_end = np.dot(R_start, R) 

        self.init_quater = quaternion_from_matrix(R_end)

        R_end[0, 3] = self.init_position_x + self.delta_x
        R_end[1, 3] = self.init_position_y + self.delta_y
        R_end[2, 3] = self.init_position_z + self.delta_z
        R_end = np.dot(R_end, self.inv_end2probe)

        self.go(R_end, motion_time=motion_time)
        time.sleep(0.5)

    def rotate_y_by_quater(self, angle, fix_frame=True, motion_time=4.0):

        R_start = quaternion_matrix(self.init_quater)
        R = self.operator_rotate_y(beta=angle)
        if fix_frame:
            R_end = np.dot(R, R_start)  
        else:
            R_end = np.dot(R_start, R) 
        
        self.init_quater = quaternion_from_matrix(R_end)
        R_end[0, 3] = self.init_position_x + self.delta_x
        R_end[1, 3] = self.init_position_y + self.delta_y
        R_end[2, 3] = self.init_position_z + self.delta_z

        R_end = np.dot(R_end, self.inv_end2probe)

        self.go(R_end, motion_time=motion_time)
        time.sleep(0.5)

    def rotate_z_by_quater(self, angle, fix_frame=True, motion_time=4.0):
        R_start = quaternion_matrix(self.init_quater)
        R = self.operator_rotate_z(alpha=angle)
        if fix_frame:
            R_end = np.dot(R, R_start)  
        else:
            R_end = np.dot(R_start, R) 
        
        self.init_quater = quaternion_from_matrix(R_end)
        # print('end quater, {}'.format(self.init_quater))
        R_end[0, 3] = self.init_position_x + self.delta_x
        R_end[1, 3] = self.init_position_y + self.delta_y
        R_end[2, 3] = self.init_position_z + self.delta_z

        R_end = np.dot(R_end, self.inv_end2probe)
        # print('go', R_end)
        self.go(R_end, motion_time=motion_time)
        time.sleep(0.5)

    # ===== 纯位置控制的平移, 旋转
    def rotate_x_by_quater_cartesian(self, angle, fix_frame=True, motion_time=4.0):
        R_global = quaternion_matrix(self.init_quater)

        current_pose = copy.deepcopy(self.current_pose)
        current_pose = np.dot(current_pose, self.T_end2probe)

        R = self.operator_rotate_x(gamma=angle)
        if fix_frame:
            R_global_end = np.dot(R, R_global)
            R_end = np.dot(R, current_pose)
        else:
            R_global_end = np.dot(R_global, R)
            R_end = np.dot(current_pose, R) 
        
        self.init_quater = quaternion_from_matrix(R_global_end) 

        R_end = np.dot(R_end, self.inv_end2probe)
        x = R_end[0][3]
        y = R_end[1][3]
        z = R_end[2][3]
        cartesian_quater = pos_matrix_to_quat_hm(R_end[:3,:3]) 
        self.move_to_object_Cartesian(x, y, z, cartesian_quater, motion_time=motion_time)
    
    def rotate_y_by_quater_cartesian(self, angle, fix_frame=True, motion_time=4.0):
        R_global = quaternion_matrix(self.init_quater)

        current_pose = copy.deepcopy(self.current_pose)
        current_pose = np.dot(current_pose, self.T_end2probe) 

        R = self.operator_rotate_y(angle)
        if fix_frame:
            R_global_end = np.dot(R, R_global)
            R_end = np.dot(R, current_pose) 
        else:
            R_global_end = np.dot(R_global, R)
            R_end = np.dot(current_pose, R) 
        
        self.init_quater = quaternion_from_matrix(R_global_end) 

        R_end = np.dot(R_end, self.inv_end2probe)
        x = R_end[0][3]
        y = R_end[1][3]
        z = R_end[2][3]
        cartesian_quater = pos_matrix_to_quat_hm(R_end[:3,:3]) 
        self.move_to_object_Cartesian(x, y, z, cartesian_quater, motion_time=motion_time)

    def rotate_z_by_quater_cartesian(self, angle, fix_frame=True, motion_time=4.0):
        R_global = quaternion_matrix(self.init_quater)

        current_pose = copy.deepcopy(self.current_pose)
        current_pose = np.dot(current_pose, self.T_end2probe) 

        R = self.operator_rotate_z(angle)
        if fix_frame:
            R_global_end = np.dot(R, R_global)
            R_end = np.dot(R, current_pose) 
        else:
            R_global_end = np.dot(R_global, R)
            R_end = np.dot(current_pose, R)
        
        self.init_quater = quaternion_from_matrix(R_global_end)

        R_end = np.dot(R_end, self.inv_end2probe)
        x = R_end[0][3]
        y = R_end[1][3]
        z = R_end[2][3]
        cartesian_quater = pos_matrix_to_quat_hm(R_end[:3,:3])
        self.move_to_object_Cartesian(x, y, z, cartesian_quater, motion_time=motion_time)

    def close(self):
        rospy.signal_shutdown("Example finished.")
    

if  __name__ == "__main__":
    franka =  MoveItFranka()
    franka.go_home()


