import rospy
from controller_manager_msgs.srv import LoadController, UnloadController, SwitchController, ListControllers, ListControllerTypes, ReloadControllerLibraries


def load_controller(controller_name):
    rospy.wait_for_service('/controller_manager/load_controller')
    try:
        load_controller_service = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
        response = load_controller_service(controller_name)
        if response.ok:
            print("Controller loaded successfully")
        else:
            print("Failed to load controller")
    except rospy.ServiceException as e:
        print("Service call failed:", str(e))


def unload_controller(controller_name):
    rospy.wait_for_service('/controller_manager/unload_controller')
    try:
        unload_controller_service = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)
        response = unload_controller_service(controller_name)
        if response.ok:
            print("Controller unloaded successfully")
        else:
            print("Failed to unload controller")
    except rospy.ServiceException as e:
        print("Service call failed:", str(e))


def switch_controllers(start_controllers, stop_controllers, strictness,  start_asap, timeout):
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
        if response.ok:
            print("Switching controllers successful")
        else:
            print("Failed to switch controllers")
    except rospy.ServiceException as e:
        print("Service call failed:", str(e))


def list_controllers():
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


def list_controller_types():
    rospy.wait_for_service('/controller_manager/list_controller_types')
    try:
        list_controller_types_service = rospy.ServiceProxy('/controller_manager/list_controller_types', ListControllerTypes)
        response = list_controller_types_service()
        for controller_type in response.types:
            print("Controller type:", controller_type)
    except rospy.ServiceException as e:
        print("Service call failed:", str(e))


def reload_controller_libraries():
    rospy.wait_for_service('/controller_manager/reload_controller_libraries')
    try:
        reload_controller_libraries_service = rospy.ServiceProxy('/controller_manager/reload_controller_libraries', ReloadControllerLibraries)
        response = reload_controller_libraries_service()
        if response.ok:
            print("Controller libraries reloaded successfully")
        else:
            print("Failed to reload controller libraries")
    except rospy.ServiceException as e:
        print("Service call failed:", str(e))


if __name__ == '__main__':
    rospy.init_node('controller_manager_client')
    
    # 调用示例
    # position_force_hybird_controller = 'position_force_hybird_controller'
    # joint_impedance_controller = 'joint_impedance_example_controller'
    controller_dice = list_controllers()
    'franka_state_controler' in controller_dice.keys()
    controller_dice['']

    # load_controller('position_force_hybird_controller')
    # # load_controller('joint_impedance_example_controller')
    # load_controller('cartesian_pose_ZJK_controller')

    # # unload_controller('position_force_hybird_controller')
    # switch_controllers(['position_force_hybird_controller'], ['cartesian_pose_ZJK_controller'], 3, True, 1)
    # switch_controllers(['cartesian_pose_ZJK_controller'], ['position_force_hybird_controller'], 3, True, 1)
    # # list_controllers()
    # # list_controller_types()
    # reload_controller_libraries()
