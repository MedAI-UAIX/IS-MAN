// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/joint_position_example_controller_topic.h>  // 加载修改后的头文件

#include <cmath>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

namespace franka_example_controllers {

bool JointPositionExampleTopicController::init(hardware_interface::RobotHW* robot_hardware,
                                          ros::NodeHandle& node_handle) {
  position_joint_interface_ = robot_hardware->get<hardware_interface::PositionJointInterface>();
  if (position_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointPositionExampleTopicController: Error getting position joint interface from hardware!");
    return false;
  }

  // 获取关节名称
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("JointPositionExampleTopicController: Could not parse joint names");
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("JointPositionExampleTopicController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }

  // 获取每个关节的控制句柄
  position_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      position_joint_handles_[i] = position_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& e) {
      ROS_ERROR_STREAM(
          "JointPositionExampleTopicController: Exception getting joint handles: " << e.what());
      return false;
    }
  }

  // 订阅目标关节角度和插值时间话题
  sub_joint_angle_ = node_handle.subscribe("joint_angle_lmd", 20, &JointPositionExampleTopicController::jointAngleCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());


  // std::array<double, 7> q_start{{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
  // for (size_t i = 0; i < q_start.size(); i++) {
  //   if (std::abs(position_joint_handles_[i].getPosition() - q_start[i]) > 0.1) {
  //     ROS_ERROR_STREAM(
  //         "JointPositionExampleController: Robot is not in the expected starting position for "
  //         "running this example. Run `roslaunch franka_example_controllers move_to_start.launch "
  //         "robot_ip:=<robot-ip> load_gripper:=<has-attached-gripper>` first.");
  //     return false;
  //   }
  // }
  ROS_INFO("joint position init finished");
  return true;
}

void JointPositionExampleTopicController::starting(const ros::Time& /* time */) {
  for (size_t i = 0; i < 7; ++i) {
    initial_pose_[i] = position_joint_handles_[i].getPosition();
    target_pose_[i] = initial_pose_[i];  // 初始化目标位置为当前位置
  }
  elapsed_time_ = ros::Duration(0.0);  // 初始化时间
  duration_ = ros::Duration(5.0);  // 默认插值时间为5秒, 后面由话题决定
}

void JointPositionExampleTopicController::update(const ros::Time& /*time*/,
                                            const ros::Duration& period) {
  elapsed_time_ += period;
  // ROS_INFO("init finished");
  // double delta_angle = M_PI / 16 * (1 - std::cos(M_PI / 5.0 * elapsed_time_.toSec())) * 0.2;
  // for (size_t i = 0; i < 7; ++i) {
  //   if (i == 4) {
  //     position_joint_handles_[i].setCommand(initial_pose_[i] - delta_angle);
  //   } else {
  //     position_joint_handles_[i].setCommand(initial_pose_[i] + delta_angle);
  //   }
  // }

  // 计算归一化时间t，范围是[0, 1]
  double t = std::min(1.0, elapsed_time_.toSec() / duration_.toSec());

  // 使用五次多项式插值逐渐从起始位置平滑过渡到目标位置
  for (size_t i = 0; i < 7; ++i) {
    double interpolated_angle = computeCubicInterpolation(initial_pose_[i], target_pose_[i], t);
    position_joint_handles_[i].setCommand(interpolated_angle);  // 发送插值后的角度控制指令
  }
}

void JointPositionExampleTopicController::jointAngleCallback(const sensor_msgs::JointState& msg) {
  ROS_INFO("Callback triggered");
  ROS_INFO("Message received: Position size = %zu", msg.position.size());
  if (msg.position.size() == 7) {
    // 收到新的目标角度时，更新插值目标
    for (size_t i = 0; i < 7; ++i) {
      ROS_INFO("Joint %zu: %f", i, msg.position[i]);
      initial_pose_[i] = position_joint_handles_[i].getPosition();  // 当前状态作为插值起点
      target_pose_[i] = msg.position[i];  // 更新目标角度
    }
    elapsed_time_ = ros::Duration(0.0);  // 重置插值时间
    // duration_ = ros::Duration(msg.header.stamp.toSec());  // 使用消息中的插值时间
    duration_ = ros::Duration(5.0);  // 使用消息中的插值时间
    ROS_INFO("duration_=%lf", duration_);
  } else {
    ROS_WARN("Received joint state message does not contain exactly 7 positions.");
  }
}

// 五次多项式插值函数
double JointPositionExampleTopicController::computeCubicInterpolation(double start_angle, double target_angle, double t) {
  // 使用五次多项式插值公式，保证平滑性
  return start_angle + (target_angle - start_angle) * (10 * pow(t, 3) - 15 * pow(t, 4) + 6 * pow(t, 5));
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointPositionExampleTopicController,
                       controller_interface::ControllerBase)
