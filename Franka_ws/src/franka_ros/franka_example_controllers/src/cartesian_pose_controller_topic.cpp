// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/cartesian_pose_controller_topic.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <controller_interface/controller_base.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <hardware_interface/hardware_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>

namespace franka_example_controllers {

bool CartesianPoseControllerTopic::init(hardware_interface::RobotHW* robot_hardware,
                                                                                    ros::NodeHandle& node_handle) {
  cartesian_pose_interface_ = robot_hardware->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface_ == nullptr) {
    ROS_ERROR(
        "CartesianPoseControllerTopic: Could not get Cartesian Pose "
        "interface from hardware");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("CartesianPoseControllerTopic: Could not get parameter arm_id");
    return false;
  }

  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianPoseControllerTopic: Exception getting Cartesian handle: " << e.what());
    return false;
  }

  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("CartesianPoseControllerTopic: Could not get state interface from hardware");
    return false;
  }


  //初始化 话题订阅
  sub_equilibrium_pose_ = node_handle.subscribe("equilibrium_pose", 1000, &CartesianPoseControllerTopic::equilibriumPoseCallback, this,
  ros::TransportHints().reliable().tcpNoDelay());
  
  return true;
}

void CartesianPoseControllerTopic::starting(const ros::Time& /* time */) {
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_c;
  elapsed_time_ = ros::Duration(0.0);
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(initial_pose_.data()));
  start_position_=transform.translation();
  start_orientation_=transform.rotation();
  stop_position_=transform.translation();
  stop_orientation_=transform.rotation();
  moving_time_ = 1.0;
}

void CartesianPoseControllerTopic::update(const ros::Time& /* time */,
                                            const ros::Duration& period) {
  // elapsed_time_ += period;

  if (elapsed_time_.toSec()<moving_time_)
{
  elapsed_time_ += period;
  fifth_intP(parax, start_position_[0], stop_position_[0], moving_time_);
  fifth_intP(paray, start_position_[1], stop_position_[1], moving_time_);
  fifth_intP(paraz, start_position_[2], stop_position_[2], moving_time_);
  double nowx = getintP(parax, elapsed_time_.toSec());
  double nowy = getintP(paray, elapsed_time_.toSec());
  double nowz = getintP(paraz, elapsed_time_.toSec());
  Eigen::Quaterniond now_Quat;
  now_Quat = start_orientation_.slerp(Funtion_t(elapsed_time_.toSec() / moving_time_), stop_orientation_);
  Eigen::Matrix3d now_R(now_Quat);
  std::array<double, 16> new_pose;
  new_pose[0] = now_R(0, 0);          new_pose[1] = now_R(1, 0);          new_pose[2] = now_R(2, 0);          new_pose[3] = 0;
  new_pose[4] = now_R(0, 1);          new_pose[5] = now_R(1, 1);          new_pose[6] = now_R(2, 1);          new_pose[7] = 0;
  new_pose[8] = now_R(0, 2);          new_pose[9] = now_R(1, 2);          new_pose[10] = now_R(2, 2);          new_pose[11] = 0;
  new_pose[12] = nowx;          new_pose[13] = nowy;          new_pose[14] = nowz;          new_pose[15] = 1;
  cartesian_pose_handle_->setCommand(new_pose);
}
}

void CartesianPoseControllerTopic::equilibriumPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg){
  //将接收到的16矩阵转成4*4的矩阵
  // ROS_INFO("callback");
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_c;
  elapsed_time_ = ros::Duration(0.0);
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(initial_pose_.data()));
  start_position_=transform.translation();
  start_orientation_=transform.rotation();

  stop_position_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;

  stop_orientation_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  moving_time_  =msg->header.stamp.toSec();
  
  ROS_INFO("moving_time_:%lf",moving_time_);

}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianPoseControllerTopic,
                       controller_interface::ControllerBase)
