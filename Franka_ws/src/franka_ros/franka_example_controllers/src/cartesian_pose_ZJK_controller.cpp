// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/cartesian_pose_ZJK_controller.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <controller_interface/controller_base.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <hardware_interface/hardware_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_example_controllers {

void CartesianPoseZJKController::fifth_intP(std::array<double, 6>& parameter, double start, double end, double timelen) {
	//五次线性插值得到6个系数
    parameter[0] = start;
    parameter[1] = 0;
    parameter[2] = 0;
    parameter[3] = 10 * (end - start) / pow(timelen, 3);
    parameter[4] = -15 * (end - start) / pow(timelen, 4);
    parameter[5] = 6 * (end - start) / pow(timelen, 5);
}

double CartesianPoseZJKController::getintP(std::array<double, 6> parameter, double time) {
	//通过六个系数计算time时的值
    return parameter[0] + parameter[1] *time + parameter[2] * pow(time, 2) +parameter[3] * pow(time, 3) + parameter[4] * pow(time, 4) +
        parameter[5] * pow(time, 5);
}

double CartesianPoseZJKController::Funtion_t(double x) {
//平滑时间系数
    return -20 * pow(x, 7) + 70 * pow(x, 6) - 84 * pow(x, 5) + 35 * pow(x, 4);
}

bool CartesianPoseZJKController::init(hardware_interface::RobotHW* robot_hardware,
                                          ros::NodeHandle& node_handle) {

  sub_equilibrium_pose_ = node_handle.subscribe(
      "equilibrium_pose", 20, &CartesianPoseZJKController::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());//get pose from the topic equilibrium_pose
  
  cartesian_pose_interface_ = robot_hardware->get<franka_hw::FrankaPoseCartesianInterface>();

  if (cartesian_pose_interface_ == nullptr) {
    ROS_ERROR(
        "CartesianPoseExampleController: Could not get Cartesian Pose "
        "interface from hardware");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("CartesianPoseExampleController: Could not get parameter arm_id");
    return false;
  }

  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianPoseExampleController: Exception getting Cartesian handle: " << e.what());
    return false;
  }

  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("CartesianPoseExampleController: Could not get state interface from hardware");
    return false;
  }


  //=============测试
  // std::vector<std::string> joint_names;
  // if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
  //   ROS_ERROR(
  //       "cartesian_position_force_hybird_controller: Invalid or no joint_names parameters provided, "
  //       "aborting controller init!");
  //   return false;
  // }
  // auto* model_interface = robot_hardware->get<franka_hw::FrankaModelInterface>();
  // if (model_interface == nullptr) {
  //   ROS_ERROR_STREAM(
  //       "cartesian_position_force_hybird_controller: Error getting model interface from hardware");
  //   return false;
  // }
  // try {
  //   model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
  //       model_interface->getHandle(arm_id + "_model"));
  // } catch (hardware_interface::HardwareInterfaceException& ex) {
  //   ROS_ERROR_STREAM(
  //       "cartesian_position_force_hybird_controller: Exception getting model handle from interface: "
  //       << ex.what());
  //   return false;
  // }

  // auto* effort_joint_interface = robot_hardware->get<hardware_interface::EffortJointInterface>();
  // if (effort_joint_interface == nullptr) {
  //   ROS_ERROR_STREAM(
  //       "cartesian_position_force_hybird_controller: Error getting effort joint interface from hardware");
  //   return false;
  // }
  // for (size_t i = 0; i < 7; ++i) {
  //   try {
  //     joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
  //   } catch (const hardware_interface::HardwareInterfaceException& ex) {
  //     ROS_ERROR_STREAM(
  //         "cartesian_position_force_hybird_controller: Exception getting joint handles: " << ex.what());
  //     return false;
  //   }
  // }
  // try {
  //   auto state_handle = state_interface->getHandle(arm_id + "_robot");

  //   std::array<double, 7> q_start{{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
  //   for (size_t i = 0; i < q_start.size(); i++) {
  //     if (std::abs(state_handle.getRobotState().q_d[i] - q_start[i]) > 0.1) {
  //       ROS_ERROR_STREAM(
  //           "CartesianPoseExampleController: Robot is not in the expected starting position for "
  //           "running this example. Run `roslaunch franka_example_controllers move_to_start.launch "
  //           "robot_ip:=<robot-ip> load_gripper:=<has-attached-gripper>` first.");
  //       return false;
  //     }
  //   }
  // } catch (const hardware_interface::HardwareInterfaceException& e) {
  //   ROS_ERROR_STREAM(
  //       "CartesianPoseExampleController: Exception getting state handle: " << e.what());
  //   return false;
  // }

  return true;
}

void CartesianPoseZJKController::starting(const ros::Time& /* time */) {
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
  elapsed_time_ = ros::Duration(0.0);
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(initial_pose_.data()));
  start_position_=transform.translation();
  start_orientation_=transform.rotation();
  stop_position_=transform.translation();
  stop_orientation_=transform.rotation();

  // ROS_INFO("start_position_x:%lf",start_position_[0]);
  // ROS_INFO("start_position_y:%lf",start_position_[1]);
  // ROS_INFO("start_position_z:%lf",start_position_[2]);
  // ROS_INFO("start_position_rx:%lf",start_orientation_.coeffs()[0]);
  // ROS_INFO("start_position_ry:%lf",start_orientation_.coeffs()[1]);
  // ROS_INFO("start_position_rz:%lf",start_orientation_.coeffs()[2]);
  // ROS_INFO("start_position_rw:%lf",start_orientation_.coeffs()[3]);

  // ROS_INFO("start_position_x:%lf",stop_position_[0]);
  // ROS_INFO("start_position_y:%lf",stop_position_[1]);
  // ROS_INFO("start_position_z:%lf",stop_position_[2]);
  // ROS_INFO("start_position_rx:%lf",stop_orientation_.coeffs()[0]);
  // ROS_INFO("start_position_ry:%lf",stop_orientation_.coeffs()[1]);
  // ROS_INFO("start_position_rz:%lf",stop_orientation_.coeffs()[2]);
  // ROS_INFO("start_position_rw:%lf",stop_orientation_.coeffs()[3]);

  moving_time_ = 1.0;

  
}

void CartesianPoseZJKController::update(const ros::Time& /* time */,
                                            const ros::Duration& period) {
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
// else{

//   cartesian_pose_handle_->setCommand(initial_pose_);
// // }

}


void CartesianPoseZJKController::equilibriumPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  ROS_INFO("Callback triggered");
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
  
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

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianPoseZJKController,
                       controller_interface::ControllerBase)
