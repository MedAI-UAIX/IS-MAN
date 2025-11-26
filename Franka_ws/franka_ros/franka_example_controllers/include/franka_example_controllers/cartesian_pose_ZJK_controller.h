// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <array>
#include <memory>
#include <string>
#include <geometry_msgs/PoseStamped.h>
#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <Eigen/Dense>
namespace franka_example_controllers {


class CartesianPoseZJKController
    : public controller_interface::MultiInterfaceController<franka_hw::FrankaPoseCartesianInterface,
                                                            franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  // Equilibrium pose subscriber
  ros::Subscriber sub_equilibrium_pose_;

  franka_hw::FrankaPoseCartesianInterface* cartesian_pose_interface_;
  std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
  //std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  ros::Duration elapsed_time_;
  Eigen::Vector3d  start_position_,stop_position_;
  Eigen::Quaterniond start_orientation_,stop_orientation_;
  std::array<double, 16> initial_pose_{};
  std::array<double, 6> parax,paray,paraz;
  double moving_time_;
  void fifth_intP(std::array<double, 6>& parameter, double start, double end, double timelen); 

double getintP(std::array<double, 6> parameter, double time);
double Funtion_t(double x);

  void equilibriumPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
};

}  // namespace franka_example_controllers
