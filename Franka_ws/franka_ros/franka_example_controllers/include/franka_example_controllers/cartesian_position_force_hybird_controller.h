// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include<franka_example_controllers/EndForceTorque.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>

#include <franka_example_controllers/compliance_paramConfig.h>
#include<franka_example_controllers/cartesian_position_force_paramConfig.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

//笛卡尔位置控制
#include <array>
#include <franka_hw/franka_cartesian_command_interface.h>


namespace franka_example_controllers {

class cartesian_position_force_hybird_controller : 
public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                franka_hw::FrankaPoseCartesianInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;
  void stopping(const ros::Time& time) override;// 
 private:
  // Saturation
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d);  // NOLINT (readability-identifier-naming)

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  double filter_params_{0.005};
  double nullspace_stiffness_{20.0};
  double nullspace_stiffness_target_{20.0};
  const double delta_tau_max_{1.0};
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  std::mutex position_and_orientation_d_target_mutex_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;

  // Dynamic reconfigure
  std::unique_ptr<dynamic_reconfigure::Server<franka_example_controllers::cartesian_position_force_paramConfig>>
      dynamic_server_compliance_param_;
  ros::NodeHandle dynamic_reconfigure_compliance_param_node_;
  void complianceParamCallback(franka_example_controllers::compliance_paramConfig  & config,
                               uint32_t level);
  
  void PositionForceParamCallback(franka_example_controllers::cartesian_position_force_paramConfig &config,uint32_t level);
  // bool is_contact_scanning;// scanning?
  std::int8_t is_contact_scanning;
  double ext_force,force_PID_P,force_PID_I,force_PID_D;

  // Equilibrium pose subscriber
  ros::Subscriber sub_equilibrium_pose_;

  void equilibriumPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);

    //publish the end force
  ros::Publisher pub_end_force_torque;

  //笛卡尔位置控制
  franka_hw::FrankaPoseCartesianInterface* cartesian_pose_interface_;
  std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;

  std::vector<double> k_gains_;
  std::vector<double> d_gains_;
  double coriolis_factor_{1.0};
  std::array<double, 7> dq_filtered_;
  std::array<double, 16> initial_pose_;

  ros::Duration elapsed_time_;
  // Eigen::Vector3d  start_position_,stop_position_;
  // Eigen::Quaterniond start_orientation_,stop_orientation_;
  // std::array<double, 16> initial_state{};
  std::array<double, 6> parax,paray,paraz;
  double moving_time_;
  void fifth_intP(std::array<double, 6>& parameter, double start, double end, double timelen); 

double getintP(std::array<double, 6> parameter, double time);
double Funtion_t(double x);

Eigen::Matrix<double, 6, 1> desired_force_error_last,inte_force_error;
};

}  // namespace franka_example_controllers
