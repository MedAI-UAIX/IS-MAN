#pragma once

#include <array>
#include <memory>
#include <string>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

namespace franka_example_controllers {

class CartesianPoseZJKController : public controller_interface::MultiInterfaceController<
                                        franka_hw::FrankaVelocityCartesianInterface,
                                        franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& nh) override;
  bool initRequest(hardware_interface::RobotHW* robot_hw,
                   ros::NodeHandle& root_nh,
                   ros::NodeHandle& controller_nh,
                   controller_interface::ControllerBase::ClaimedResources& claimed_resources) override;
  void update(const ros::Time& time, const ros::Duration& period) override;
  void starting(const ros::Time& time) override;
  void stopping(const ros::Time& time) override;

 private:
  franka_hw::FrankaVelocityCartesianInterface* velocity_cartesian_interface_ = nullptr;
  std::unique_ptr<franka_hw::FrankaCartesianVelocityHandle> cartesian_velocity_handle_;

  double velocity_z_ = 0.0;
  double velocity_z_prev_ = 0.0;
  double acc_filtered_ = 0.0;
  double position_z_relative_ = 0.0;

  double Md_ = 5.0;
  double Bd_ = 40.0;
  double Kd_ = 300.0;
  double fz_target_ = 1.0;

  double acc_filter_alpha_ = 0.2;
  double vel_filter_alpha_ = 0.2;

  double getMeasuredFz();
};

}  // namespace franka_example_controllers
