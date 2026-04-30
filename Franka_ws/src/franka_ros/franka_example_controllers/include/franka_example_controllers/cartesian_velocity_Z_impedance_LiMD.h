// #pragma once

// #include <memory>
// #include <string>
// #include <array>

// #include <controller_interface/multi_interface_controller.h>
// #include <franka_hw/franka_cartesian_command_interface.h>
// #include <franka_hw/franka_state_interface.h>
// #include <hardware_interface/robot_hw.h>
// #include <ros/node_handle.h>
// #include <ros/time.h>

// namespace franka_example_controllers {

// class CartesianVelocityZImpedanceLiMD
//     : public controller_interface::MultiInterfaceController<
//           franka_hw::FrankaVelocityCartesianInterface,
//           franka_hw::FrankaStateInterface> {
//  public:
//   bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
//   void starting(const ros::Time&) override;
//   void update(const ros::Time&, const ros::Duration& period) override;
//   void stopping(const ros::Time&) override;

//  private:
//   franka_hw::FrankaVelocityCartesianInterface* velocity_cartesian_interface_ = nullptr;
//   std::unique_ptr<franka_hw::FrankaCartesianVelocityHandle> velocity_cartesian_handle_;

//   franka_hw::FrankaStateInterface* state_interface_ = nullptr;
//   std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;  // 使用指针避免默认构造

//   double velocity_z_ = 0.0;
//   double velocity_z_prev_ = 0.0;
//   double acc_filtered_ = 0.0;
//   double position_z_relative_ = 0.0;

//   double Md_ = 20.0;
//   double Bd_ = 150.0;
//   double Kd_ = 500.0;
//   double fz_target_ = 1.0;

//   double acc_filter_alpha_ = 0.8;
//   double vel_filter_alpha_ = 0.8;
// };

// }  // namespace franka_example_controllers




#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <ros/subscriber.h>

#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace franka_example_controllers {

class CartesianVelocityZImpedanceLiMD
    : public controller_interface::MultiInterfaceController<
          franka_hw::FrankaVelocityCartesianInterface,
          franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;
  void stopping(const ros::Time&) override;

 private:
  // 接口
  franka_hw::FrankaVelocityCartesianInterface* velocity_cartesian_interface_ = nullptr;
  std::unique_ptr<franka_hw::FrankaCartesianVelocityHandle> velocity_cartesian_handle_;
  franka_hw::FrankaStateInterface* state_interface_ = nullptr;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;

  // Z 轴阻抗控制变量
  double velocity_z_ = 0.0;
  double velocity_z_prev_ = 0.0;
  double acc_filtered_ = 0.0;
  double position_z_relative_ = 0.0;
  double Md_ = 5.0;
  double Bd_ = 140.0;
  double Kd_ = 100.0;
  double fz_target_ = 4.0;
  double acc_filter_alpha_ = 1.0;
  double vel_filter_alpha_ = 1.0;

  // 插值控制（5 DoF）
  bool interpolating_ = true;
  ros::Time interpolation_start_time_;
  // ros::Duration interpolation_duration_; // 移动时间(这个是测试的时候，写的固定时间，实际部署不会用到)
  double interpolation_duration_;  // 移动时间(这个是实际部署的时候，话题传输过来的时间)
  Eigen::Affine3d start_pose_;
  Eigen::Affine3d target_pose_;
  std::vector<Eigen::Affine3d> target_sequence_;
  size_t current_target_idx_ = 0;

  Eigen::Quaterniond fixed_orientation_;
  std::vector<Eigen::Vector3d> target_positions_;
  Eigen::Vector3d target_position_;
  Eigen::Vector3d start_position_;
  Eigen::Affine3d last_interp_pose_;
  // 新增变量
  ros::Time next_interpolation_start_time_;



  // 话题订阅
  ros::Subscriber target_pose_subscriber_;
  void targetPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);

};

}  // namespace franka_example_controllers
