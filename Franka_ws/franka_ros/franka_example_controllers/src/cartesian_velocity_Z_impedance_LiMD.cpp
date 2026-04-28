// #include <franka_example_controllers/cartesian_velocity_Z_impedance_LiMD.h>
// #include <pluginlib/class_list_macros.h>
// #include <ros/ros.h>
// #include <Eigen/Dense>

// namespace franka_example_controllers {

// bool CartesianVelocityZImpedanceLiMD::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
//   std::string arm_id;
//   if (!node_handle.getParam("arm_id", arm_id)) {
//     ROS_ERROR("CartesianVelocityZImpedanceLiMD: Failed to get param arm_id");
//     return false;
//   }

//   velocity_cartesian_interface_ = robot_hw->get<franka_hw::FrankaVelocityCartesianInterface>();
//   state_interface_ = robot_hw->get<franka_hw::FrankaStateInterface>();
//   if (!velocity_cartesian_interface_ || !state_interface_) {
//     ROS_ERROR("CartesianVelocityZImpedanceLiMD: Missing required hardware interface");
//     return false;
//   }

//   try {
//     velocity_cartesian_handle_ = std::make_unique<franka_hw::FrankaCartesianVelocityHandle>(
//         velocity_cartesian_interface_->getHandle(arm_id + "_robot"));
//     state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
//         state_interface_->getHandle(arm_id + "_robot"));
//   } catch (const hardware_interface::HardwareInterfaceException& e) {
//     ROS_ERROR_STREAM("CartesianVelocityZImpedanceLiMD: Exception getting handle: " << e.what());
//     return false;
//   }

//   return true;
// }

// void CartesianVelocityZImpedanceLiMD::starting(const ros::Time&) {
//   velocity_z_ = 0.0;
//   velocity_z_prev_ = 0.0;
//   acc_filtered_ = 0.0;
//   position_z_relative_ = 0.0;
// }

// void CartesianVelocityZImpedanceLiMD::update(const ros::Time&, const ros::Duration& period) {
//   const double dt = period.toSec();
//   const auto& state = state_handle_->getRobotState();  // 使用指针访问
//   const double fz_meas = state.O_F_ext_hat_K[2];

//   double force_error = -(fz_target_ - fz_meas);
//   if (std::abs(force_error) < 0.2)
//     force_error = 0.0;

//   double acc_raw = (force_error - Bd_ * velocity_z_ - Kd_ * position_z_relative_) / Md_;
//   acc_filtered_ = acc_filter_alpha_ * acc_raw + (1.0 - acc_filter_alpha_) * acc_filtered_;
//   velocity_z_ += acc_filtered_ * dt;

//   velocity_z_ = vel_filter_alpha_ * velocity_z_ + (1.0 - vel_filter_alpha_) * velocity_z_prev_;
//   velocity_z_prev_ = velocity_z_;

//   double delta_z = velocity_z_ * dt;
//   position_z_relative_ += delta_z;

//   Eigen::Map<const Eigen::Matrix4d> O_T_EE(state.O_T_EE.data());
//   Eigen::Matrix3d R_base_tool = O_T_EE.block<3,3>(0,0);
//   Eigen::Vector3d z_tool_in_base = R_base_tool.col(2);
//   Eigen::Vector3d v_base = velocity_z_ * z_tool_in_base;

//   std::array<double, 6> v_cmd = {{v_base[0], v_base[1], v_base[2], 0.0, 0.0, 0.0}};
//   velocity_cartesian_handle_->setCommand(v_cmd);
// }

// void CartesianVelocityZImpedanceLiMD::stopping(const ros::Time&) {
//   // Let robot stop gracefully.
// }

// }  // namespace franka_example_controllers

// PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianVelocityZImpedanceLiMD,
//                        controller_interface::ControllerBase)





// // 实现动态往返运动
// #include <franka_example_controllers/cartesian_velocity_Z_impedance_LiMD.h>
// #include <pluginlib/class_list_macros.h>
// #include <ros/ros.h>
// #include <Eigen/Dense>

// namespace franka_example_controllers {

// bool CartesianVelocityZImpedanceLiMD::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
//   std::string arm_id;
//   if (!node_handle.getParam("arm_id", arm_id)) {
//     ROS_ERROR("Failed to get param arm_id");
//     return false;
//   }

//   velocity_cartesian_interface_ = robot_hw->get<franka_hw::FrankaVelocityCartesianInterface>();
//   state_interface_ = robot_hw->get<franka_hw::FrankaStateInterface>();
//   if (!velocity_cartesian_interface_ || !state_interface_) {
//     ROS_ERROR("Missing required hardware interface");
//     return false;
//   }

//   try {
//     velocity_cartesian_handle_ = std::make_unique<franka_hw::FrankaCartesianVelocityHandle>(
//         velocity_cartesian_interface_->getHandle(arm_id + "_robot"));
//     state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
//         state_interface_->getHandle(arm_id + "_robot"));
//   } catch (const hardware_interface::HardwareInterfaceException& e) {
//     ROS_ERROR_STREAM("Handle error: " << e.what());
//     return false;
//   }

//   return true;
// }

// void CartesianVelocityZImpedanceLiMD::starting(const ros::Time& time) {
//   velocity_z_ = 0.0;
//   velocity_z_prev_ = 0.0;
//   acc_filtered_ = 0.0;
//   position_z_relative_ = 0.0;

//   interpolating_ = true;
//   interpolation_start_time_ = time;
//   interpolation_duration_ = ros::Duration(5.0);

//   Eigen::Affine3d current_pose(Eigen::Matrix4d(state_handle_->getRobotState().O_T_EE.data()));
//   fixed_orientation_ = Eigen::Quaterniond(current_pose.rotation());

//   Eigen::Vector3d p0 = current_pose.translation();
//   target_positions_ = {
//     p0 + Eigen::Vector3d(0.0,  0.1, 0.1),
//     p0 + Eigen::Vector3d(0.0, -0.1, 0.0),
//     p0 + Eigen::Vector3d(0.0,  0.0, -0.1)
//   };

//   current_target_idx_ = 0;
//   start_position_ = p0;
//   target_position_ = target_positions_[current_target_idx_];
// }

// void CartesianVelocityZImpedanceLiMD::update(const ros::Time& time, const ros::Duration& period) {
//   const double dt = period.toSec();
//   const auto& state = state_handle_->getRobotState();
//   const double fz_meas = state.O_F_ext_hat_K[2];

//   double error_force = -(fz_target_ - fz_meas);
//   if (std::abs(error_force) < 0.2) error_force = 0.0;
//   double acc_raw = (error_force - Bd_ * velocity_z_ - Kd_ * position_z_relative_) / Md_;
//   acc_filtered_ = acc_filter_alpha_ * acc_raw + (1.0 - acc_filter_alpha_) * acc_filtered_;
//   velocity_z_ += acc_filtered_ * dt;
//   velocity_z_ = vel_filter_alpha_ * velocity_z_ + (1.0 - vel_filter_alpha_) * velocity_z_prev_;
//   velocity_z_prev_ = velocity_z_;
//   double delta_z = velocity_z_ * dt;
//   position_z_relative_ += delta_z;

//   Eigen::Affine3d current_pose(Eigen::Matrix4d(state.O_T_EE.data()));
//   Eigen::Vector3d z_tool = current_pose.linear().col(2);
//   Eigen::Vector3d v_z = velocity_z_ * z_tool;

//   Eigen::Vector3d v_trans = Eigen::Vector3d::Zero();

//   if (interpolating_) {
//     ros::Duration elapsed = time - interpolation_start_time_;
//     double t = elapsed.toSec();
//     double T = interpolation_duration_.toSec();

//     if (t >= T) {
//       current_target_idx_ = (current_target_idx_ + 1) % target_positions_.size();
//       start_position_ = target_position_;  // 起点 = 上段终点，保证速度连续
//       target_position_ = target_positions_[current_target_idx_];
//       interpolation_start_time_ = time;
//       t = 0.0;
//     }

//     double s = 10 * pow(t / T, 3) - 15 * pow(t / T, 4) + 6 * pow(t / T, 5);
//     Eigen::Vector3d p_interp = (1.0 - s) * start_position_ + s * target_position_;
//     Eigen::Vector3d dp = p_interp - current_pose.translation();

//     v_trans = 2.0 * dp;
//     v_trans = v_trans.cwiseMax(-0.3).cwiseMin(0.3);
//   }

//   std::array<double, 6> v_cmd = {{
//     v_trans[0] + v_z[0],
//     v_trans[1] + v_z[1],
//     v_trans[2] + v_z[2],
//     0.0, 0.0, 0.0
//   }};
//   velocity_cartesian_handle_->setCommand(v_cmd);
// }

// void CartesianVelocityZImpedanceLiMD::stopping(const ros::Time&) {}

// }  // namespace franka_example_controllers

// PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianVelocityZImpedanceLiMD,
//                        controller_interface::ControllerBase)




// // // v3 接收话题  实现动态往返运动。但是绕y、Z轴旋转有问题
// #include <franka_example_controllers/cartesian_velocity_Z_impedance_LiMD.h>
// #include <pluginlib/class_list_macros.h>
// #include <ros/ros.h>
// #include <Eigen/Dense>

// namespace franka_example_controllers {

// bool CartesianVelocityZImpedanceLiMD::init(hardware_interface::RobotHW* robot_hw,
//                                            ros::NodeHandle& node_handle) {
//   std::string arm_id;
//   if (!node_handle.getParam("arm_id", arm_id)) {
//     ROS_ERROR("Failed to get param arm_id");
//     return false;
//   }

//   velocity_cartesian_interface_ = robot_hw->get<franka_hw::FrankaVelocityCartesianInterface>();
//   state_interface_ = robot_hw->get<franka_hw::FrankaStateInterface>();
//   if (!velocity_cartesian_interface_ || !state_interface_) {
//     ROS_ERROR("Missing required hardware interface");
//     return false;
//   }

//   try {
//     velocity_cartesian_handle_ = std::make_unique<franka_hw::FrankaCartesianVelocityHandle>(
//         velocity_cartesian_interface_->getHandle(arm_id + "_robot"));
//     state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
//         state_interface_->getHandle(arm_id + "_robot"));
//   } catch (const hardware_interface::HardwareInterfaceException& e) {
//     ROS_ERROR_STREAM("Handle error: " << e.what());
//     return false;
//   }

//   target_pose_subscriber_ = node_handle.subscribe<geometry_msgs::PoseStamped>(
//       "target_pose", 1, &CartesianVelocityZImpedanceLiMD::targetPoseCallback, this);

//   return true;
// }

// void CartesianVelocityZImpedanceLiMD::starting(const ros::Time& time) {
//   velocity_z_ = 0.0;
//   velocity_z_prev_ = 0.0;
//   acc_filtered_ = 0.0;
//   position_z_relative_ = 0.0;

//   interpolating_ = false;

//   Eigen::Affine3d current_pose(Eigen::Matrix4d(state_handle_->getRobotState().O_T_EE.data()));
//   start_pose_ = current_pose;
//   target_pose_ = current_pose;
// }

// void CartesianVelocityZImpedanceLiMD::update(const ros::Time& time, const ros::Duration& period) {
//   const double dt = period.toSec();
//   const auto& state = state_handle_->getRobotState();
//   const double fz_meas = state.O_F_ext_hat_K[2];

//   // --- Z 轴阻抗控制 ---
//   double error_force = -(fz_target_ - fz_meas);
//   if (std::abs(error_force) < 0.2) error_force = 0.0;
//   double acc_raw = (error_force - Bd_ * velocity_z_ - Kd_ * position_z_relative_) / Md_;
//   acc_filtered_ = acc_filter_alpha_ * acc_raw + (1.0 - acc_filter_alpha_) * acc_filtered_;
//   velocity_z_ += acc_filtered_ * dt;
//   velocity_z_ = vel_filter_alpha_ * velocity_z_ + (1.0 - vel_filter_alpha_) * velocity_z_prev_;
//   velocity_z_prev_ = velocity_z_;
//   double delta_z = velocity_z_ * dt;
//   position_z_relative_ += delta_z;

//   Eigen::Affine3d current_pose(Eigen::Matrix4d(state.O_T_EE.data()));
//   Eigen::Vector3d z_tool = current_pose.linear().col(2);
//   Eigen::Vector3d v_z = velocity_z_ * z_tool;

//   // --- 插值轨迹速度 ---
//   Eigen::Vector3d v_trans = Eigen::Vector3d::Zero();
//   Eigen::Vector3d w_omega = Eigen::Vector3d::Zero();

//   if (interpolating_) {
//     ros::Duration elapsed = time - interpolation_start_time_;
//     double t = elapsed.toSec();
//     // double T = interpolation_duration_.toSec();
//     double T = interpolation_duration_;

//     if (t >= T) {
//       interpolating_ = false;
//       // 避免末端速度跳变，继续发送零速度指令一段时间由控制器内部收敛
//       v_trans.setZero();
//       w_omega.setZero();
//     } else {
//       double tau = t / T;
//       double s_dot = (30 * pow(tau, 2) - 60 * pow(tau, 3) + 30 * pow(tau, 4)) / T;

//       Eigen::Vector3d dp = target_pose_.translation() - start_pose_.translation();
//       v_trans = dp * s_dot;

//       Eigen::Quaterniond q_start(start_pose_.rotation());
//       Eigen::Quaterniond q_target(target_pose_.rotation());
//       Eigen::Quaterniond q_err = q_start.inverse() * q_target;
//       Eigen::AngleAxisd aa_err(q_err);
//       double angle = aa_err.angle();
//       Eigen::Vector3d axis = aa_err.axis();
//       Eigen::Vector3d dtheta = angle * axis;
//       // w_omega = dtheta * s_dot;
//       // 工具系角速度转为 base 角速度
//       w_omega = current_pose.rotation() * dtheta * s_dot;  // 修复y、z轴的bug

//       v_trans = v_trans.cwiseMax(-0.3).cwiseMin(0.3);
//       w_omega = w_omega.cwiseMax(-1.0).cwiseMin(1.0);
//     }
//   }

//   std::array<double, 6> v_cmd = {{
//     v_trans[0] + v_z[0],
//     v_trans[1] + v_z[1],
//     v_trans[2] + v_z[2],
//     w_omega[0], w_omega[1], w_omega[2]
//   }};
//   velocity_cartesian_handle_->setCommand(v_cmd);
// }

// void CartesianVelocityZImpedanceLiMD::targetPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
//   Eigen::Affine3d current_pose(Eigen::Matrix4d(state_handle_->getRobotState().O_T_EE.data()));
//   start_pose_ = current_pose;

//   Eigen::Affine3d T;
//   T.translation() << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;

//   Eigen::Quaterniond q(msg->pose.orientation.w,
//                        msg->pose.orientation.x,
//                        msg->pose.orientation.y,
//                        msg->pose.orientation.z);
//   T.linear() = q.normalized().toRotationMatrix();

//   target_pose_ = T;
//   interpolation_start_time_ = ros::Time::now();
// //   interpolation_duration_ = ros::Duration(5.0);
//   interpolation_duration_ = msg->header.stamp.toSec();  // 话题传输过来的时间戳充当运动时间
//   interpolating_ = true;

//   ROS_INFO_STREAM("New target received, interpolating pose.\n"
//                   << "Start: \n" << start_pose_.matrix()
//                   << "\nTarget:\n" << target_pose_.matrix()
//                   << "\nMove time:\n" << interpolation_duration_);
// }

// void CartesianVelocityZImpedanceLiMD::stopping(const ros::Time&) {}

// }  // namespace franka_example_controllers

// PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianVelocityZImpedanceLiMD,
//                        controller_interface::ControllerBase)





// v4 修复  当末端姿态有偏转（尤其是倾斜）时，阻抗特性“变硬”
#include <franka_example_controllers/cartesian_velocity_Z_impedance_LiMD.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <Eigen/Dense>

namespace franka_example_controllers {

bool CartesianVelocityZImpedanceLiMD::init(hardware_interface::RobotHW* robot_hw,
                                           ros::NodeHandle& node_handle) {
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("Failed to get param arm_id");
    return false;
  }

  velocity_cartesian_interface_ = robot_hw->get<franka_hw::FrankaVelocityCartesianInterface>();
  state_interface_ = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (!velocity_cartesian_interface_ || !state_interface_) {
    ROS_ERROR("Missing required hardware interface");
    return false;
  }

  try {
    velocity_cartesian_handle_ = std::make_unique<franka_hw::FrankaCartesianVelocityHandle>(
        velocity_cartesian_interface_->getHandle(arm_id + "_robot"));
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM("Handle error: " << e.what());
    return false;
  }

  target_pose_subscriber_ = node_handle.subscribe<geometry_msgs::PoseStamped>(
      "target_pose", 1, &CartesianVelocityZImpedanceLiMD::targetPoseCallback, this);

  return true;
}

void CartesianVelocityZImpedanceLiMD::starting(const ros::Time& time) {
  velocity_z_ = 0.0;
  velocity_z_prev_ = 0.0;
  acc_filtered_ = 0.0;
  position_z_relative_ = 0.0;

  interpolating_ = false;

  Eigen::Affine3d current_pose(Eigen::Matrix4d(state_handle_->getRobotState().O_T_EE.data()));
  start_pose_ = current_pose;
  target_pose_ = current_pose;
}

void CartesianVelocityZImpedanceLiMD::update(const ros::Time& time, const ros::Duration& period) {
  const double dt = period.toSec();
  const auto& state = state_handle_->getRobotState();

  // 先提取当前末端位姿
  Eigen::Affine3d current_pose(Eigen::Matrix4d(state.O_T_EE.data()));
  Eigen::Matrix3d R_base_tool = current_pose.linear();  // R_base_to_tool

  // 将基坐标系下的力转换为工具坐标系下
  Eigen::Vector3d f_base(state.O_F_ext_hat_K[0],
                         state.O_F_ext_hat_K[1],
                         state.O_F_ext_hat_K[2]);
  Eigen::Vector3d f_tool = R_base_tool.transpose() * f_base;
  double fz_meas_tool = f_tool(2);

  // --- Z 轴阻抗控制 ---
  double error_force = (fz_target_ - fz_meas_tool);
  // if (std::abs(error_force) < 0.2) error_force = 0.0;
  if (std::abs(error_force) < 0.05) error_force = 0.0;

  double acc_raw = (error_force - Bd_ * velocity_z_ - Kd_ * position_z_relative_) / Md_;
  acc_filtered_ = acc_filter_alpha_ * acc_raw + (1.0 - acc_filter_alpha_) * acc_filtered_;
  velocity_z_ += acc_filtered_ * dt;
  velocity_z_ = vel_filter_alpha_ * velocity_z_ + (1.0 - vel_filter_alpha_) * velocity_z_prev_;
  velocity_z_prev_ = velocity_z_;
  double delta_z = velocity_z_ * dt;
  position_z_relative_ += delta_z;

  Eigen::Vector3d z_tool = current_pose.linear().col(2);
  Eigen::Vector3d v_z = velocity_z_ * z_tool;

  // --- 插值轨迹速度 ---
  Eigen::Vector3d v_trans = Eigen::Vector3d::Zero();
  Eigen::Vector3d w_omega = Eigen::Vector3d::Zero();

  if (interpolating_) {
    ros::Duration elapsed = time - interpolation_start_time_;
    double t = elapsed.toSec();
    double T = interpolation_duration_;

    if (t >= T) {
      interpolating_ = false;
      v_trans.setZero();
      w_omega.setZero();
    } else {
      double tau = t / T;
      double s_dot = (30 * pow(tau, 2) - 60 * pow(tau, 3) + 30 * pow(tau, 4)) / T;

      Eigen::Vector3d dp = target_pose_.translation() - start_pose_.translation();
      v_trans = dp * s_dot;

      Eigen::Quaterniond q_start(start_pose_.rotation());
      Eigen::Quaterniond q_target(target_pose_.rotation());
      Eigen::Quaterniond q_err = q_start.inverse() * q_target;
      Eigen::AngleAxisd aa_err(q_err);
      double angle = aa_err.angle();
      Eigen::Vector3d axis = aa_err.axis();
      Eigen::Vector3d dtheta = angle * axis;

      // ⚠️ 转为 base 坐标系下角速度，解决 y/z 转轴跳变问题
      w_omega = current_pose.rotation() * dtheta * s_dot;

      v_trans = v_trans.cwiseMax(-0.3).cwiseMin(0.3);
      w_omega = w_omega.cwiseMax(-1.0).cwiseMin(1.0);
    }
  }

  std::array<double, 6> v_cmd = {{
    v_trans[0] + v_z[0],
    v_trans[1] + v_z[1],
    v_trans[2] + v_z[2],
    w_omega[0], w_omega[1], w_omega[2]
  }};
  velocity_cartesian_handle_->setCommand(v_cmd);
}


void CartesianVelocityZImpedanceLiMD::targetPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  Eigen::Affine3d current_pose(Eigen::Matrix4d(state_handle_->getRobotState().O_T_EE.data()));
  start_pose_ = current_pose;

  Eigen::Affine3d T;
  T.translation() << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;

  Eigen::Quaterniond q(msg->pose.orientation.w,
                       msg->pose.orientation.x,
                       msg->pose.orientation.y,
                       msg->pose.orientation.z);
  T.linear() = q.normalized().toRotationMatrix();

  target_pose_ = T;
  interpolation_start_time_ = ros::Time::now();
//   interpolation_duration_ = ros::Duration(5.0);
  interpolation_duration_ = msg->header.stamp.toSec();  // 话题传输过来的时间戳充当运动时间
  interpolating_ = true;

  ROS_INFO_STREAM("New target received, interpolating pose.\n"
                  << "Start: \n" << start_pose_.matrix()
                  << "\nTarget:\n" << target_pose_.matrix()
                  << "\nMove time:\n" << interpolation_duration_);
}

void CartesianVelocityZImpedanceLiMD::stopping(const ros::Time&) {}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianVelocityZImpedanceLiMD,
                       controller_interface::ControllerBase)




