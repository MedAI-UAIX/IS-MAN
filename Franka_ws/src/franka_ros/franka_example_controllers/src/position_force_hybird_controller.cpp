// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/position_force_hybird_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include<condition_variable>
#include <franka_example_controllers/pseudo_inversion.h>
#include<thread>




namespace franka_example_controllers {

bool position_force_hybird_controller::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  //calcu_invers=false;
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;
  sub_f_ext_ = node_handle.subscribe(
      "/franka_state_controller/F_ext", 10,
      &position_force_hybird_controller::f_ext_callback, this,
      ros::TransportHints().reliable().tcpNoDelay());
  // sub_f_ext_ = node_handle.subscribe(
  //     "/robotiq_ft_sensor_calibra", 10,
  //     &position_force_hybird_controller::f_ext_callback, this,
  //     ros::TransportHints().reliable().tcpNoDelay());

  sub_equilibrium_pose_ = node_handle.subscribe(
     "equilibrium_pose", 20, &position_force_hybird_controller::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());//get pose from the topic equilibrium_pose
  sub_pose_ = node_handle.subscribe(
      "required_pose", 20, &position_force_hybird_controller::PoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());//get pose from the topic required_pose
  sub_add_pose_ = node_handle.subscribe(
      "add_pose", 20, &position_force_hybird_controller::add_PoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());//get pose from the topic required_pose
  //pub_end_force_torque=node_handle.advertise<franka_example_controllers::EndForceTorque>("EndForceTorque",1);
  
//initialize to get hardware_interface
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("position_force_hybird_controller: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "position_force_hybird_controller: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }
  
  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "position_force_hybird_controller: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "position_force_hybird_controller: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "position_force_hybird_controller: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "position_force_hybird_controller: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "position_force_hybird_controller: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "position_force_hybird_controller: Exception getting joint handles: " << ex.what());
      return false;
    }
  }
  
 
  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "/dynamic_reconfigure_position_force_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_example_controllers::position_force_paramConfig>>(

      dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&position_force_hybird_controller::PositionForceParamCallback, this, _1, _2));

  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  cartesian_stiffness_.setZero();
  cartesian_damping_.setZero();

  return true;
}

Eigen::Matrix<double, 6, 1> desired_force_error_last,inte_force_error;
void position_force_hybird_controller::starting(const ros::Time& /*time*/) {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // //transfer the jacobian to the global for calculate reverse  no need for mutex 
  // Eigen::Map<const Eigen::Matrix<double,6,7>> jacobian_matrix(jacobian_array.data());
  // quanju_jacobian=jacobian_matrix;
  // calcu_invers=true;
  // start_move.notify_all();
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set equilibrium point to current state
  position_d_ = initial_transform.translation();
  ROS_INFO_STREAM("Initial position (x, y, z): " << position_d_.transpose());
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.rotation());
//initiaize last value
  inte_force_error.setZero();
  desired_force_error_last.setZero();
  // set nullspace equilibrium configuration to initial q
  q_d_nullspace_ = q_initial;
  

}

void position_force_hybird_controller::update(const ros::Time& /*time*/,
                                                 const ros::Duration& period) {
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
 //std::array<double, 7> gravity_array= model_handle_->getGravity();
  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
      //the measured torques
  Eigen::Map<Eigen::Matrix<double, 7, 1>>  tau_J(  
      robot_state.tau_J.data());
  // Eigen::Map<Eigen::Matrix<double, 7, 1>>  gravity(
  //     gravity_array.data());

  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());

  // compute error to desired pose
  // position error
  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << position - position_d_;
  error.head(3)=transform.rotation().transpose()*error.head(3);
  // orientation error
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();

  // Transform to base frame
  error.tail(3) << -transform.rotation() * error.tail(3);

  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

  // pseudoinverse for nullspace handling
  // kinematic pseuoinverse
  // Eigen::MatrixXd jacobian_transpose_pinv;
  // pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // Cartesian PD control with damping ratio = 1
  if(is_contact_scanning)
  {
  cartesian_stiffness_(2,2)=0;
  cartesian_damping_(2,2)=0;
  }
  Eigen::VectorXd diagnoal(6);
	diagnoal << 1, 1, 1, 1, 1, 1;
  //我们上面没变角度的坐标系，角度还是基坐标系
  //两个分快矩阵，右下角是对角度111,分别把基坐标B转换到末端E，以及反过来
  Eigen::Matrix<double,6,6> fenkuan_B2E,fenkuan_E2B;
  fenkuan_E2B=diagnoal.asDiagonal();
  fenkuan_B2E=diagnoal.asDiagonal();
  fenkuan_E2B.block<3,3>(0,0)=transform.rotation();
  fenkuan_B2E.block<3,3>(0,0)=transform.rotation().transpose();
  tau_task << jacobian.transpose() *fenkuan_E2B*
                  (-cartesian_stiffness_ * error- cartesian_damping_*(fenkuan_B2E*jacobian * dq));
  
  if(is_contact_scanning)
  {
    std::array<double, 7> tau_d_calculated;
    double clap_gravity=0;//define clap_gravity positive
    Eigen::VectorXd desired_force(6),desired_force_error(6), tau_d_calculated_vector(7),diff_desired_force_error(6);
    //external force with PD controller
    // Eigen::Matrix<double, 7, 1> tau_q_measured=tau_J-gravity;
    // Eigen::Matrix<double, 6, 1> force_measured=jacobian_transpose_pinv*tau_q_measured;
   // ROS_INFO("force measured:%lf,%lf,%lf",force_measured(0,0),force_measured(1,0),force_measured(2,0));
   //发布测得的力，用于监测
    //franka_example_controllers::EndForceTorque pubed;
    //Eigen::VectorXd::Map(&pubed.ForceTorque[0],6)=force_measured;
    //pub_end_force_torque.publish(pubed);
    Eigen::Matrix<double, 6, 1> force_measured;
    {
      std::lock_guard<std::mutex> lock(f_ext_mutex_);
      force_measured = f_ext_measured_;
    }
    desired_force << 0, 0, ext_force-clap_gravity, 0, 0, 0;
    desired_force_error << 0, 0, ext_force-clap_gravity-force_measured(2,0), 0, 0, 0;
    //ROS_INFO("force error:%lf",desired_force_error(2,0));
    diff_desired_force_error=desired_force_error-desired_force_error_last;
    inte_force_error=inte_force_error+desired_force_error;
    desired_force_error_last=desired_force_error;
    //                                                            feedforward      P                        I                                   D
    tau_d_calculated_vector = jacobian.transpose() * fenkuan_E2B*(force_PID_P*desired_force_error+force_PID_I *inte_force_error+force_PID_D*diff_desired_force_error);
  // Desired torque

  tau_d << tau_task  + coriolis+tau_d_calculated_vector;
  }
  else
  {
    tau_d << tau_task  + coriolis;
  }
  //tau_d << tau_task  + coriolis;
  // Saturate torque rate to avoid discontinuities
  tau_d << saturateTorqueRate(tau_d, tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d(i));
  }



  // update parameters changed online either through dynamic reconfigure or through the interactive
  // target by filtering
  cartesian_stiffness_ =
      filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_ =
      filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
  nullspace_stiffness_ =
      filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
}

Eigen::Matrix<double, 7, 1> position_force_hybird_controller::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}
void position_force_hybird_controller::stopping(const ros::Time& time)
{
  //calcu_invers=false;
}
// void position_force_hybird_controller::complianceParamCallback(
//     franka_example_controllers::compliance_paramConfig& config,
//     uint32_t /*level*/) {
//   cartesian_stiffness_target_.setIdentity();
//   cartesian_stiffness_target_.topLeftCorner(3, 3)
//       << config.translational_stiffness * Eigen::Matrix3d::Identity();
//   cartesian_stiffness_target_.bottomRightCorner(3, 3)
//       << config.rotational_stiffness * Eigen::Matrix3d::Identity();
//   cartesian_damping_target_.setIdentity();
//   // Damping ratio = 1
//   cartesian_damping_target_.topLeftCorner(3, 3)
//       << 2.0 * sqrt(config.translational_stiffness) * Eigen::Matrix3d::Identity();
//   cartesian_damping_target_.bottomRightCorner(3, 3)
//       << 2.0 * sqrt(config.rotational_stiffness) * Eigen::Matrix3d::Identity();
//   nullspace_stiffness_target_ = config.nullspace_stiffness;
// }
void position_force_hybird_controller::PositionForceParamCallback(franka_example_controllers::position_force_paramConfig &config,uint32_t level)
{
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_.topLeftCorner(3, 3)<<config.translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3)
      << config.rotational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.setIdentity();
  // Damping ratio = 1
  cartesian_damping_target_.topLeftCorner(3, 3)
      << 2.0 * sqrt(config.translational_stiffness) * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.bottomRightCorner(3, 3)
      << 2.0 * sqrt(config.rotational_stiffness) * Eigen::Matrix3d::Identity();

  is_contact_scanning=config.is_scaning;

  ext_force=config.ext_force;
  force_PID_P=config.force_p;
  force_PID_I=config.force_I;
  force_PID_D=config.force_D;
  filter_params_ = config.filter_params;
}
void position_force_hybird_controller::equilibriumPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
}
void position_force_hybird_controller::PoseCallback(
    const geometry_msgs::PoseConstPtr& msg) {
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_target_ << msg->position.x, msg->position.y, msg->position.z;
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->orientation.x, msg->orientation.y,
      msg->orientation.z, msg->orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
}
void position_force_hybird_controller::add_PoseCallback(
    const geometry_msgs::PoseConstPtr& msg) {
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  ROS_INFO("I am post");
  ROS_INFO("%lf,%lf,%lf",position_d_target_[0],position_d_target_[1],position_d_target_[2]);
  position_d_target_ << position_d_target_[0]+msg->position.x, position_d_target_[1]+msg->position.y, position_d_target_[2]+msg->position.z;
//  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
//  Eigen::Quaterniond change;
//  change.coeffs()<< msg->orientation.x, msg->orientation.y,
//      msg->orientation.z, msg->orientation.w;
//  orientation_d_target_=change*last_orientation_d_target;
//  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
//    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
//  }
  
 // dynamic_server_compliance_param_->updateConfig(); 可能要改接触状态，这里是为了将节点接受到的参数发给服务器
}
void position_force_hybird_controller::f_ext_callback(
    const geometry_msgs::WrenchStamped::ConstPtr& msg) {
  std::lock_guard<std::mutex> lock(f_ext_mutex_);
  f_ext_measured_ << msg->wrench.force.x,
                     msg->wrench.force.y,
                     msg->wrench.force.z,
                     msg->wrench.torque.x,
                     msg->wrench.torque.y,
                     msg->wrench.torque.z;
}
}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::position_force_hybird_controller,
                       controller_interface::ControllerBase)
