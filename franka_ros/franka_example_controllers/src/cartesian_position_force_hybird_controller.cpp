// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
/*
时间：2024年1月11
作者：LiMD
目的：写在力位混合控制器的基础上，加上纯位置控制的模式
*/

//头文件，定义了一些变量/类
#include <franka_example_controllers/cartesian_position_force_hybird_controller.h>

#include <cmath>
#include <memory>

//力位混合控制器
#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include<condition_variable>
#include <franka_example_controllers/pseudo_inversion.h>
#include<thread>

// 位置控制器
#include <stdexcept>
#include <string>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <hardware_interface/hardware_interface.h>




namespace franka_example_controllers {

bool cartesian_position_force_hybird_controller::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  //calcu_invers=false;
  // std::vector<double> cartesian_stiffness_vector;
  // std::vector<double> cartesian_damping_vector;

  //===== 话题订阅，获取目标点4*4齐次矩阵
  sub_equilibrium_pose_ = node_handle.subscribe(
      "equilibrium_pose", 20, &cartesian_position_force_hybird_controller::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());//get pose from the topic equilibrium_pose
  
  /*
  ====== 话题发布
  创建了一个名为"EndForceTorque"的主题（topic），
  用于发布Franka机器人的末端力和力矩信息。主题名称可以根据需要进行更改。

这里的node_handle是一个ROS节点句柄（node handle），用于与ROS系统进行通信。
advertise函数用于创建一个发布者对象，它将消息发布到指定的主题上。

<franka_example_controllers::EndForceTorque>是发布者的消息类型，它指定了发布的数据的格式和结构。
在这种情况下，发布的消息是franka_example_controllers包中定义的EndForceTorque消息类型。

最后的参数"1"表示发布者的消息队列大小，指定了发布者在消息队列中保存的消息数量。
这里设置为1，表示只保留最新的一条消息
  */
  // pub_end_force_torque=node_handle.advertise<franka_example_controllers::EndForceTorque>("EndForceTorque",1);
  
//initialize to get hardware_interface

 

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("cartesian_position_force_hybird_controller: Could not read parameter arm_id");
    return false;
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "cartesian_position_force_hybird_controller: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }
  
  if (!node_handle.getParam("k_gains", k_gains_) || k_gains_.size() != 7) {
    ROS_ERROR(
        "JointImpedanceExampleController:  Invalid or no k_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("d_gains", d_gains_) || d_gains_.size() != 7) {
    ROS_ERROR(
        "JointImpedanceExampleController:  Invalid or no d_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  double publish_rate(30.0);
  if (!node_handle.getParam("publish_rate", publish_rate)) {
    ROS_INFO_STREAM("JointImpedanceExampleController: publish_rate not found. Defaulting to "
                    << publish_rate);
  }
  rate_trigger_ = franka_hw::TriggerRate(publish_rate);

  if (!node_handle.getParam("coriolis_factor", coriolis_factor_)) {
    ROS_INFO_STREAM("JointImpedanceExampleController: coriolis_factor not found. Defaulting to "
                    << coriolis_factor_);
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "cartesian_position_force_hybird_controller: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "cartesian_position_force_hybird_controller: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

   // 获取笛卡尔控制器，并且判断是否获取成功
auto* cartesian_pose_interface_ = robot_hw->get<franka_hw::FrankaPoseCartesianInterface>();
if (cartesian_pose_interface_ == nullptr) {
    ROS_ERROR(
        "CartesianPoseExampleController: Could not get Cartesian Pose "
        "interface from hardware");
    return false;
  }
  //成功的话，获取笛卡尔位置控制器的话柄
  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianPoseExampleController: Exception getting Cartesian handle: " << e.what());
    return false;
  }

 

  //获取机器人状态的话柄
  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "cartesian_position_force_hybird_controller: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "cartesian_position_force_hybird_controller: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }


  
  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM("ForceExampleController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM("ForceExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "/dynamic_reconfigure_position_force_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_example_controllers::cartesian_position_force_paramConfig>>(

      dynamic_reconfigure_compliance_param_node_);
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&cartesian_position_force_hybird_controller::PositionForceParamCallback, this, _1, _2));

  // position_d_.setZero();
  // orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  // position_d_target_.setZero();
  // orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  // cartesian_stiffness_.setZero();
  // cartesian_damping_.setZero();

  std::fill(dq_filtered_.begin(), dq_filtered_.end(), 0);

  return true;
}

void cartesian_position_force_hybird_controller::starting(const ros::Time& /*time*/) {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  initial_pose_ = initial_state.O_T_EE_d;
  // get jacobian
  // std::array<double, 42> jacobian_array =
  //     model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  // Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE_c.data()));
  // elapsed_time_ = ros::Duration(0.0);
  
  // // set equilibrium point to current state
  // //初始位置的位置，姿态
  // position_d_ = initial_transform.translation();
  // orientation_d_ = initial_transform.rotation();
  // //目标位置的位置，姿态
  // position_d_target_ = initial_transform.translation();
  // orientation_d_target_ =initial_transform.rotation();
//initiaize last value
  // inte_force_error.setZero();
  // desired_force_error_last.setZero();
  // set nullspace equilibrium configuration to initial q
  // q_d_nullspace_ = q_initial;
  
  moving_time_ = 0.1;
}

void cartesian_position_force_hybird_controller::update(const ros::Time& /*time*/,
                                                 const ros::Duration& period) {
  if (elapsed_time_.toSec()<moving_time_)
  {
    elapsed_time_ += period;
    fifth_intP(parax, position_d_[0], position_d_target_[0], moving_time_);
    fifth_intP(paray, position_d_[1], position_d_target_[1], moving_time_);
    fifth_intP(paraz, position_d_[2], position_d_target_[2], moving_time_);
    double nowx = getintP(parax, elapsed_time_.toSec());
    double nowy = getintP(paray, elapsed_time_.toSec());
    double nowz = getintP(paraz, elapsed_time_.toSec());
    Eigen::Quaterniond now_Quat;
    now_Quat = orientation_d_.slerp(Funtion_t(elapsed_time_.toSec() / moving_time_), orientation_d_target_);
    // ROS_INFO("orientation_d_target_0:%lf",orientation_d_target_.coeffs()[0]);
    // ROS_INFO("orientation_d_target_1:%lf",orientation_d_target_.coeffs()[1]);
    // ROS_INFO("orientation_d_target_2:%lf",orientation_d_target_.coeffs()[2]);
    // ROS_INFO("orientation_d_target_3:%lf",orientation_d_target_.coeffs()[3]);
    Eigen::Matrix3d now_R(now_Quat);
  // ROS_INFO("start_position_x:%lf",position_d_target_[0]);
  // ROS_INFO("start_position_y:%lf",position_d_target_[1]);
  // ROS_INFO("start_position_z:%lf",position_d_target_[2]);
  // ROS_INFO("start_position_rx:%lf",now_Quat.coeffs()[0]);
  // ROS_INFO("start_position_ry:%lf",now_Quat.coeffs()[1]);
  // ROS_INFO("start_position_rz:%lf",now_Quat.coeffs()[2]);
  // ROS_INFO("start_position_rw:%lf",now_Quat.coeffs()[3]);
    std::array<double, 16> new_pose;
    new_pose[0] = now_R(0, 0);          new_pose[1] = now_R(1, 0);          new_pose[2] = now_R(2, 0);          new_pose[3] = 0;
    new_pose[4] = now_R(0, 1);          new_pose[5] = now_R(1, 1);          new_pose[6] = now_R(2, 1);          new_pose[7] = 0;
    new_pose[8] = now_R(0, 2);          new_pose[9] = now_R(1, 2);          new_pose[10] = now_R(2, 2);          new_pose[11] = 0;
    new_pose[12] = nowx;          new_pose[13] = nowy;          new_pose[14] = nowz;          new_pose[15] = 1;
    cartesian_pose_handle_->setCommand(new_pose);
    // ROS_INFO("cartesian x:%lf",nowx);
  }                                                
//   // get state variables
//   franka::RobotState robot_state = state_handle_->getRobotState();
//   std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
//   std::array<double, 42> jacobian_array =
//       model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
//  std::array<double, 7> gravity_array= model_handle_->getGravity();
//   // convert to Eigen
//   /*
//     coriolis：一个大小为7x1的向量，用于存储机器人各关节的科里奥利力（Coriolis force）。
//     jacobian：一个大小为6x7的矩阵，用于存储机器人当前时刻的雅可比矩阵（Jacobian matrix）。
//     q：一个大小为7x1的向量，用于存储机器人各关节的当前角度。
//     dq：一个大小为7x1的向量，用于存储机器人各关节的当前角速度。
//     tau_J_d：一个大小为7x1的向量，用于存储机器人当前时刻的期望关节力矩（desired joint torque），也就是控制器计算得到的理论值。
//     tau_J：一个大小为7x1的向量，用于存储机器人当前时刻的实际关节力矩（measured joint torque），也就是从机器人硬件接口中读取得到的真实值。
//     gravity：一个大小为7x1的向量，用于存储机器人的重力向量，也就是重力对各关节的作用力矩。
//   */
//   Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
//   Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
//   Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
//   Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
//   Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
//       robot_state.tau_J_d.data());
//       //the measured torques
//   Eigen::Map<Eigen::Matrix<double, 7, 1>>  tau_J(  
//       robot_state.tau_J.data());
//   Eigen::Map<Eigen::Matrix<double, 7, 1>>  gravity(
//       gravity_array.data());

//   Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
//   Eigen::Vector3d position(transform.translation());
//   Eigen::Quaterniond orientation(transform.rotation());

//   // compute error to desired pose
//   // position error
//  Eigen::Matrix<double, 6, 1> error;
//   error.head(3) << position - position_d_;
//   error.head(3)=transform.rotation().transpose()*error.head(3);
//   // orientation error
//   if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
//     orientation.coeffs() << -orientation.coeffs();
//   }
//   // "difference" quaternion
//   Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
//   error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();

//   // Transform to base frame
//   error.tail(3) << -transform.rotation() * error.tail(3);

//   // compute control
//   // allocate variables
//   Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

//   // pseudoinverse for nullspace handling
//   // kinematic pseuoinverse
//   Eigen::MatrixXd jacobian_transpose_pinv;
//   pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

//   // Cartesian PD control with damping ratio = 1
//   if(is_contact_scanning==2)
//   {
//   cartesian_stiffness_(2,2)=0;
//   cartesian_damping_(2,2)=0;
//   }
//   Eigen::VectorXd diagnoal(6);
// 	diagnoal << 1, 1, 1, 1, 1, 1;
//   //我们上面没变角度的坐标系，角度还是基坐标系
//   //两个分快矩阵，右下角是对角度111,分别把基坐标B转换到末端E，以及反过来
//   Eigen::Matrix<double,6,6> fenkuan_B2E,fenkuan_E2B;
//   fenkuan_E2B=diagnoal.asDiagonal();
//   fenkuan_B2E=diagnoal.asDiagonal();
//   fenkuan_E2B.block<3,3>(0,0)=transform.rotation();
//   fenkuan_B2E.block<3,3>(0,0)=transform.rotation().transpose();
//   tau_task << jacobian.transpose() *fenkuan_E2B*
//                   (-cartesian_stiffness_ * error- cartesian_damping_*(fenkuan_B2E*jacobian * dq));
  
//   if(is_contact_scanning==2)
//   {
//     std::array<double, 7> tau_d_calculated;
//     double clap_gravity=0;//define clap_gravity positive
//     Eigen::VectorXd desired_force(6),desired_force_error(6), tau_d_calculated_vector(7),diff_desired_force_error(6);
//     //external force with PD controller
//     Eigen::Matrix<double, 7, 1> tau_q_measured=tau_J-gravity;
//     Eigen::Matrix<double, 6, 1> force_measured=jacobian_transpose_pinv*tau_q_measured;
//    // ROS_INFO("force measured:%lf,%lf,%lf",force_measured(0,0),force_measured(1,0),force_measured(2,0));
//    //发布测得的力，用于监测
//     franka_example_controllers::EndForceTorque pubed;
//     Eigen::VectorXd::Map(&pubed.ForceTorque[0],6)=force_measured;
//     pub_end_force_torque.publish(pubed);
//     desired_force << 0, 0, ext_force-clap_gravity, 0, 0, 0;
//     desired_force_error << 0, 0, ext_force-clap_gravity+force_measured(2,0), 0, 0, 0;
//     // ROS_INFO("force error:%lf",desired_force_error(2,0));
//     diff_desired_force_error=desired_force_error-desired_force_error_last;
//     inte_force_error=inte_force_error+desired_force_error;
//     desired_force_error_last=desired_force_error;
//     //                                                            feedforward      P                        I                                   D
//     tau_d_calculated_vector = jacobian.transpose() * fenkuan_E2B*(desired_force+force_PID_P*desired_force_error+force_PID_I *inte_force_error+force_PID_D*diff_desired_force_error);
//   // Desired torque

//   tau_d << tau_task  + coriolis+tau_d_calculated_vector;
//   }
//   else
//   {
//     tau_d << tau_task  + coriolis;
//   }
//   //tau_d << tau_task  + coriolis;
//   // Saturate torque rate to avoid discontinuities
//   tau_d << saturateTorqueRate(tau_d, tau_J_d);
//   for (size_t i = 0; i < 7; ++i) {
//     joint_handles_[i].setCommand(tau_d(i));
//   }




//   // update parameters changed online either through dynamic reconfigure or through the interactive
//   // target by filtering
//   cartesian_stiffness_ =
//       filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
//   cartesian_damping_ =
//       filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
//   nullspace_stiffness_ =
//       filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
//   std::lock_guard<std::mutex> position_d_target_mutex_lock(
//       position_and_orientation_d_target_mutex_);
//   position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
//   orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
// }

// Eigen::Matrix<double, 7, 1> cartesian_position_force_hybird_controller::saturateTorqueRate(
//     const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
//     const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
//   Eigen::Matrix<double, 7, 1> tau_d_saturated{};
//   for (size_t i = 0; i < 7; i++) {
//     double difference = tau_d_calculated[i] - tau_J_d[i];
//     tau_d_saturated[i] =
//         tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
//   }
//   return tau_d_saturated;

}


void cartesian_position_force_hybird_controller::stopping(const ros::Time& time)
{
  //calcu_invers=false;
}


void cartesian_position_force_hybird_controller::complianceParamCallback(
    franka_example_controllers::compliance_paramConfig& config,
    uint32_t /*level*/) {
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_.topLeftCorner(3, 3)
      << config.translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3)
      << config.rotational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.setIdentity();
  // Damping ratio = 1
  cartesian_damping_target_.topLeftCorner(3, 3)
      << 2.0 * sqrt(config.translational_stiffness) * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.bottomRightCorner(3, 3)
      << 2.0 * sqrt(config.rotational_stiffness) * Eigen::Matrix3d::Identity();
  nullspace_stiffness_target_ = config.nullspace_stiffness;
}


void cartesian_position_force_hybird_controller::PositionForceParamCallback(franka_example_controllers::cartesian_position_force_paramConfig &config,uint32_t level)
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
    
  if(is_contact_scanning==0){
    ROS_INFO("cartesian");
  }
  else if(is_contact_scanning==1){
    ROS_INFO("imdepance");
  }
  else if(is_contact_scanning==2){
    ROS_INFO("hybird");
  }
}


void cartesian_position_force_hybird_controller::equilibriumPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  // std::lock_guard<std::mutex> position_d_target_mutex_lock(
  //     position_and_orientation_d_target_mutex_);

  // Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  // if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
  //   orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  // }

  // //位置控制
  franka::RobotState initial_state = cartesian_pose_handle_->getRobotState().O_T_EE_c;

  elapsed_time_ = ros::Duration(0.0);

  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.data()));
  position_d_ = initial_transform.translation();
  orientation_d_ = initial_transform.rotation();

  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;

  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  moving_time_  =msg->header.stamp.toSec();
  ROS_INFO("moving_time_:%lf",moving_time_);

    ROS_INFO("start_position_x:%lf",position_d_target_[0]);
  ROS_INFO("start_position_y:%lf",position_d_target_[1]);
  ROS_INFO("start_position_z:%lf",position_d_target_[2]);
  ROS_INFO("start_position_rx:%lf",orientation_d_target_.coeffs()[0]);
  ROS_INFO("start_position_ry:%lf",orientation_d_target_.coeffs()[1]);
  ROS_INFO("start_position_rz:%lf",orientation_d_target_.coeffs()[2]);
  ROS_INFO("start_position_rw:%lf",orientation_d_target_.coeffs()[3]);
}

// ----------------------------路径点插值
void cartesian_position_force_hybird_controller::fifth_intP(std::array<double, 6>& parameter, double start, double end, double timelen) {
	//五次线性插值得到6个系数
    parameter[0] = start;
    parameter[1] = 0;
    parameter[2] = 0;
    parameter[3] = 10 * (end - start) / pow(timelen, 3);
    parameter[4] = -15 * (end - start) / pow(timelen, 4);
    parameter[5] = 6 * (end - start) / pow(timelen, 5);
}

double cartesian_position_force_hybird_controller::getintP(std::array<double, 6> parameter, double time) {
	//通过六个系数计算time时的值
    return parameter[0] + parameter[1] *time + parameter[2] * pow(time, 2) +parameter[3] * pow(time, 3) + parameter[4] * pow(time, 4) +
        parameter[5] * pow(time, 5);
}

double cartesian_position_force_hybird_controller::Funtion_t(double x) {
//平滑时间系数
    return -20 * pow(x, 7) + 70 * pow(x, 6) - 84 * pow(x, 5) + 35 * pow(x, 4);
}
}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::cartesian_position_force_hybird_controller,
                       controller_interface::ControllerBase)
