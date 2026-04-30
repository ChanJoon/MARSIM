
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <quadrotor_dynamics.hpp>
#include <std_msgs/Float32MultiArray.h>

using namespace std;
using namespace Eigen;

namespace
{
constexpr double kMaxReasonableDt = 0.5;
constexpr double kQuaternionNormWarnEps = 1e-9;

bool vectorAllFinite(const Vector3d& value)
{
    return std::isfinite(value(0)) && std::isfinite(value(1)) && std::isfinite(value(2));
}

bool vectorAllFinite(const Vector4d& value)
{
    return std::isfinite(value(0)) && std::isfinite(value(1)) && std::isfinite(value(2)) && std::isfinite(value(3));
}

std::vector<double> paramVectorOrDefault(ros::NodeHandle& n, const std::string& name, const std::vector<double>& fallback)
{
    std::vector<double> value;
    if(n.getParam(name, value) && value.size() == fallback.size())
    {
        return value;
    }
    return fallback;
}

MatrixXd marsimDefaultMotorPositions(double arm_length)
{
    const double l = arm_length * std::sqrt(2.0) / 2.0;
    MatrixXd motor_pos(4, 3);
    motor_pos <<  l, -l, 0,
                 -l,  l, 0,
                  l,  l, 0,
                 -l, -l, 0;
    return motor_pos;
}

MatrixXd predictnavHummingbirdMotorPositions(double arm_length)
{
    MatrixXd motor_pos(4, 3);
    motor_pos <<  arm_length, 0.0, 0.0,
                  0.0, arm_length, 0.0,
                 -arm_length, 0.0, 0.0,
                  0.0, -arm_length, 0.0;
    return motor_pos;
}
}

std_msgs::Float32MultiArray RPM_msg;
Vector4d RPM_input = Vector4d::Zero();
bool has_valid_rpm_input = false;
void RPMCallbck(const std_msgs::Float32MultiArray& msg)
{
    RPM_msg = msg;
    if (RPM_msg.data.size() != 4)
    {
        ROS_WARN_THROTTLE(1.0, "cmd_RPM length %zu != 4; zeroing input", RPM_msg.data.size());
        RPM_input = Vector4d::Zero();
        has_valid_rpm_input = false;
        return;
    }

    Vector4d candidate;
    for(int i = 0; i < 4; ++i)
    {
        candidate(i) = RPM_msg.data[i];
    }
    if(!vectorAllFinite(candidate))
    {
        ROS_WARN_THROTTLE(1.0, "cmd_RPM contained non-finite values [%.3f, %.3f, %.3f, %.3f]; zeroing input", candidate(0), candidate(1), candidate(2), candidate(3));
        RPM_input = Vector4d::Zero();
        has_valid_rpm_input = false;
        return;
    }

    RPM_input = candidate;
    has_valid_rpm_input = true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "quadrotor_dynamics");

    ros::NodeHandle n("~");


    double init_x, init_y, init_z,mass;
    double simulation_rate;
    double motor_arm_length, motor_force_constant, motor_moment_constant, min_rpm, max_rpm;
    std::string quad_name;
    std::string vehicle_profile;
    n.param("vehicle_profile", vehicle_profile, std::string("marsim_default"));
    const bool use_predictnav_hummingbird = (vehicle_profile == "predictnav_hummingbird");
    const double default_mass = use_predictnav_hummingbird ? 0.716 : 0.9;
    n.param("mass", mass, default_mass);
    n.param("init_state_x", init_x, 0.0);
    n.param("init_state_y", init_y, 0.0);
    n.param("init_state_z", init_z, 1.0);
    n.param("simulation_rate", simulation_rate, 200.0);
    n.param("quadrotor_name", quad_name, std::string("quadrotor"));
    n.param("motor_arm_length", motor_arm_length, use_predictnav_hummingbird ? 0.17 : 0.22);
    n.param("motor_force_constant", motor_force_constant, use_predictnav_hummingbird ? 8.54858e-06 : 3.0 * 8.98132e-9);
    n.param("motor_moment_constant", motor_moment_constant, use_predictnav_hummingbird ? 1.3677728816219314e-07 : 0.07 * (3.0 * 0.062) * (3.0 * 8.98132e-9));
    n.param("min_rpm", min_rpm, 0.0);
    n.param("max_rpm", max_rpm, use_predictnav_hummingbird ? 838.0 : 35000.0);

    ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>("odom", 100);
    ros::Publisher imu_pub = n.advertise<sensor_msgs::Imu>("imu", 10);
    // ros::Subscriber cmd_sub = n.subscribe("cmd", 100, &cmd_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber rpm_sub = n.subscribe("cmd_RPM", 100, RPMCallbck);

    Matrix3d Internal_mat;
    const std::vector<double> inertia_fallback = use_predictnav_hummingbird
        ? std::vector<double>{0.007, 0.007, 0.012}
        : std::vector<double>{2.64e-3, 2.64e-3, 4.96e-3};
    const std::vector<double> inertia_diag = paramVectorOrDefault(n, "inertia_diag", inertia_fallback);
    Internal_mat << inertia_diag[0],0,0,
                    0,inertia_diag[1],0,
                    0,0,inertia_diag[2];
    quadrotor_dynamics quadrotor(mass, Internal_mat);

    Vector3d init_pos;
    Vector4d init_q, actuator;
    init_pos << init_x, init_y, init_z;
    init_q << 1,0,0.0,0.0;
    quadrotor.init(init_pos , init_q);
    MatrixXd motor_pos;
    Vector4d yaw_torque_coeff;
    if(use_predictnav_hummingbird)
    {
        motor_pos = predictnavHummingbirdMotorPositions(motor_arm_length);
        yaw_torque_coeff << motor_moment_constant, -motor_moment_constant, motor_moment_constant, -motor_moment_constant;
    }
    else
    {
        motor_pos = marsimDefaultMotorPositions(motor_arm_length);
        yaw_torque_coeff << -motor_moment_constant, -motor_moment_constant, motor_moment_constant, motor_moment_constant;
    }
    quadrotor.configureMotorModel(motor_pos, yaw_torque_coeff, motor_force_constant, min_rpm, max_rpm);
    ROS_INFO(
        "[quadrotor_dynamics] vehicle_profile=%s mass=%.3f inertia=[%.6f %.6f %.6f] arm=%.3f kF=%.9g kM=%.9g max_rpm=%.1f",
        vehicle_profile.c_str(),
        mass,
        inertia_diag[0],
        inertia_diag[1],
        inertia_diag[2],
        motor_arm_length,
        motor_force_constant,
        motor_moment_constant,
        max_rpm
    );
    actuator << 0.0,0.0,0.0,0.0;
    // quadrotor.setActuatoroutput(actuator);

    ros::Rate rate(simulation_rate);
    rate.sleep();

    ros::Time last_time = ros::Time::now();
    while(n.ok())
    {
        ros::spinOnce();
        if (!has_valid_rpm_input)
        {
            RPM_input = Vector4d::Zero();
        }
        if(!vectorAllFinite(RPM_input))
        {
            ROS_WARN_THROTTLE(1.0, "RPM_input became non-finite [%.3f, %.3f, %.3f, %.3f]; zeroing before integration", RPM_input(0), RPM_input(1), RPM_input(2), RPM_input(3));
            RPM_input = Vector4d::Zero();
            has_valid_rpm_input = false;
        }
        quadrotor.setRPM(RPM_input);

        ros::Time now_time = ros::Time::now();
        const double dt = (now_time-last_time).toSec();
        if(!std::isfinite(dt) || dt <= 0.0 || dt > kMaxReasonableDt)
        {
            ROS_WARN_THROTTLE(1.0, "Skipping dynamics step with invalid dt=%.6f", dt);
            last_time = now_time;
            rate.sleep();
            continue;
        }
        quadrotor.step_forward(dt);
        last_time = now_time;

        //publish odometry
        nav_msgs::Odometry odom;
        odom.header.frame_id = "world";
        odom.header.stamp = now_time;
        Vector3d pos,vel,acc,angular_vel;
        pos = quadrotor.getPos();
        vel = quadrotor.getVel();
        acc = quadrotor.getAcc();
        angular_vel = quadrotor.getAngularVel();
        Vector4d quat;
        quat = quadrotor.getQuat();
        Matrix3d R_body2world;
        R_body2world = quadrotor.getR();
        const double quat_norm = quat.norm();
        const bool invalid_odom_state = !vectorAllFinite(pos) || !vectorAllFinite(vel) || !vectorAllFinite(acc) || !vectorAllFinite(angular_vel) || !vectorAllFinite(quat) || !std::isfinite(quat_norm);
        if(invalid_odom_state)
        {
            ROS_ERROR_THROTTLE(1.0, "Publishing odom with non-finite state pos=[%.3f %.3f %.3f] vel=[%.3f %.3f %.3f] omega=[%.3f %.3f %.3f] quat=[%.6f %.6f %.6f %.6f]", pos(0), pos(1), pos(2), vel(0), vel(1), vel(2), angular_vel(0), angular_vel(1), angular_vel(2), quat(0), quat(1), quat(2), quat(3));
        }
        else if(quat_norm <= kQuaternionNormWarnEps)
        {
            ROS_ERROR_THROTTLE(1.0, "Publishing odom with near-zero quaternion norm=%.3e quat=[%.6f %.6f %.6f %.6f]", quat_norm, quat(0), quat(1), quat(2), quat(3));
        }
        odom.pose.pose.position.x = pos(0);
        odom.pose.pose.position.y = pos(1);
        odom.pose.pose.position.z = pos(2);
        odom.pose.pose.orientation.w = quat(0);
        odom.pose.pose.orientation.x = quat(1);
        odom.pose.pose.orientation.y = quat(2);
        odom.pose.pose.orientation.z = quat(3);
        odom.twist.twist.linear.x = vel(0);
        odom.twist.twist.linear.y = vel(1);
        odom.twist.twist.linear.z = vel(2);
        odom.twist.twist.angular.x = angular_vel(0);
        odom.twist.twist.angular.y = angular_vel(1);
        odom.twist.twist.angular.z = angular_vel(2);
        odom_pub.publish(odom);

        //imu generate
        sensor_msgs::Imu imu_msg;
        imu_msg.header.frame_id = "/" + quad_name;
        imu_msg.header.stamp = now_time;
        imu_msg.orientation.w = quat(0);
        imu_msg.orientation.x = quat(1);
        imu_msg.orientation.y = quat(2);
        imu_msg.orientation.z = quat(3);
        imu_msg.angular_velocity.x = angular_vel(0);
        imu_msg.angular_velocity.y = angular_vel(1);
        imu_msg.angular_velocity.z = angular_vel(2);
        acc = R_body2world.inverse() * (acc + Eigen::Vector3d(0,0,-9.8));
        imu_msg.linear_acceleration.x = acc(0);
        imu_msg.linear_acceleration.y = acc(1);
        imu_msg.linear_acceleration.z = acc(2);
        imu_pub.publish(imu_msg);

        rate.sleep();
    }

    return 0;
  
}
