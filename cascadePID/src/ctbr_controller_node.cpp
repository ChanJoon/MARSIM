#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <std_msgs/Float32MultiArray.h>

#include <eigen3/Eigen/Dense>
#include <cmath>

static constexpr double G_ACC   = 9.81;
static constexpr double K_F     = 3.0 * 8.98132e-9;
static constexpr double K_T     = 0.07 * (3.0 * 0.062) * K_F;
static constexpr double ARM     = 0.22;
static constexpr double MIN_RPM = 0.0;
static constexpr double MAX_RPM = 35000.0;

static nav_msgs::Odometry g_odom;
static bool g_odom_valid = false;

static quadrotor_msgs::PositionCommand g_ctbr_cmd;
static bool g_ctbr_valid = false;
static ros::Time g_ctbr_stamp;

static Eigen::Vector3d g_hold_pos{0, 0, 1};
static double g_hold_yaw = 0.0;
static bool g_hold_valid = false;
static bool g_prev_ctbr_active = false;

static Eigen::Vector3d g_k_x{7, 7, 11};
static Eigen::Vector3d g_k_v{4, 4, 7};
static Eigen::Vector3d g_k_R{1.5, 1.5, 0.5};
static Eigen::Vector3d g_k_Omega{0.3, 0.3, 0.15};
static double g_mass = 1.9;
static double g_ctbr_timeout = 0.15;
static Eigen::Matrix3d g_J = Eigen::Matrix3d::Zero();

static ros::Publisher g_rpm_pub;

static Eigen::Vector3d vee(const Eigen::Matrix3d& M)
{
    return Eigen::Vector3d(M(2, 1), M(0, 2), M(1, 0));
}

static Eigen::Matrix4d buildMixer()
{
    const double L = ARM * std::sqrt(2.0) / 2.0;
    Eigen::Matrix4d M;
    M.row(0) << K_F, K_F, K_F, K_F;
    M.row(1) << K_F * (-L), K_F * (L), K_F * (L), K_F * (-L);
    M.row(2) << K_F * (L), K_F * (-L), K_F * (L), K_F * (-L);
    M.row(3) << -K_T, -K_T, K_T, K_T;
    return M;
}

static double yawFromQuat(const geometry_msgs::Quaternion& q_msg)
{
    Eigen::Quaterniond q(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
    q.normalize();
    Eigen::Matrix3d R = q.toRotationMatrix();
    return std::atan2(R(1, 0), R(0, 0));
}

static void latchHoldTarget()
{
    const auto& pose = g_odom.pose.pose;
    g_hold_pos = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
    g_hold_yaw = yawFromQuat(pose.orientation);
    g_hold_valid = true;
}

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    g_odom = *msg;
    g_odom_valid = true;
    if (!g_hold_valid) latchHoldTarget();
}

void ctbrCallback(const quadrotor_msgs::PositionCommand::ConstPtr& msg)
{
    g_ctbr_cmd = *msg;
    g_ctbr_valid = true;
    g_ctbr_stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
}

static void publishRpm(double f_total, const Eigen::Vector3d& tau)
{
    static const Eigen::Matrix4d Mixer = buildMixer();
    static const Eigen::Matrix4d MixerInv = Mixer.inverse();

    const double f_clamped = std::max(0.0, std::min(f_total, g_mass * 4.0 * G_ACC));
    Eigen::Vector4d wrench(f_clamped, tau(0), tau(1), tau(2));
    Eigen::Vector4d rpm_sq = MixerInv * wrench;

    std_msgs::Float32MultiArray rpm_msg;
    rpm_msg.data.resize(4);
    for (int i = 0; i < 4; ++i)
    {
        double n = rpm_sq(i) > 0.0 ? std::sqrt(rpm_sq(i)) : 0.0;
        n = std::max(MIN_RPM, std::min(MAX_RPM, n));
        rpm_msg.data[i] = static_cast<float>(n);
    }
    g_rpm_pub.publish(rpm_msg);
}

static void runHoldController(const Eigen::Vector3d& pos,
                              const Eigen::Vector3d& vel,
                              const Eigen::Matrix3d& R,
                              const Eigen::Vector3d& omega)
{
    if (!g_hold_valid) latchHoldTarget();

    const Eigen::Vector3d e3(0, 0, 1);
    const Eigen::Vector3d e_x = pos - g_hold_pos;
    const Eigen::Vector3d e_v = vel;
    Eigen::Vector3d F_des = -(g_k_x.asDiagonal() * e_x)
                          - (g_k_v.asDiagonal() * e_v)
                          + g_mass * G_ACC * e3;

    double f_total = F_des.dot(R * e3);
    f_total = std::max(0.0, std::min(f_total, g_mass * 4.0 * G_ACC));

    Eigen::Vector3d b3_des = F_des.norm() < 1e-6 ? e3 : F_des.normalized();
    Eigen::Vector3d b1_yaw(std::cos(g_hold_yaw), std::sin(g_hold_yaw), 0.0);
    Eigen::Vector3d b2_des = b3_des.cross(b1_yaw);
    if (b2_des.norm() < 1e-6) b2_des = Eigen::Vector3d(0, 1, 0);
    b2_des.normalize();
    Eigen::Vector3d b1_des = b2_des.cross(b3_des);
    b1_des.normalize();

    Eigen::Matrix3d R_des;
    R_des.col(0) = b1_des;
    R_des.col(1) = b2_des;
    R_des.col(2) = b3_des;

    const Eigen::Matrix3d eR_mat = 0.5 * (R_des.transpose() * R - R.transpose() * R_des);
    const Eigen::Vector3d e_R = vee(eR_mat);
    const Eigen::Vector3d e_Omega = omega;
    const Eigen::Vector3d tau = -(g_k_R.asDiagonal() * e_R)
                              - (g_k_Omega.asDiagonal() * e_Omega)
                              + omega.cross(g_J * omega);

    publishRpm(f_total, tau);
}

static void runCtbrController(const Eigen::Vector3d& omega,
                              const quadrotor_msgs::PositionCommand& cmd)
{
    const Eigen::Vector3d omega_des(cmd.angular_velocity.x, cmd.angular_velocity.y, cmd.angular_velocity.z);
    const double f_total = cmd.thrust.z;
    const Eigen::Vector3d omega_err = omega - omega_des;
    const Eigen::Vector3d tau = -(g_k_Omega.asDiagonal() * omega_err)
                              + omega.cross(g_J * omega);
    publishRpm(f_total, tau);
}

void controlLoop(const ros::TimerEvent&)
{
    if (!g_odom_valid) return;

    const auto& pose = g_odom.pose.pose;
    const auto& twist = g_odom.twist.twist;

    Eigen::Vector3d pos(pose.position.x, pose.position.y, pose.position.z);
    Eigen::Vector3d vel(twist.linear.x, twist.linear.y, twist.linear.z);
    Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    q.normalize();
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Vector3d omega(twist.angular.x, twist.angular.y, twist.angular.z);

    const ros::Time now = ros::Time::now();
    const bool ctbr_active = g_ctbr_valid && ((now - g_ctbr_stamp).toSec() <= g_ctbr_timeout);

    if (!ctbr_active && g_prev_ctbr_active) latchHoldTarget();
    if (!ctbr_active)
    {
        runHoldController(pos, vel, R, omega);
        ROS_INFO_THROTTLE(1.0, "[ctbr_controller] hold mode active");
    }
    else
    {
        runCtbrController(omega, g_ctbr_cmd);
        ROS_INFO_THROTTLE(1.0, "[ctbr_controller] ctbr mode active");
    }

    g_prev_ctbr_active = ctbr_active;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ctbr_controller");
    ros::NodeHandle nh("~");

    std::vector<double> k_x_vec, k_v_vec, k_R_vec, k_Omega_vec, inertia_vec;
    nh.param("mass", g_mass, 1.9);
    nh.param("ctbr_timeout", g_ctbr_timeout, 0.15);

    if (nh.getParam("k_x", k_x_vec) && k_x_vec.size() == 3)
        g_k_x = Eigen::Map<Eigen::Vector3d>(k_x_vec.data());
    if (nh.getParam("k_v", k_v_vec) && k_v_vec.size() == 3)
        g_k_v = Eigen::Map<Eigen::Vector3d>(k_v_vec.data());
    if (nh.getParam("k_R", k_R_vec) && k_R_vec.size() == 3)
        g_k_R = Eigen::Map<Eigen::Vector3d>(k_R_vec.data());
    if (nh.getParam("k_Omega", k_Omega_vec) && k_Omega_vec.size() == 3)
        g_k_Omega = Eigen::Map<Eigen::Vector3d>(k_Omega_vec.data());
    if (nh.getParam("inertia_diag", inertia_vec) && inertia_vec.size() == 3)
        g_J = Eigen::DiagonalMatrix<double, 3>(inertia_vec[0], inertia_vec[1], inertia_vec[2]);
    else
        g_J = Eigen::DiagonalMatrix<double, 3>(0.03, 0.03, 0.06);

    double init_x, init_y, init_z;
    nh.param("init_state_x", init_x, 0.0);
    nh.param("init_state_y", init_y, 0.0);
    nh.param("init_state_z", init_z, 1.0);
    g_hold_pos = Eigen::Vector3d(init_x, init_y, init_z);

    double controller_rate;
    nh.param("controller_rate", controller_rate, 200.0);

    g_rpm_pub = nh.advertise<std_msgs::Float32MultiArray>("cmd_RPM", 100);
    ros::Subscriber odom_sub = nh.subscribe("odom", 100, odomCallback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber ctbr_sub = nh.subscribe("ctbr_cmd", 10, ctbrCallback, ros::TransportHints().tcpNoDelay());
    ros::Timer timer = nh.createTimer(ros::Duration(1.0 / controller_rate), controlLoop);

    ROS_INFO("[ctbr_controller] started: mass=%.2f kg, rate=%.0f Hz, timeout=%.2f s", g_mass, controller_rate, g_ctbr_timeout);
    ros::spin();
    return 0;
}
