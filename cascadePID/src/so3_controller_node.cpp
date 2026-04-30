/*
 * so3_controller_node.cpp
 *
 * Geometric tracking controller on SE(3) for MARSIM quadrotor simulation.
 * Reference: T. Lee, M. Leok, N. H. McClamroch, "Geometric tracking control
 *            of a quadrotor UAV on SE(3)", CDC 2010.
 *
 * Drop-in replacement for cascadePID_node. Same topic contract:
 *   Sub: ~odom           (nav_msgs/Odometry)
 *   Sub: ~position_cmd   (quadrotor_msgs/PositionCommand)
 *   Pub: ~cmd_RPM        (std_msgs/Float32MultiArray, 4 elements)
 *
 * Default gains are ballpark values for a 1.5–2 kg quadrotor; tune for your
 * specific vehicle. All gains are exposed as rosparams:
 *   k_x          [7, 7, 11]   — position error gain (N/m)
 *   k_v          [4, 4, 7]    — velocity error gain (N·s/m)
 *   k_R          [1.5, 1.5, 0.5]  — attitude error gain (N·m/rad)
 *   k_Omega      [0.3, 0.3, 0.15] — angular velocity error gain (N·m·s/rad)
 *   mass         1.5          — vehicle mass (kg)
 *   inertia_diag [0.03, 0.03, 0.06] — diagonal inertia tensor (kg·m^2)
 *
 * Motor model (matches quadrotor_dynamics.hpp, hardcoded constants):
 *   k_F = 3 * 8.98132e-9   thrust coefficient (N / RPM^2)
 *   k_T = 0.07 * 3*0.062 * k_F  drag coefficient
 *   arm_length = 0.22 m, X-type layout, min_rpm = 0, max_rpm = 35000
 *
 * The controller rate defaults to 200 Hz matching cascadePID_node.
 */

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <std_msgs/Float32MultiArray.h>

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>

static constexpr double G_ACC    = 9.81;
static constexpr double MARSIM_K_F = 3.0 * 8.98132e-9;
static constexpr double MARSIM_K_T = 0.07 * (3.0 * 0.062) * MARSIM_K_F;
static constexpr double MARSIM_ARM = 0.22;
static constexpr double PREDICTNAV_HUMMINGBIRD_K_F = 8.54858e-06;
static constexpr double PREDICTNAV_HUMMINGBIRD_K_T = 1.3677728816219314e-07;
static constexpr double PREDICTNAV_HUMMINGBIRD_ARM = 0.17;
static constexpr double MIN_RPM_DEFAULT = 0.0;
static constexpr double MARSIM_MAX_RPM = 35000.0;
static constexpr double PREDICTNAV_HUMMINGBIRD_MAX_RPM = 838.0;

// ---------- state ----------
static nav_msgs::Odometry g_odom;
static bool g_odom_valid = false;

static Eigen::Vector3d g_pos_des{0, 0, 1};
static Eigen::Vector3d g_vel_des{0, 0, 0};
static Eigen::Vector3d g_acc_des{0, 0, 0};
static double          g_yaw_des = 0.0;

// ---------- parameters (set in main) ----------
static Eigen::Vector3d g_k_x{7, 7, 11};
static Eigen::Vector3d g_k_v{4, 4, 7};
static Eigen::Vector3d g_k_R{1.5, 1.5, 0.5};
static Eigen::Vector3d g_k_Omega{0.3, 0.3, 0.15};
static double          g_mass = 1.5;
static Eigen::Matrix3d g_J    = Eigen::Matrix3d::Zero();
static double          g_motor_force_constant = MARSIM_K_F;
static double          g_motor_moment_constant = MARSIM_K_T;
static double          g_motor_arm_length = MARSIM_ARM;
static double          g_min_rpm = MIN_RPM_DEFAULT;
static double          g_max_rpm = MARSIM_MAX_RPM;
static bool            g_predictnav_hummingbird_layout = false;

static ros::Publisher g_rpm_pub;

// Skew-symmetric matrix from a 3-vector
static Eigen::Matrix3d hat(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S <<   0, -v(2),  v(1),
        v(2),     0, -v(0),
       -v(1),  v(0),     0;
    return S;
}

// vee operator: extract axial vector from skew-symmetric matrix
static Eigen::Vector3d vee(const Eigen::Matrix3d& M)
{
    return Eigen::Vector3d(M(2, 1), M(0, 2), M(1, 0));
}

// Build the 4x4 mixer matrix (RPM^2 → [thrust; tau_x; tau_y; tau_z])
// matching cascadePID.hpp::setInternal() exactly.
static Eigen::Matrix4d buildMixerInv()
{
    if (g_predictnav_hummingbird_layout)
    {
        Eigen::Matrix4d M;
        M.row(0) << g_motor_force_constant, g_motor_force_constant, g_motor_force_constant, g_motor_force_constant;
        M.row(1) << 0.0, g_motor_force_constant * g_motor_arm_length, 0.0, -g_motor_force_constant * g_motor_arm_length;
        M.row(2) << -g_motor_force_constant * g_motor_arm_length, 0.0, g_motor_force_constant * g_motor_arm_length, 0.0;
        M.row(3) << g_motor_moment_constant, -g_motor_moment_constant, g_motor_moment_constant, -g_motor_moment_constant;
        return M;
    }

    const double L  = g_motor_arm_length * std::sqrt(2.0) / 2.0;
    // Motor positions (X-type):
    //  0: (+L, -L, 0)  CCW  → -k_T yaw sign
    //  1: (-L, +L, 0)  CCW  → -k_T yaw sign
    //  2: (+L, +L, 0)  CW   → +k_T yaw sign
    //  3: (-L, -L, 0)  CW   → +k_T yaw sign
    Eigen::Matrix4d M;
    //                 m0    m1    m2    m3
    M.row(0) <<  g_motor_force_constant,  g_motor_force_constant,  g_motor_force_constant,  g_motor_force_constant;          // total thrust coef
    M.row(1) <<  g_motor_force_constant*(-L), g_motor_force_constant*( L), g_motor_force_constant*( L), g_motor_force_constant*(-L);  // roll (tau_x)
    M.row(2) <<  g_motor_force_constant*(-L), g_motor_force_constant*( L), g_motor_force_constant*(-L), g_motor_force_constant*( L);  // pitch (tau_y)
    M.row(3) << -g_motor_moment_constant,     -g_motor_moment_constant,      g_motor_moment_constant,       g_motor_moment_constant;       // yaw (tau_z)
    return M;
}

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    g_odom       = *msg;
    g_odom_valid = true;
}

void posCmdCallback(const quadrotor_msgs::PositionCommand::ConstPtr& msg)
{
    g_pos_des = Eigen::Vector3d(msg->position.x, msg->position.y, msg->position.z);
    g_vel_des = Eigen::Vector3d(msg->velocity.x, msg->velocity.y, msg->velocity.z);
    g_acc_des = Eigen::Vector3d(msg->acceleration.x, msg->acceleration.y, msg->acceleration.z);
    g_yaw_des = msg->yaw;
}

void controlLoop(const ros::TimerEvent&)
{
    if (!g_odom_valid) return;

    // --- Current state ---
    const auto& p_odom = g_odom.pose.pose;
    const auto& t_odom = g_odom.twist.twist;

    Eigen::Vector3d pos(p_odom.position.x, p_odom.position.y, p_odom.position.z);
    Eigen::Vector3d vel(t_odom.linear.x,  t_odom.linear.y,  t_odom.linear.z);
    Eigen::Quaterniond q(p_odom.orientation.w, p_odom.orientation.x,
                         p_odom.orientation.y, p_odom.orientation.z);
    q.normalize();
    Eigen::Matrix3d R = q.toRotationMatrix();

    // Body angular velocity comes in the odometry twist in body frame for MARSIM
    Eigen::Vector3d Omega(t_odom.angular.x, t_odom.angular.y, t_odom.angular.z);

    // --- Position and velocity errors ---
    Eigen::Vector3d e_x = pos - g_pos_des;
    Eigen::Vector3d e_v = vel - g_vel_des;

    // --- Desired force in world frame (Lee 2010, eq. 16) ---
    const Eigen::Vector3d e3(0, 0, 1);
    Eigen::Vector3d F_des = -(g_k_x.asDiagonal() * e_x)
                            - (g_k_v.asDiagonal() * e_v)
                            + g_mass * G_ACC * e3
                            + g_mass * g_acc_des;

    // --- Desired total thrust (projection onto current body z) ---
    double f_total = F_des.dot(R * e3);
    // Clamp to physically meaningful range
    f_total = std::max(0.0, std::min(f_total, g_mass * 4.0 * G_ACC));

    // --- Desired attitude (Lee 2010, eq. 7) ---
    Eigen::Vector3d b3_des = F_des.normalized();
    if (F_des.norm() < 1e-6) b3_des = e3;

    // b1_des from desired yaw
    Eigen::Vector3d b1_yaw(std::cos(g_yaw_des), std::sin(g_yaw_des), 0.0);
    Eigen::Vector3d b2_des = b3_des.cross(b1_yaw);
    if (b2_des.norm() < 1e-6) b2_des = Eigen::Vector3d(0, 1, 0);
    b2_des.normalize();
    Eigen::Vector3d b1_des = b2_des.cross(b3_des);
    b1_des.normalize();

    Eigen::Matrix3d R_des;
    R_des.col(0) = b1_des;
    R_des.col(1) = b2_des;
    R_des.col(2) = b3_des;

    // --- Attitude and angular velocity errors (Lee 2010, eq. 10-11) ---
    Eigen::Matrix3d eR_mat = 0.5 * (R_des.transpose() * R - R.transpose() * R_des);
    Eigen::Vector3d e_R    = vee(eR_mat);
    Eigen::Vector3d e_Omega = Omega;  // desired angular velocity = 0 for hover/tracking

    // --- Desired torque (Lee 2010, eq. 15) ---
    Eigen::Vector3d tau = -(g_k_R.asDiagonal() * e_R)
                          - (g_k_Omega.asDiagonal() * e_Omega)
                          + Omega.cross(g_J * Omega);

    // --- Motor allocation ---
    // Solve [F, tau_x, tau_y, tau_z]^T = Mixer * [n0^2, n1^2, n2^2, n3^2]^T
    static const Eigen::Matrix4d Mixer    = buildMixerInv();
    static const Eigen::Matrix4d MixerInv = Mixer.inverse();

    Eigen::Vector4d wrench(f_total, tau(0), tau(1), tau(2));
    Eigen::Vector4d rpm_sq = MixerInv * wrench;

    std_msgs::Float32MultiArray rpm_msg;
    rpm_msg.data.resize(4);
    for (int i = 0; i < 4; ++i)
    {
        double n = (rpm_sq(i) > 0.0) ? std::sqrt(rpm_sq(i)) : 0.0;
        n = std::max(g_min_rpm, std::min(g_max_rpm, n));
        rpm_msg.data[i] = static_cast<float>(n);
    }

    g_rpm_pub.publish(rpm_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "so3_controller");
    ros::NodeHandle nh("~");

    // --- Gains ---
    std::vector<double> k_x_vec, k_v_vec, k_R_vec, k_Omega_vec, inertia_vec;
    std::string vehicle_profile;
    nh.param("vehicle_profile", vehicle_profile, std::string("marsim_default"));
    g_predictnav_hummingbird_layout = (vehicle_profile == "predictnav_hummingbird");

    nh.param("mass", g_mass, g_predictnav_hummingbird_layout ? 0.716 : 1.5);
    nh.param("motor_arm_length", g_motor_arm_length, g_predictnav_hummingbird_layout ? PREDICTNAV_HUMMINGBIRD_ARM : MARSIM_ARM);
    nh.param("motor_force_constant", g_motor_force_constant, g_predictnav_hummingbird_layout ? PREDICTNAV_HUMMINGBIRD_K_F : MARSIM_K_F);
    nh.param("motor_moment_constant", g_motor_moment_constant, g_predictnav_hummingbird_layout ? PREDICTNAV_HUMMINGBIRD_K_T : MARSIM_K_T);
    nh.param("min_rpm", g_min_rpm, MIN_RPM_DEFAULT);
    nh.param("max_rpm", g_max_rpm, g_predictnav_hummingbird_layout ? PREDICTNAV_HUMMINGBIRD_MAX_RPM : MARSIM_MAX_RPM);

    if (nh.getParam("k_x", k_x_vec) && k_x_vec.size() == 3)
        g_k_x = Eigen::Map<Eigen::Vector3d>(k_x_vec.data());
    else
        g_k_x = Eigen::Vector3d(7.0, 7.0, 11.0);

    if (nh.getParam("k_v", k_v_vec) && k_v_vec.size() == 3)
        g_k_v = Eigen::Map<Eigen::Vector3d>(k_v_vec.data());
    else
        g_k_v = Eigen::Vector3d(4.0, 4.0, 7.0);

    if (nh.getParam("k_R", k_R_vec) && k_R_vec.size() == 3)
        g_k_R = Eigen::Map<Eigen::Vector3d>(k_R_vec.data());
    else
        g_k_R = Eigen::Vector3d(1.5, 1.5, 0.5);

    if (nh.getParam("k_Omega", k_Omega_vec) && k_Omega_vec.size() == 3)
        g_k_Omega = Eigen::Map<Eigen::Vector3d>(k_Omega_vec.data());
    else
        g_k_Omega = Eigen::Vector3d(0.3, 0.3, 0.15);

    if (nh.getParam("inertia_diag", inertia_vec) && inertia_vec.size() == 3)
        g_J = Eigen::DiagonalMatrix<double, 3>(inertia_vec[0], inertia_vec[1], inertia_vec[2]);
    else
        g_J = Eigen::DiagonalMatrix<double, 3>(0.03, 0.03, 0.06);

    // --- Initial state ---
    double init_x, init_y, init_z;
    nh.param("init_state_x", init_x, 0.0);
    nh.param("init_state_y", init_y, 0.0);
    nh.param("init_state_z", init_z, 1.0);
    g_pos_des = Eigen::Vector3d(init_x, init_y, init_z);

    double controller_rate;
    nh.param("controller_rate", controller_rate, 200.0);

    // --- Publishers and subscribers ---
    g_rpm_pub = nh.advertise<std_msgs::Float32MultiArray>("cmd_RPM", 100);

    ros::Subscriber odom_sub = nh.subscribe("odom", 100, odomCallback,
                                            ros::TransportHints().tcpNoDelay());
    ros::Subscriber cmd_sub  = nh.subscribe("position_cmd", 10, posCmdCallback,
                                            ros::TransportHints().tcpNoDelay());

    ros::Timer timer = nh.createTimer(ros::Duration(1.0 / controller_rate), controlLoop);

    ROS_INFO(
        "[so3_controller] started: vehicle_profile=%s mass=%.3f kg, rate=%.0f Hz, arm=%.3f kF=%.9g kM=%.9g max_rpm=%.1f",
        vehicle_profile.c_str(),
        g_mass,
        controller_rate,
        g_motor_arm_length,
        g_motor_force_constant,
        g_motor_moment_constant,
        g_max_rpm
    );

    ros::spin();
    return 0;
}
