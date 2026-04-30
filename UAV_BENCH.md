# MARSIM — `uav-bench` fork

This branch (`ChanJoon/MARSIM@uav-bench`) is vendored as a submodule into the
[eval_workspace](https://github.com/ChanJoon/eval_workspace) benchmarking
repository at `third_party/MARSIM`. It rebases on top of
[`hku-mars/MARSIM@ubuntu20`](https://github.com/hku-mars/MARSIM).

## Compatibility statement

All additions are opt-in. With every new launch argument left at its default,
`roslaunch test_interface single_drone_avia.launch` reproduces the upstream
`ubuntu20` golden path byte-for-byte:
- static PCD map, CPU `pcl_render_node`, cascadePID controller, legacy polar
  LiDAR shader on the GPU renderer.

None of the upstream-authored files on the hot path are modified in a way that
changes default behaviour:
- `cascadePID/src/cascadePID_node.cpp` (unchanged)
- `mars_drone_sim/src/quadrotor_dynamics_node.cpp` (unchanged)
- `mars_drone_sim/include/dynamics.hpp`, `quadrotor_dynamics.hpp` (unchanged)
- `local_sensing/src/pointcloud_render_node.cpp` (unchanged)
- `local_sensing/include/360camera.vs` (unchanged; used when `sensor_type=lidar`)

## Extensions

| Branch | Feature | Opt-in knob | Default |
|--------|---------|-------------|---------|
| 1 | Conditional ground-plane injection in published global cloud | `use_ground_plane:=true` (+ `ground_resolution`, `ground_size_x/y`, `ground_z`) | off |
| 2 | Seeded random map generator (`random_map_node`) | `map_mode:=random map_seed:=<int> scene_type:=<0..7>` (+ `yopo_*` knobs) | `map_mode=static` |
| 3 | GPU pinhole depth camera on `opengl_render_node` with `sensor_msgs/Image 32FC1` + `CameraInfo (plumb_bob)` | `use_gpu_:=true sensor_type:=depth_pinhole` (+ `image_width/height`, `fx/fy/cx/cy`, `near_clip`, `far_clip`); shortcut `single_drone_depth_pinhole.launch` | `sensor_type=lidar` |
| 4 | Lee-2010 geometric SE(3) controller as a drop-in for cascadePID, same `/quad_0/cmdRPM` contract | `controller_type:=so3` | `controller_type=pid` |

### Branch 3 shader contract

- `local_sensing/include/camera.vs` / `camera.fs` are used only when
  `sensor_type=depth_pinhole`. They pass metric view-space depth (metres) as a
  varying to the fragment shader, which discards fragments outside
  `[near_clip, far_clip]` and writes depth into the red channel.
- The pinhole readback path reads `GL_RED`/`GL_FLOAT` directly, so there is no
  NDC linearisation step in `opengl_sim.hpp::read_depth_pinhole()`.
- The polar LiDAR path still uses `360camera.vs` unchanged.

### Branch 4 controller contract

- `so3_controller_node` subscribes `~odom` (`nav_msgs/Odometry`) and
  `~position_cmd` (`quadrotor_msgs/PositionCommand`) and publishes `~cmd_RPM`
  (`std_msgs/Float32MultiArray`, 4 motors) — identical topic types to
  `cascadePID_node`, so the dynamics node does not change.
- Default gains and mass in `single_drone.xml` are aligned with the dynamics
  node (`mass = 1.9`).

## Rebase / pin-bump policy

- One feature branch = one submodule pin bump in `eval_workspace`.
- Each bump is recorded in the eval_workspace `CHANGELOG.md` under a dated
  section that names the branch merged into `uav-bench`.
- `uav-bench` is rebased onto `hku-mars/MARSIM@ubuntu20` when upstream
  publishes compatible fixes.

## Using it from eval_workspace

Top-level support/navigation lives in `eval_workspace/README.md`.
Detailed manual MARSIM workflow examples live in `eval_workspace/scripts/README.md` under "MARSIM fork v0.2 extensions".
