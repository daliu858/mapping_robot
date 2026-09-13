#!/bin/bash
. /opt/ros/melodic/setup.sh
. /home/jetbot/catkin_ws/devel/setup.sh
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=192.168.3.31
echo "=== procs ==="
pgrep -a roslaunch || echo no_roslaunch
pgrep -a move_base || echo no_move_base
echo "=== params ==="
timeout 8 rosparam get /move_base/global_costmap/inflation_layer/inflation_radius
timeout 8 rosparam get /move_base/GlobalPlanner/default_tolerance
timeout 8 rosparam get /move_base/GlobalPlanner/allow_unknown
echo "=== pose ==="
timeout 5 rostopic echo -n 1 /amcl_pose
echo "=== make_plan ==="
timeout 15 rosservice call /move_base/make_plan "start:
  header:
    frame_id: 'map'
  pose:
    position: {x: -0.61, y: 3.61, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.67, w: 0.74}
goal:
  header:
    frame_id: 'map'
  pose:
    position: {x: -1.5, y: 3.61, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.67, w: 0.74}
tolerance: 0.5" 2>&1 | head -n 40
echo "=== move_base tail ==="
grep -E "start|obstacle|off the global|Failed to get|Aborting|plan" /tmp/amcl_survey_20260827.log | tail -n 40
echo "=== latest move_base log ==="
MB=$(ls -t /home/jetbot/.ros/log/latest/move_base*.log 2>/dev/null | head -1)
echo "$MB"
if [ -n "$MB" ]; then tail -n 80 "$MB"; fi
