#!/bin/bash
# Snapshot the live AMCL survey, then seal the bag. Do not start a new run.
set -euo pipefail
export ROS_DISTRO=melodic
set +u
# shellcheck disable=SC1091
source /opt/ros/melodic/setup.bash
# shellcheck disable=SC1091
source /home/jetbot/catkin_ws/devel/setup.bash
set -u
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=192.168.3.31
export ROS_HOSTNAME=192.168.3.31

echo "=== BAG POINTER ==="
cat /home/jetbot/current_survey_bag
echo
echo "=== BAG FILES ==="
ls -lh /home/jetbot/semantic_survey_home_20260827_19.bag* 2>/dev/null || true
echo
echo "=== DISK ==="
df -h / | tail -1
echo
echo "=== NODES ==="
rosnode list | grep -E 'joy|offline|amcl|jetbot|gscam|stereo|rplidar|map_server|scan' || true
echo
echo "=== AMCL ==="
timeout 4 rostopic echo -n 1 /amcl_pose || echo "AMCL_ECHO_TIMEOUT"
echo
echo "=== COMPLETE ==="
rosservice call /offline_recorder/complete "{}"
echo
echo "=== AFTER ==="
ls -lh /home/jetbot/semantic_survey_home_20260827_19.bag* 2>/dev/null || true
echo
echo "=== JOY STOP ==="
rosnode kill /joy_teleop || true
echo sealed
