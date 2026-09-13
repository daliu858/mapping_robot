#!/bin/bash
# Seal the checked AMCL survey bag after a joystick second pass.
set -euo pipefail
export ROS_DISTRO=melodic
set +u
# shellcheck disable=SC1091
source /opt/ros/melodic/setup.bash
# shellcheck disable=SC1091
source /home/jetbot/catkin_ws/devel/setup.bash
set -u
export ROS_MASTER_URI=http://127.0.0.1:11311
ROS_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
: "${ROS_ADDR:=192.168.3.31}"
export ROS_IP="$ROS_ADDR"
export ROS_HOSTNAME="$ROS_ADDR"
rosservice call /offline_recorder/complete "{}"
