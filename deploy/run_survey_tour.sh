#!/bin/bash
source /opt/ros/melodic/setup.bash
source /home/jetbot/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
ROS_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
: "${ROS_ADDR:=192.168.3.31}"
export ROS_IP="$ROS_ADDR"
export ROS_HOSTNAME="$ROS_ADDR"
export PYTHONUNBUFFERED=1
exec python /home/jetbot/survey_tour_20260827.py
