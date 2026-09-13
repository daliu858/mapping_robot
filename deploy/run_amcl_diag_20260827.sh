#!/bin/bash
source /opt/ros/melodic/setup.bash
source /home/jetbot/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=192.168.3.31
export ROS_HOSTNAME=192.168.3.31
exec roslaunch /home/jetbot/amcl_diag_20260827.launch
