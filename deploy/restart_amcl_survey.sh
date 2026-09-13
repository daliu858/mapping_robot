#!/bin/bash
# Stop this robot's ROS stack, then start one new AMCL second-pass survey.
set -eu
: "${SURVEY_BAG:?SURVEY_BAG must be selected by start_nav_survey.sh}"

# Match executable command lines, not generic fragments such as '/amcl'.
# A broad fragment can also match the remote shell that launched this script
# and sever the operator's SSH session before diagnostics are printed.
kill_node() {
  pkill -9 -f "$1([[:space:]]|$)" 2>/dev/null || true
}

if pgrep -x roslaunch >/dev/null; then
  killall -INT roslaunch 2>/dev/null || true
  sleep 8
fi
killall -9 roslaunch rosmaster rosout rosbag 2>/dev/null || true
kill_node '/home/jetbot/catkin_ws/devel/lib/jetbot_pro/checked_rosbag_record.py'
kill_node '/home/jetbot/catkin_ws/devel/lib/jetbot_pro/scan_watchdog.py'
kill_node '/home/jetbot/catkin_ws/devel/lib/jetbot_pro/joy_teleop.py'
kill_node '/home/jetbot/catkin_ws/src/jetbot_pro/scripts/joy_teleop.py'
kill_node '/home/jetbot/catkin_ws/devel/lib/jetbot_pro/odom_ekf.py'
kill_node '/home/jetbot/survey_tour_20260827.py'
kill_node '/opt/ros/melodic/lib/amcl/amcl'
kill_node '/opt/ros/melodic/lib/map_server/map_server'
kill_node '/opt/ros/melodic/lib/move_base/move_base'
kill_node '/home/jetbot/catkin_ws/devel/lib/gscam/gscam'
kill_node '/home/jetbot/catkin_ws/devel/lib/rplidar_ros/rplidarNode'
sleep 2
if pgrep -a roslaunch >/dev/null || pgrep -a rosmaster >/dev/null; then
  echo leftover_ros
  pgrep -a roslaunch || true
  pgrep -a rosmaster || true
  exit 1
fi
echo no_roslaunch
# Never remove a prior bag here. The selected output was checked unused before
# startup, and checked_rosbag_record.py independently refuses every conflict.
df -h / | tail -1
if sudo -n true 2>/dev/null; then
  if sudo -n systemctl restart nvargus-daemon; then
    echo nvargus_restarted
  else
    echo nvargus_restart_failed
  fi
  sleep 4
else
  echo nvargus_restart_skipped_no_passwordless_sudo
fi
ls -l /dev/ttyACM0 /dev/ttyACM1 /dev/input/js0 || true
: > /tmp/amcl_survey_20260827.log
SURVEY_BAG="$SURVEY_BAG" setsid nohup \
  /home/jetbot/run_amcl_survey_20260827.sh \
  > /tmp/amcl_survey_20260827.log 2>&1 < /dev/null &
echo PID=$!
