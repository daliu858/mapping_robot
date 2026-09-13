#!/bin/bash
set -e
# Melodic's catkin profile reads ROS_DISTRO before assigning it.  nounset
# (`set -u`) therefore aborts source and leaves no .active bag.
export ROS_DISTRO="${ROS_DISTRO:-melodic}"
set +u
source /opt/ros/melodic/setup.bash
source /home/jetbot/catkin_ws/devel/setup.bash
set -u
export ROS_MASTER_URI=http://127.0.0.1:11311
# OLED IP can move after a reboot.  Prefer the live address, fall back to
# the last known Jetson address so a missing hostname -I does not abort.
ROS_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
: "${ROS_ADDR:=192.168.3.31}"
export ROS_IP="$ROS_ADDR"
export ROS_HOSTNAME="$ROS_ADDR"
: "${SURVEY_BAG:?SURVEY_BAG must name a new .bag output}"
case "$SURVEY_BAG" in
  /home/jetbot/semantic_survey_home_20260827_*.bag) ;;
  *) echo "refusing unexpected SURVEY_BAG: $SURVEY_BAG" >&2; exit 2 ;;
esac
for output in "$SURVEY_BAG" "$SURVEY_BAG.active" \
              "$SURVEY_BAG.complete.json" "$SURVEY_BAG.failed.json"; do
  if [ -e "$output" ]; then
    echo "refusing existing survey output: $output" >&2
    exit 2
  fi
done
exec roslaunch /home/jetbot/amcl_survey_20260827.launch bag:="$SURVEY_BAG"
