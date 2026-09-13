#!/bin/bash
BAG=$(cat /home/jetbot/current_survey_bag 2>/dev/null || true)
echo "=== current bag ==="
if [ -n "$BAG" ]; then
  echo "$BAG"
  ls -l "$BAG"* 2>/dev/null || echo no_current_output
else
  echo no_current_survey_bag
fi
echo "=== procs ==="
pgrep -a roslaunch || echo no_roslaunch
echo "=== log ==="
grep -E "survey waiting|survey healthy|survey recording failed|GStreamer|has died" /tmp/amcl_survey_20260827.log | tail -n 15
. /opt/ros/melodic/setup.sh
. /home/jetbot/catkin_ws/devel/setup.sh
echo "=== status ==="
timeout 3 rostopic echo -n 1 /survey/recording_status 2>/dev/null || echo no_status
