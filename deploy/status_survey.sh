#!/bin/bash
BAG=$(cat /home/jetbot/current_survey_bag 2>/dev/null || true)
echo "=== procs ==="
pgrep -a roslaunch || echo no_roslaunch
pgrep -a rosmaster || echo no_rosmaster
pgrep -a gscam || echo no_gscam
pgrep -a nvargus || echo no_nvargus
echo "=== current bag ==="
if [ -n "$BAG" ]; then
  echo "$BAG"
  ls -l "$BAG"* 2>/dev/null || echo no_current_output
else
  echo no_current_survey_bag
fi
echo "=== sudo ==="
sudo -n true && echo sudo_nopass || echo sudo_needs_pass
echo "=== log grep ==="
grep -E "survey waiting|survey healthy|GStreamer|has died|nvargus" /tmp/amcl_survey_20260827.log | tail -n 30
