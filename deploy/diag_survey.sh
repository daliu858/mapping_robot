#!/bin/bash
set -e
echo "=== time ==="
date -u
date
echo "=== processes ==="
pgrep -a roslaunch || echo no_roslaunch
pgrep -a rosmaster || echo no_rosmaster
pgrep -a python || true
echo "=== survey log tail ==="
tail -n 80 /tmp/amcl_survey_20260827.log || echo no_survey_log
echo "=== latest ros log dir ==="
ls -lt /home/jetbot/.ros/log | head
echo "=== recorder logs ==="
ls -lt /home/jetbot/.ros/log/*/offline_recorder-*.log 2>/dev/null | head
echo "=== bag files ==="
ls -lt /home/jetbot/semantic_survey_home_20260827_*.bag* 2>/dev/null | head -20
echo "=== failed json ==="
cat /home/jetbot/semantic_survey_home_20260827_07.bag.failed.json 2>/dev/null || echo no_07_failed
echo "=== run script ==="
cat /home/jetbot/run_amcl_survey_20260827.sh
echo "=== launch bag name ==="
grep -n bag /home/jetbot/amcl_survey_20260827.launch | head
