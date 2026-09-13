#!/bin/bash
echo "=== 10 files ==="
ls -lh /home/jetbot/semantic_survey_home_20260827_10.bag* 2>/dev/null
echo "=== complete json ==="
cat /home/jetbot/semantic_survey_home_20260827_10.bag.complete.json 2>/dev/null
echo "=== procs ==="
pgrep -a roslaunch || echo no_roslaunch
pgrep -a move_base || echo no_move_base
echo "=== tour log ==="
cat /tmp/survey_tour.log
echo "=== move_base errors ==="
ls -t /home/jetbot/.ros/log/latest/move_base*.log 2>/dev/null | head -1
LATEST=$(ls -t /home/jetbot/.ros/log/latest/move_base*.log 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  grep -E "Abort|error|Error|WARN|costmap|plan|safety|oscillat|Clearing" "$LATEST" | tail -n 60
fi
echo "=== survey log nav ==="
grep -E "move_base|TEB|Abort|safety|ERROR|Got new plan" /tmp/amcl_survey_20260827.log | tail -n 40
