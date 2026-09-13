#!/bin/bash
echo "=== tour log ==="
cat /tmp/survey_tour.log
echo "=== procs ==="
pgrep -af tour || echo no_tour
pgrep -a roslaunch || echo no_roslaunch
pgrep -a move_base || echo no_move_base
echo "=== bag ==="
ls -lh /home/jetbot/semantic_survey_home_20260827_11.bag* 2>/dev/null
echo "=== python err ==="
pgrep -af survey_tour || true
