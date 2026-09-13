#!/bin/bash
BAG=$(cat /home/jetbot/current_survey_bag 2>/dev/null || true)
echo "=== time ==="
date
echo "=== procs ==="
pgrep -a roslaunch || echo no_roslaunch
pgrep -af survey_tour || echo no_tour
pgrep -a move_base || echo no_move_base
echo "=== current bag ==="
echo "${BAG:-no_current_survey_bag}"
[ -z "$BAG" ] || ls -lh "$BAG"* 2>/dev/null
echo "=== tour ==="
grep survey_tour /home/jetbot/.ros/log/latest/rosout.log 2>/dev/null | grep -E "goal |reached|skipping|blocked|tour finished|complete:|FATAL|openloop|min_front|min_scan|amcl-drift|safety" | tail -n 30
echo "=== tour log ==="
tail -n 20 /tmp/survey_tour.log 2>/dev/null || echo no_survey_tour_log
echo "=== fatal ==="
grep -E "survey recording failed|has died|GStreamer|Safety stop" /tmp/amcl_survey_20260827.log | tail -n 20
echo "=== failed ==="
[ -z "$BAG" ] || cat "$BAG.failed.json" 2>/dev/null
echo "=== complete ==="
[ -z "$BAG" ] || cat "$BAG.complete.json" 2>/dev/null
