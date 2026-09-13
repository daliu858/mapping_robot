#!/bin/bash
echo "=== 12 failed ==="
cat /home/jetbot/semantic_survey_home_20260827_12.bag.failed.json 2>/dev/null
echo
echo "=== survey errors ==="
grep -E "safety_stop|Safety stop|valid control|valid plan|AMCL|stale|GStreamer|scan" /tmp/amcl_survey_20260827.log | tail -n 50
echo "=== recorder ==="
REC=$(ls -t /home/jetbot/.ros/log/*/offline_recorder-*.log 2>/dev/null | head -1)
echo "$REC"
grep -E "stale|AMCL|waiting|healthy|failed" "$REC" 2>/dev/null | tail -n 20
