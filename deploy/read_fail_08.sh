#!/bin/bash
echo "=== failed json ==="
cat /home/jetbot/semantic_survey_home_20260827_08.bag.failed.json
echo
echo "=== recorder waiting lines ==="
grep -E "survey waiting|survey healthy|survey recording failed|ValueError|camera" /home/jetbot/.ros/log/latest/offline_recorder-16.log
echo
echo "=== last 80 of survey log (node deaths) ==="
grep -E "ERROR|FATAL|died|process\[|offline_recorder|gscam|rplidar|AMCL|not ready" /tmp/amcl_survey_20260827.log | tail -n 80
