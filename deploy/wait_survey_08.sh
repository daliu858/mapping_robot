#!/bin/bash
BAG=/home/jetbot/semantic_survey_home_20260827_08.bag
for i in $(seq 1 90); do
  echo "--- tick $i ---"
  date
  if [ -f "${BAG}.active" ]; then
    echo ACTIVE_YES
  fi
  if [ -f "${BAG}.complete.json" ]; then
    echo COMPLETE_YES
    cat "${BAG}.complete.json"
    exit 0
  fi
  if [ -f "${BAG}.failed.json" ]; then
    echo FAILED_YES
    cat "${BAG}.failed.json"
    echo "--- recorder log ---"
    ls -t /home/jetbot/.ros/log/*/offline_recorder-*.log 2>/dev/null | head -1 | xargs -I{} tail -n 40 {}
    exit 1
  fi
  if grep -q "survey healthy; recording to ${BAG}" /tmp/amcl_survey_20260827.log 2>/dev/null; then
    echo HEALTHY_YES
    grep -E "survey healthy|survey waiting|survey recording failed" /tmp/amcl_survey_20260827.log | tail -n 8
    ls -l "${BAG}.active" "${BAG}" 2>/dev/null || true
    pgrep -a roslaunch | head
    exit 0
  fi
  if grep -q "survey recording failed" /tmp/amcl_survey_20260827.log 2>/dev/null; then
    echo RECORDER_FAILED
    grep -E "survey waiting|survey recording failed|survey healthy" /tmp/amcl_survey_20260827.log | tail -n 15
    exit 1
  fi
  grep -E "survey waiting|survey healthy" /tmp/amcl_survey_20260827.log | tail -n 2 || true
  pgrep -a roslaunch >/dev/null || echo no_roslaunch
  sleep 2
done
echo TIMEOUT
tail -n 40 /tmp/amcl_survey_20260827.log
ls -t /home/jetbot/.ros/log/*/offline_recorder-*.log 2>/dev/null | head -1 | xargs -I{} tail -n 40 {}
exit 1
