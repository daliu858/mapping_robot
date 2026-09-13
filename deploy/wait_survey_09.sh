#!/bin/bash
BAG=$(cat /home/jetbot/current_survey_bag 2>/dev/null || true)
if [ -z "$BAG" ]; then
  echo no_current_survey_bag >&2
  exit 2
fi
i=1
while [ "$i" -le 75 ]; do
  echo "--- tick $i ---"
  date
  if [ -f "${BAG}.failed.json" ]; then
    echo FAILED_YES
    cat "${BAG}.failed.json"
    echo "--- waiting lines ---"
    grep -E "survey waiting|survey healthy|survey recording failed" /tmp/amcl_survey_20260827.log | tail -n 20
    exit 1
  fi
  if grep -q "survey healthy; recording to ${BAG}" /tmp/amcl_survey_20260827.log 2>/dev/null; then
    echo HEALTHY_YES
    ls -l "${BAG}.active" "${BAG}" 2>/dev/null || true
    pgrep -a roslaunch | head
    exit 0
  fi
  grep -E "survey waiting|GStreamer|has died|survey recording failed" /tmp/amcl_survey_20260827.log | tail -n 3 || true
  pgrep -a roslaunch >/dev/null || echo no_roslaunch
  i=$((i + 1))
  sleep 2
done
echo TIMEOUT
tail -n 50 /tmp/amcl_survey_20260827.log
exit 1
