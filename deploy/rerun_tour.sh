#!/bin/bash
BAG=$(cat /home/jetbot/current_survey_bag 2>/dev/null || true)
if [ -z "$BAG" ] || [ ! -f "$BAG.active" ]; then
  echo "no active current survey; refusing to start tour" >&2
  exit 2
fi
cp /tmp/survey_tour_20260827.py /home/jetbot/survey_tour_20260827.py
cp /tmp/survey_tour_logic.py /home/jetbot/survey_tour_logic.py
sed -i 's/\r$//' /home/jetbot/survey_tour_20260827.py \
  /home/jetbot/survey_tour_logic.py /home/jetbot/run_survey_tour.sh
pkill -f '/home/jetbot/survey_tour_20260827.py([[:space:]]|$)' || true
sleep 1
: > /tmp/survey_tour.log
setsid nohup /home/jetbot/run_survey_tour.sh > /tmp/survey_tour.log 2>&1 < /dev/null &
echo TOUR_PID=$!
sleep 8
tail -n 20 /tmp/survey_tour.log
ls -lh "$BAG.active" 2>/dev/null
pgrep -a roslaunch || echo no_roslaunch
