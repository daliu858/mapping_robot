#!/bin/bash
echo "=== df ==="
df -h / | tail -1
echo "=== bags ==="
ls -lh /home/jetbot/*.bag* 2>/dev/null
echo "=== local costmap yaml ==="
grep -n "plugins\|static_layer\|obstacle_layer" /home/jetbot/catkin_ws/src/jetbot_pro/config/diff/local_costmap_params.yaml
echo "=== devel yaml? ==="
ls /home/jetbot/catkin_ws/src/jetbot_pro/config/diff/local_costmap_params.yaml
echo "=== 04 keep ==="
ls -lh /home/jetbot/semantic_survey_home_20260804* /home/jetbot/maps/*.yaml 2>/dev/null | head
