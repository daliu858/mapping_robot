#!/bin/bash
# Restart only /joy_teleop with survey stick-forward = +vx.
# Does not stop AMCL, cameras, or the checked recorder.
set -euo pipefail
export ROS_DISTRO=melodic
set +u
# shellcheck disable=SC1091
source /opt/ros/melodic/setup.bash
# shellcheck disable=SC1091
source /home/jetbot/catkin_ws/devel/setup.bash
set -u
export ROS_MASTER_URI=http://127.0.0.1:11311
ROS_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
: "${ROS_ADDR:=192.168.3.31}"
export ROS_IP="$ROS_ADDR"
export ROS_HOSTNAME="$ROS_ADDR"

for src in /tmp/joystick_teleop.launch /tmp/semantic_survey.launch; do
  if [ ! -f "$src" ]; then
    echo "missing $src" >&2
    exit 1
  fi
done
python -c '
import sys
for path in sys.argv[1:]:
    with open(path, "rb") as stream:
        data = stream.read().replace(b"\r\n", b"\n").replace(b"\r", b"")
    with open(path, "wb") as stream:
        stream.write(data)
' /tmp/joystick_teleop.launch /tmp/semantic_survey.launch
cp /tmp/joystick_teleop.launch /home/jetbot/catkin_ws/src/jetbot_pro/launch/joystick_teleop.launch
cp /tmp/semantic_survey.launch /home/jetbot/catkin_ws/src/jetbot_pro/launch/semantic_survey.launch

rosparam set /joy_teleop/scale_linear 0.15
rosparam set /joy_teleop/scale_linear_turbo 0.30
rosparam set /joy_teleop/require_enable_button false

rosnode kill /joy_teleop || true
for _ in 1 2 3 4 5 6 7 8; do
  if ! rosnode list 2>/dev/null | grep -qx '/joy_teleop'; then
    break
  fi
  sleep 0.5
done

nohup rosrun jetbot_pro joy_teleop.py __name:=joy_teleop \
  _require_enable_button:=false \
  _scale_linear:=0.15 \
  _scale_linear_turbo:=0.30 \
  >/tmp/joy_teleop_flip.log 2>&1 &
disown || true

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if rosnode list 2>/dev/null | grep -qx '/joy_teleop'; then
    echo "JOY_FLIPPED scale_linear=$(rosparam get /joy_teleop/scale_linear)"
    rosnode list | grep joy || true
    ls -lh /home/jetbot/semantic_survey_home_20260827_*.bag.active 2>/dev/null || true
    exit 0
  fi
  sleep 0.5
done
echo "joy_teleop failed to come back; see /tmp/joy_teleop_flip.log" >&2
exit 1
