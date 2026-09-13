#!/bin/bash
# Start the checked AMCL second pass for a joystick drive. No openloop tour.
# This is the single deployment entry point: it chooses one unused bag and
# passes that same name to launch and status tooling.
set -euo pipefail

# Exit contract (mirrored in releases/README.md and the pipeline contract
# tests). This entry script only reports STARTUP; the operator drives and
# seals the bag with complete_survey.sh.
#   0  deployed + recording .active + joystick teleop up (JOYSTICK_READY
#      is the only success word and the only zero-exit path)
#   1  recorder failed to start (FAILED_START / NO_ACTIVE)
#   2  no unused bag slot in 19..99
#   3  another instance holds the entry lock
#
# Single-instance lock: two concurrent starters would select the same bag
# number and one would then die on the recorder's conflict check. Serialize
# the whole entry instead; fd 9 is closed in every spawned child so the lock
# dies with this script, not with the detached survey.
ENTRY_LOCK=/home/jetbot/current_survey_bag.lock
exec 9>"$ENTRY_LOCK"
if ! flock -n 9; then
  echo "another start_nav_survey.sh holds $ENTRY_LOCK; not racing bag selection" >&2
  exit 3
fi

# Fail closed before selecting a bag or changing the running system.  df -P
# reports 1K blocks, so 1536 MiB is 1572864 blocks; do not use localized
# human-readable units here.
if ! DISK_AVAILABLE_KB=$(df -P -k / | awk 'NR == 2 { print $4 }'); then
  echo "DISK_LOW: unable to query / free space; refusing survey start" >&2
  exit 1
fi
case "$DISK_AVAILABLE_KB" in
  ''|*[!0-9]*)
    echo "DISK_LOW: unable to parse / free space; refusing survey start" >&2
    exit 1
    ;;
esac
if [ "$DISK_AVAILABLE_KB" -lt 1572864 ]; then
  echo "DISK_LOW: / has less than 1536 MiB available; refusing survey start" >&2
  exit 1
fi

# The survey must remain at the robot-tested 15 fps.  Validate the file that
# this entry point is about to deploy, before any cp or roslaunch side effect.
if ! grep -q 'name="fps" value="15"' /tmp/amcl_survey_20260827.launch; then
  echo "FPS_INVALID: amcl_survey_20260827.launch must set fps to 15" >&2
  exit 1
fi

# A stale Argus failure needs an operator action when this script cannot use
# passwordless sudo.  The restart helper performs the actual non-interactive
# restart; this preflight only checks capability and never asks for a password.
if grep -Eq 'Failed to create CaptureSession|Could not get gstreamer sample' \
    /tmp/amcl_survey_20260827.log 2>/dev/null; then
  if ! sudo -n -l systemctl restart nvargus-daemon >/dev/null 2>&1; then
    echo "NVARGUS_STALE: previous camera failure; operator must restart nvargus; this script does not sudo" >&2
    exit 1
  fi
fi

BAG_ROOT=/home/jetbot/semantic_survey_home_20260827
BAG_POINTER=/home/jetbot/current_survey_bag
BAG=""
# 01..18 are historical attempts even when their failed files were removed;
# never recycle those identifiers and confuse later forensic review.
for i in $(seq 19 99); do
  candidate="${BAG_ROOT}_$(printf '%02d' "$i").bag"
  unused=true
  for suffix in "" ".active" ".complete.json" ".failed.json"; do
    if [ -e "${candidate}${suffix}" ]; then
      unused=false
      break
    fi
  done
  if [ "$unused" = true ]; then
    BAG="$candidate"
    break
  fi
done
if [ -z "$BAG" ]; then
  echo "no unused survey bag slot in ${BAG_ROOT}_19..99" >&2
  exit 2
fi
export SURVEY_BAG="$BAG"
echo "selected survey bag: $BAG"
cp /tmp/amcl_survey_20260827.launch /home/jetbot/amcl_survey_20260827.launch
cp /tmp/run_amcl_survey_20260827.sh /home/jetbot/run_amcl_survey_20260827.sh
cp /tmp/complete_survey.sh /home/jetbot/complete_survey.sh
cp /tmp/joystick_teleop.launch /home/jetbot/catkin_ws/src/jetbot_pro/launch/joystick_teleop.launch
cp /tmp/global_costmap_params.yaml /home/jetbot/catkin_ws/src/jetbot_pro/config/diff/global_costmap_params.yaml
cp /tmp/base_global_planner_param.yaml /home/jetbot/catkin_ws/src/jetbot_pro/config/diff/base_global_planner_param.yaml
cp /tmp/local_costmap_params.yaml /home/jetbot/catkin_ws/src/jetbot_pro/config/diff/local_costmap_params.yaml
cp /tmp/costmap_common_params.yaml /home/jetbot/catkin_ws/src/jetbot_pro/config/diff/costmap_common_params.yaml
cp /tmp/teb_local_planner_params.yaml /home/jetbot/catkin_ws/src/jetbot_pro/config/diff/teb_local_planner_params.yaml
cp /tmp/move_base_params.yaml /home/jetbot/catkin_ws/src/jetbot_pro/config/diff/move_base_params.yaml
cp /tmp/lidar.launch /home/jetbot/catkin_ws/src/jetbot_pro/launch/lidar.launch
cp /tmp/nav.launch /home/jetbot/catkin_ws/src/jetbot_pro/launch/nav.launch
cp /tmp/semantic_survey.launch /home/jetbot/catkin_ws/src/jetbot_pro/launch/semantic_survey.launch
cp /tmp/record_offline.launch /home/jetbot/catkin_ws/src/jetbot_pro/launch/record_offline.launch
# Dual-copy every Python sibling the launch graph imports.  A missing
# devel copy is ImportError and kills required nodes (bag 19 died twice:
# map_identity, then replay_contract).
for py in checked_rosbag_record.py scan_clip.py scan_clip_core.py \
          map_identity.py replay_contract.py; do
  cp "/tmp/$py" "/home/jetbot/catkin_ws/src/jetbot_pro/scripts/$py"
  cp "/tmp/$py" "/home/jetbot/catkin_ws/devel/lib/jetbot_pro/$py"
done
# roslaunch resolves the installed/devel executable, so both copies must be
# executable and identical. Do not rely on catkin's prior mode bits.
chmod +x /home/jetbot/catkin_ws/src/jetbot_pro/scripts/checked_rosbag_record.py \
  /home/jetbot/catkin_ws/devel/lib/jetbot_pro/checked_rosbag_record.py \
  /home/jetbot/catkin_ws/src/jetbot_pro/scripts/scan_clip.py \
  /home/jetbot/catkin_ws/devel/lib/jetbot_pro/scan_clip.py
chmod +x /home/jetbot/run_amcl_survey_20260827.sh \
  /home/jetbot/complete_survey.sh /tmp/complete_survey.sh \
  /tmp/restart_amcl_survey.sh
# Publish the pointer only after deployment succeeds; status tools must not
# mistake a partially copied attempt for the current run.
printf '%s\n' "$BAG" > "${BAG_POINTER}.tmp"
mv -f "${BAG_POINTER}.tmp" "$BAG_POINTER"
bash /tmp/restart_amcl_survey.sh 9>&-
i=1
while [ "$i" -le 45 ]; do
  if [ -f "${BAG}.failed.json" ]; then
    echo FAILED_START
    cat "${BAG}.failed.json"
    tail -n 40 /tmp/amcl_survey_20260827.log
    exit 1
  fi
  if [ -f "${BAG}.active" ]; then
    echo RECORDING_YES
    ls -l "${BAG}.active"
    break
  fi
  grep -E "survey waiting|survey healthy|GStreamer|has died" /tmp/amcl_survey_20260827.log | tail -n 2 || true
  i=$((i + 1))
  sleep 2
done
if [ ! -f "${BAG}.active" ]; then
  echo NO_ACTIVE
  tail -n 50 /tmp/amcl_survey_20260827.log
  exit 1
fi
echo JOYSTICK_READY
echo "drive with the stick (no trigger). when finished: bash /tmp/complete_survey.sh"
exit 0
