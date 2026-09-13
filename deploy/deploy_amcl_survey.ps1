# Stage the AMCL second-pass files to jetbot:/tmp, then start_nav_survey.sh
# copies them into place.  Run from the PC after the car is on WiFi.
# Usage: .\releases\deploy_amcl_survey.ps1 [-Host jetbot@192.168.3.31]
param(
  [string]$Target = "jetbot@192.168.3.31"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not (Test-Path (Join-Path $Root "releases\start_nav_survey.sh"))) {
  $Root = (Get-Location).Path
}

$pairs = @(
  @("releases\start_nav_survey.sh", "/tmp/start_nav_survey.sh"),
  @("releases\restart_amcl_survey.sh", "/tmp/restart_amcl_survey.sh"),
  @("releases\run_amcl_survey_20260827.sh", "/tmp/run_amcl_survey_20260827.sh"),
  @("releases\complete_survey.sh", "/tmp/complete_survey.sh"),
  @("releases\amcl_survey_20260827.launch", "/tmp/amcl_survey_20260827.launch"),
  @("releases\status_survey.sh", "/tmp/status_survey.sh"),
  @("releases\status_healthy.sh", "/tmp/status_healthy.sh"),
  @("releases\status_disk.sh", "/tmp/status_disk.sh"),
  @("releases\live_nav.sh", "/tmp/live_nav.sh"),
  @("releases\read_tour.sh", "/tmp/read_tour.sh"),
  @("src\jetbot_pro\config\diff\global_costmap_params.yaml", "/tmp/global_costmap_params.yaml"),
  @("src\jetbot_pro\config\diff\base_global_planner_param.yaml", "/tmp/base_global_planner_param.yaml"),
  @("src\jetbot_pro\config\diff\local_costmap_params.yaml", "/tmp/local_costmap_params.yaml"),
  @("src\jetbot_pro\config\diff\costmap_common_params.yaml", "/tmp/costmap_common_params.yaml"),
  @("src\jetbot_pro\config\diff\teb_local_planner_params.yaml", "/tmp/teb_local_planner_params.yaml"),
  @("src\jetbot_pro\config\diff\move_base_params.yaml", "/tmp/move_base_params.yaml"),
  @("src\jetbot_pro\launch\lidar.launch", "/tmp/lidar.launch"),
  @("src\jetbot_pro\launch\nav.launch", "/tmp/nav.launch"),
  @("src\jetbot_pro\launch\semantic_survey.launch", "/tmp/semantic_survey.launch"),
  @("src\jetbot_pro\launch\joystick_teleop.launch", "/tmp/joystick_teleop.launch"),
  @("src\jetbot_pro\launch\record_offline.launch", "/tmp/record_offline.launch"),
  @("src\jetbot_pro\scripts\checked_rosbag_record.py", "/tmp/checked_rosbag_record.py"),
  @("src\jetbot_pro\scripts\scan_clip.py", "/tmp/scan_clip.py"),
  @("src\jetbot_pro\scripts\scan_clip_core.py", "/tmp/scan_clip_core.py"),
  @("src\jetbot_pro\scripts\map_identity.py", "/tmp/map_identity.py"),
  @("src\jetbot_pro\scripts\replay_contract.py", "/tmp/replay_contract.py")
)

foreach ($pair in $pairs) {
  $src = Join-Path $Root $pair[0]
  if (-not (Test-Path $src)) { throw "missing $src" }
}

Write-Host "scp $($pairs.Count) files -> ${Target}:/tmp"
$locals = $pairs | ForEach-Object { Join-Path $Root $_[0] }
& scp -o ConnectTimeout=12 @locals "${Target}:/tmp/"
if ($LASTEXITCODE -ne 0) { throw "scp failed: $LASTEXITCODE" }

# Windows checkouts often ship CRLF; bash will then fail on shebang.
$strip = @'
set -e
cd /tmp
# GNU sed treats `\r` as the letter r unless we pass a real CR.  Using
# `s/r$//` on Python would turn trailing `or` into `o` and SyntaxError.
for f in start_nav_survey.sh restart_amcl_survey.sh run_amcl_survey_20260827.sh complete_survey.sh status_survey.sh status_healthy.sh status_disk.sh live_nav.sh read_tour.sh; do
  sed -i $'s/\r$//' "$f"
  chmod +x "$f"
done
sed -i $'s/\r$//' /tmp/*.py /tmp/*.launch /tmp/*.yaml
grep -q 'name="fps" value="15"' /tmp/amcl_survey_20260827.launch
test -f /home/jetbot/maps/home_floor_20260804_glass_fixed.yaml
test -f /home/jetbot/semantic_survey_home_20260804_06.bag
df -P -k / | awk "NR==2 {print \"disk_kb_avail\", \$4}"
ls -l /dev/ttyACM0 /dev/ttyACM1 /dev/input/js0 || echo WAIT_DEVICES
echo DEPLOY_OK
'@
& ssh -o ConnectTimeout=12 $Target $strip
if ($LASTEXITCODE -ne 0) { throw "remote strip/check failed: $LASTEXITCODE" }
Write-Host "staged. next on the robot: bash /tmp/start_nav_survey.sh"
