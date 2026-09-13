# 2026-08-27 AMCL 第二趟现场文件

> 存档说明:本目录是当时真实使用的现场脚本原样存档,其中的绝对路径
> (`/home/jetbot/...`、`releases...`、内网 IP `192.168.3.31`)反映的是
> 原始工作目录布局,在本仓库布局下不保证可直接运行。

这些散文件是 JetBot ROS1 Melodic 的 **AMCL 定位 + 双目录包**修复入口，不是新的
语义巡航。车关机期间只做离线审查；没有人明确说“上车”前，不运行这些脚本。

## 单一入口与 bag 名

现场部署/启动入口为 `start_nav_survey.sh`。它会：

1. 从 `_19` 起选择一个四个路径（`.bag`、`.active`、`.complete.json`、
   `.failed.json`）都不存在的序号；`01..18` 永不复用；
2. 将本次路径原子写到 `/home/jetbot/current_survey_bag`；
3. 同步 launch、配置和脚本；
4. 重启本车 ROS 栈，等 recorder 创建 `.active` 后打印 `JOYSTICK_READY`。
   操作员用手柄开；跑完执行 `bash /tmp/complete_survey.sh` 封印 bag。

`run_amcl_survey_20260827.sh` 与 recorder 会再次拒绝路径冲突。重启脚本不删除任何
bag；不要删除 `/home/jetbot/semantic_survey_home_20260804_06.bag` 或 8/4 地图。
`status_survey.sh`、`status_healthy.sh`、`status_disk.sh`、`live_nav.sh` 都从
`current_survey_bag` 查看同一个输出，不再硬编码旧序号。

## 入口退出码契约

`start_nav_survey.sh` 只报告**启动**结果。路线由人开车完成:

- `0`:部署完成、`.active` 已出现(唯一成功词 `JOYSTICK_READY`,
  也是脚本里唯一的零退出路径);
- `1`:recorder 启动失败(`FAILED_START` / `NO_ACTIVE`);
- `2`:`_19.._99` 没有可用 bag 序号;
- `3`:另一个 `start_nav_survey.sh` 正持有
  `/home/jetbot/current_survey_bag.lock`。

入口不再自动拉起开环 tour。bag 选择用 `flock` 串行化在 fd 9 上;
`restart_amcl_survey.sh` 以 `9>&-` 关闭该 fd,锁随入口脚本消亡。

## 入口预检：磁盘、fps、nvargus

`start_nav_survey.sh` 在选号、复制部署文件和调用 `restart_amcl_survey.sh` 之前做
fail-closed 预检:

- 用 `df -P -k /` 读取 POSIX 1K blocks；可用空间小于 `1572864` blocks
  (1536 MiB) 或无法解析时打印 `DISK_LOW` 并以 1 退出。脚本不会删除任何旧 bag，
  特别不会删除 8/4 成功成果；
- 即将部署的 `/tmp/amcl_survey_20260827.launch` 必须包含 `fps=15`，否则打印
  `FPS_INVALID` 并拒绝启动。survey 入口和 `semantic_survey.launch` 默认都保持
  15fps；`stereo_camera.launch` 的标定默认 30fps 不改；
- 若上一趟 `/tmp/amcl_survey_20260827.log` 含 `Failed to create CaptureSession` 或
  `Could not get gstreamer sample`，且当前用户没有免密执行
  `systemctl restart nvargus-daemon` 的 sudo 权限，则打印 `NVARGUS_STALE` 并拒绝
  启动。脚本不索要密码、不执行交互式 sudo；操作员须先处理 nvargus。

预检失败统一使用退出码 1；因此入口的退出码 0 仍只表示已确认 `JOYSTICK_READY`。

## control_mode:survey 默认 joystick,navigation 是显式 opt-in

`amcl_survey_20260827.launch` 把 `control_mode` 固定为 `joystick`
(`test_costmap_teb_contracts.py` 锁住)。开环路点/车头锥刹停不再作为现场默认。
手柄 enable 扳机在本车上损坏，survey 的 `joystick_teleop.launch` 以
`require_enable_button:=false` 覆盖，摇杆轴可直接发 `cmd_vel`。
`joystick.yaml` 的负 `scale_linear` 不改（开环 tour 单测锁死该符号）。
现场手柄前进与车头同向：`semantic_survey.launch` 把 `scale_linear` 覆盖为
`+0.15`（turbo `+0.30`）。

`control_mode:=navigation` 仍是显式 opt-in。TEB/costmap 参数自洽不等于已在客厅
验证通过；不要默默切到 navigation。

## overlay 双拷（必须）

Melodic 的 `roslaunch` 实际执行 catkin overlay 中的：

`/home/jetbot/catkin_ws/devel/lib/jetbot_pro/checked_rosbag_record.py`

因此 recorder 和 `scan_clip.py`(连同它 import 的 `scan_clip_core.py`,
两者必须同目录)必须同时复制到：

- `/home/jetbot/catkin_ws/src/jetbot_pro/scripts/`
- `/home/jetbot/catkin_ws/devel/lib/jetbot_pro/`

`start_nav_survey.sh` 已执行双拷并对四个路径 `chmod +x`。只更新 `src/scripts`
不会修复正在被 roslaunch 使用的代码。

## AMCL drift 运行时保护

开环 tour 只在实际 `drive` 且 `vx > 0` 的心跳上更新目标距离。每个目标内，连续
3 次心跳距离至少增加 0.04m 会记录 `amcl-drift`，立即发零速并进入 tour 的异常
收尾；异常收尾调用 recorder 的 `/offline_recorder/fail`，不会把漂移路线标成
complete。转向、前方等待、安全停或无效距离会断开这段连续序列。该保护只能识别
运行时的 AMCL/符号跑偏，不能离线证明底盘动力学；上车前仍须由操作员观察
`/cmd_vel` 与 AMCL 目标距离。

## 录包门闩

`checked_rosbag_record.py` 仍然 fail-closed：传感器 stale、相机 TF 错误、rosbag
异常退出或未显式调用 `~complete` 都只产生失败结果。运动造成短暂 AMCL 协方差抖动
时，最近一次低协方差样本在 `amcl_motion_timeout_s` 的有界窗口内保活；窗口过期仍
失败。刚停到路点时也使用同一个有界窗口，避免 odom 先变零导致瞬杀。

成功必须同时得到最终 `.bag` 和 `.bag.complete.json`；只有 `.active` 或
`.failed.json` 都不算完成。

## tour 崩溃与中断的收尾语义

路线知识只在 tour 里;recorder 只拥有录制事务。因此 **`.complete.json` 只表示
录制事务完整,不表示路线成功**:一个 0 路点的 tour 仍会调用 `~complete` 保全
证据 bag,然后以自己的退出码 `2` 报告路线失败(`0` = 至少 1 个路点,
`1` = 崩溃或启动失败)。

recorder 提供 `~fail` 服务(解析为 `/offline_recorder/fail`,
`std_srvs/Trigger`),让崩溃的 tour 无需路线知识即可显式 fail-close 事务;
它与 `~complete` 一样幂等,已结束的事务保持原判,也不会与进行中的完成请求
竞争(`test_offline_transactions.py` 有对应契约测试)。

`survey_tour_20260827.py` 的收尾分两类:

- `rospy.ROSInterruptException`(操作员 Ctrl-C / roslaunch 收尾):**不**调用
  `~fail`,事务交给 recorder 自身的 shutdown 处理,`rerun_tour.sh` 可以对仍然
  健康的录制重跑 tour;
- 其他任何异常:先连发 5 次零速 Twist 停轮(`_crash_stop`),再调
  `/offline_recorder/fail` 关闭事务,`main()` 返回 1 而不是向上抛,避免
  `__main__` 兜底路径二次 abort。构造 tour 对象之前的失败(前向符号断言、
  `init_node`)才走 `__main__` 的 abort 路径。

雷达链路保持 fail-closed:`lidar.launch` 中 `scan_watchdog` 标记
`required="true"`,雷达静默时 watchdog 退出会拖垮整个 roslaunch,recorder 的
`_on_shutdown` 随即把未完成事务落成 `.failed.json`,不会留下无雷达裸跑的车。

## 进程与 sudo 安全

`restart_amcl_survey.sh` 使用完整可执行路径匹配 AMCL/map_server/move_base，不使用
会匹配 launch 路径或 SSH 命令行的宽泛 `pkill -f '/amcl'`。`nvargus-daemon` 仅在
`sudo -n` 可用时重启；否则明确打印跳过，不等待密码。

目录中的旧 `rc*`、`five_shot`、`slam_car_closure_*` 与 one-pass 压缩包只作历史
追溯，不能替代本轮已审查的散文件。
