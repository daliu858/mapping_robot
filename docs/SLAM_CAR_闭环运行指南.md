# SLAM Car 一趟式语义地图 Demo 指南

本项目输出一张供学生查看的室内语义地图。正式 demo 不包含语义导航，也不要求机器人在建图后再次自动巡航。

## 最终流程

```text
一次实体巡视
  └─ gmapping + 双目视频 + LiDAR + odom + TF
        └─ 最终 occupancy map + 原子提交 rosbag
              └─ PC：LLMDet-Large 检测
                    └─ 离线双目深度与 SLAM TF 投影
                          └─ 带 chair/table/door 标签的地图图片和 YAML
```

AMCL、`move_base` 和 TEB 不在正式 demo 的启动树中。gmapping 本身会在建图期间给出 `map -> odom`，足够把同步视频中的物体投到本次最终地图。

## 开跑前，人要做什么

1. 把机器人放在坚硬、平整的地面，不能放在毛毯或软垫上。
2. 固定双目相机板，确认镜头不再向下垂。
3. 打开 LiDAR，清走电线、小件衣物等可能卷入轮子的物体。
4. 确认机器人周围至少约 0.5 m 空间。
5. Jetson 至少留出 `2.5 GiB` 空间；bag 与地图必须使用新的 `RUN_ID`。

## 启动

```bash
roslaunch jetbot_pro slam_capture.launch \
  bag:=/home/jetbot/captures/home_RUN_ID.bag \
  map_file:=/home/jetbot/maps/home_RUN_ID.yaml \
  camera_mount:="0.105 0.030 0.094 -1.5708 0 -1.5708"
```

录制器确认以下内容后才开始 staging bag：

- `/scan`、`/odom`、`/odom_raw` 和 `/map` 持续更新；
- 左右压缩图和两个 CameraInfo 持续更新；
- TF 链包含 `map -> odom -> base_footprint -> stereo_left_optical`；
- `base_footprint -> laser_frame` 存在；
- 相机 TF 与显式填写的 `camera_mount` 相同；
- 磁盘空间达到安全线。

## 人如何带车巡视

为了让这次 demo 一次成功，正式路线采用受监督遥控，不使用尚未完成全屋验收的自动规划器。

- 低速连续经过主要房间和走廊，不必走得非常精确。
- 在 `chair`、`table`、`door` 清楚出现在相机前方时停约一秒。
- 不要在原地连续旋转；需要掉头时选择空旷位置。
- 离玻璃门和桌腿稍远，始终有人看护。

有手柄时启动命令追加 `start_joystick:=true`。没有手柄时只允许一个外部 `/cmd_vel` 控制源，禁止同时启动键盘、自动探索和其他运动脚本。

## 正确结束

1. 先完全停车。
2. 保持静止至少三秒，让 gmapping 发布最后一次地图更新。
3. 调用：

```bash
rosservice call /slam_capture/complete "{}"
```

成功响应必须是：

```text
success: True
message: "SLAM map and bag committed"
```

程序会自动保存最终 PGM/YAML、写入地图身份和唯一完成事务、关闭并验证 staging bag，再原子改名为正式 bag。中途断电或任一传感器失效时只留下 staging/failed 文件，不会伪装成成功结果。

## PC 离线处理

先复制最终 bag、`.complete.json`、地图 YAML 和 PGM，核对两端 SHA-256，然后运行：

```powershell
D:\gd_env\Scripts\python.exe tools\offline_preflight.py `
  --bag data\home_RUN_ID.bag --map-file maps\home_RUN_ID.yaml `
  --hf-home D:\hf_cache --local-only

D:\gd_env\Scripts\python.exe tools\groundingdino_detect.py `
  --bag data\home_RUN_ID.bag `
  --out data\detections_RUN_ID.json `
  --model iSEE-Laboratory/llmdet_large `
  --stride 10 --skip-doorplates `
  --hf-home D:\hf_cache --local-only
```

随后在 Jetson 离线回放双目融合，得到 `semantic_objects_RUN_ID.yaml`；电脑端完成房间分割、三层合并，并渲染一张 occupancy grid + POI 标签的最终 PNG。

## Demo 验收标准

- 一次连续巡视产生正式 bag、`.complete.json` 和最终地图；
- bag 没有 FAILED 标记，地图哈希、相机外参和 TF 全部通过；
- PC 可直接读取 bag，不需要 `rosbag filter` 修复索引；
- LLMDet-Large 至少输出 `chair/table/door` 三类测试标签；
- 标签以可读图标/文字叠加在 occupancy grid 上；
- 不要求 AMCL 收敛，也不要求机器人自动驶向 POI。
