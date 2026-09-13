# SLAM 语义地图闭环验收单

日期：2026-08-04

## 验收目标

一次受监督的家庭巡视完成以下闭环：

1. gmapping 生成完整占据栅格；
2. 同一趟录下双目、LiDAR、里程计和完整 TF；
3. 电脑端 LLMDet-Large 检测 `chair`、`table`、`door`；
4. 双目深度与录制 TF 将标签投影到 occupancy grid；
5. 输出供学生查看的最终语义地图。

AMCL、`move_base`、TEB 和语义自动巡航均不属于本次验收。

## 代码验收

- [x] 正式入口为 `slam_capture.launch`，启动树不含导航栈；
- [x] gmapping 异常时 fail-closed，不静默重启并重置地图；
- [x] 录包先写 `.recording.bag`，关闭并只读验证后才原子提交；
- [x] 最终地图保存、地图哈希和唯一完成事务写入 bag；
- [x] 离线预检接受 gmapping TF，不硬性要求 `/amcl_pose`；
- [x] 删除语义导航、自动巡航、反应式游走和旧两趟录包代码；
- [x] PC/Jetson 侧回归测试通过；
- [ ] 新 overlay 在 Jetson 上编译并完成 ROS 启动预检；
- [ ] 一次完整家庭巡视成功提交 bag + map + sidecar；
- [ ] PC 离线预检通过；
- [ ] 生成 `chair`、`table`、`door` 最终标签覆盖图。

## 真车成功条件

同一运行编号下必须同时存在：

- `<RUN>.bag`
- `<RUN>_map.yaml`
- `<RUN>_map.pgm`
- `<RUN>.complete.json`

并且 `offline_preflight.py` 验证以下 TF：

```text
map -> odom -> base_footprint -> stereo_left_optical
```

如果只留下 `.recording.bag` 或 `.failed.json`，该次运行失败，不得进入离线检测。

## TEB 状态

滚动 local costmap 已移除 StaticLayer，避免 `map -> odom` 修正把静态边界拖入局部代价地图；但 TEB 尚未完成全屋稳定性验收。由于正式闭环完全不加载 TEB，这不阻塞本次验收，也不得对外宣称 TEB 已全面修复。
