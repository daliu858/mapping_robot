# floorplan — 占据栅格图 → 带标签建筑平面图

单文件 pipeline:`make_floorplan.py`。几何全部由确定性算法完成,
LLM(可选)只负责给房间起英文名,**改不了任何坐标、面积、墙线**。

## 用法

```bash
cd floorplan
python make_floorplan.py --map 你的地图.yaml --objects 你的语义.yaml
python make_floorplan.py --map 你的地图.yaml --llm   # 额外调一次 LLM 给房间起名
```

> 注:脚本内置的默认 `--map` 路径指向作者本机的私有地图(家庭实测数据,
> 未随仓库分发),因此**必须显式传入自己的地图**;`doors_sample.yaml`
> 是虚构坐标的格式示例。

依赖:`opencv-python numpy scipy scikit-image matplotlib pyyaml ezdxf`
(`--llm` 另需 `requests` 和 `.env` 里的 `ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL`)。

## 输出(out/)

| 文件 | 说明 |
|---|---|
| `floorplan.png` | 成品平面图(房间填色 + 名字 + 面积 + 门符号 + 比例尺 + 指北针) |
| `floorplan.svg` | 同上,矢量版 |
| `floorplan.dxf` | CAD 版,分层 WALLS / DOORS / LABELS,单位米 |
| `debug_rooms.png` | 分割调试图:每个房间一个颜色,红圈=门锚点。**出图不对先看它** |

## 流程

1. 读 map_server 的 PGM+YAML(0=墙,205=未知,254=自由)
2. 清噪:孤立墙点过滤 + 只留最大自由空间连通域(玻璃漏光的外部区域全丢)
3. Manhattan 对齐:Hough 找墙主方向,整图旋正
4. 分房间:距离变换 + watershed,门洞(半宽 < `--clearance` 0.45m)处自然断开
5. 矢量化:轮廓 → 多边形 → 近轴边掰直(斜墙保留)
6. 门:语义 YAML 里 label 含 `door` 的点吸附到最近墙边,画开门弧;
   离墙超 1m 的门画红叉警告,**不瞎画**
7. 起名:默认启发式(物体投票 → Room A/B/C…);`--llm` 时把
   带数字编号的预览图发给 Claude 要一个 `{id: English name}` 的 JSON,失败自动回退
8. 出 PNG/SVG/DXF

## 调参

- 房间分多了/少了 → `--clearance`(默认 0.45,调大切得更碎,调小更容易连成一间)
- 碎块太多 → `--min-room`(默认 2.0 m²,小于它并入邻居)
- 换真实检测数据 → `--objects`,格式即 `slam-car-semantic-objects-v1`
  (label/x/y,床、沙发等物体会参与房间起名投票,并以小圆点画在图上)

## 教训纪念

本 pipeline 的第一条设计原则来自「30 个门」事件:
**要求坐标精确的活一律不给 LLM;要求判断力的活(起名)才给 LLM。**
