# Third-party components

This project builds on the following upstream software and datasets. None of
them are redistributed in this repository unless explicitly stated.

## Patched, not redistributed

- **waveshare/jetbot_pro** — https://github.com/waveshare/jetbot_pro
  Robot base package for the Waveshare JetBot Pro. Upstream declares no
  open-source license (`<license>TODO</license>`), so its source is **not**
  included here. `patches/jetbot_pro_local_changes.patch` contains only the
  local modifications as a diff against commit
  `b76d483fbd5ceb41a66be4945ed2fc712b5d36a4`; `overlay/jetbot_pro/` contains
  only original files written for this project.

- **ros-drivers/gscam** (BSD) — https://github.com/ros-drivers/gscam
  GStreamer camera driver for ROS. Patched against commit
  `1b0b8e51b91522edadd8399d4ad920f8afbc487d`. `overlay/gscam/` contains only
  original files written for this project.

- **dusty-nv/ros_deep_learning** (NVIDIA, MIT) —
  https://github.com/dusty-nv/ros_deep_learning
  `overlay/jetbot_pro/patches/ros_deep_learning_preserve_input_header.patch`
  is a small diff preserving input timestamps.

## Used at runtime, not included

- **ROS1 Melodic** (BSD and others) — https://www.ros.org/
- **gmapping / AMCL / move_base / TEB local planner / Cartographer** — standard
  ROS SLAM & navigation stacks, installed from apt.
- **LLMDet / GroundingDINO** (Apache-2.0) — open-vocabulary object detection
  models used offline on the PC; weights are downloaded separately.
- **stereo_image_proc** (BSD) — offline disparity computation.

## Referenced datasets (not included)

- **Bormann et al. room-segmentation benchmark** (Freiburg building maps) —
  used only to sanity-check the room-segmentation code
  (`tools/room_segment_bormann.py`). Obtain it from the original publication's
  distribution; the map images are not redistributed here.

## Hardware documentation

- Waveshare **JetBot Pro** and **IMX219-83 stereo camera** documentation:
  https://www.waveshare.com/wiki/ — mechanical drawings and vendor materials
  are not redistributed in this repository.
