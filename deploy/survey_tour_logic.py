#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pure, ROS-free control logic for the openloop AMCL waypoint tour.

This module is the single home of the forward-sign convention and the
head-cone lidar filter.  It is imported by ``survey_tour_20260827.py`` on
the robot and by ``src/jetbot_pro/tests/test_survey_tour_logic.py`` on the
PC, so a sign or cone regression fails CI before it can strand a survey.

Deployment: copy this file next to the tour script (same directory), see
``start_nav_survey.sh`` / ``rerun_tour.sh``.
"""
from __future__ import print_function

import math

# --- Forward sign convention -------------------------------------------
# POSITIVE /cmd_vel.linear.x drives toward the robot head (REP-103).
# Proven by bag 11 (TEB +x advanced the AMCL pose through waypoints) and
# falsified the other way by bag 18 (-0.12 drifted away from every goal).
# joystick.yaml's negative scale_linear compensates the gamepad's inverted
# stick axis and says nothing about the chassis.  Change FORWARD_SIGN only
# here, never in jetbot.cpp / joystick.yaml / TEB.
FORWARD_SIGN = 1.0
FORWARD_SPEED = 0.12
FORWARD_VX = FORWARD_SIGN * FORWARD_SPEED

TURN_WZ = 0.40
GOAL_TOL_M = 0.28
YAW_ALIGN_RAD = 0.45
# Brake threshold for the head cone.  Must stay comfortably ABOVE
# scan_clip.py's ~min_range_m (0.22): body-sector returns below that floor
# are clipped, while the kept head sector preserves real near obstacles.
# At 0.12 m/s the robot closes ~1.2 cm per scan, so 0.30 brakes well
# before the blind
# zone.  Bag 17's failure was the 360-degree minimum, not this value.
LIDAR_STOP_M = 0.30

# base_footprint -> laser_frame yaw is pi: the head lies near +-pi in scan
# angle.  The cone half-width keeps side/rear returns (sofa legs) out of
# the braking decision.
LASER_YAW_OFFSET = math.pi
FRONT_CONE_HALF_RAD = 0.70
FRONT_RANGE_CAP_M = 8.0
FRONT_RANGE_FLOOR_M = 0.18

# In-place turns sweep the head through one forward shoulder before the
# 0.70 braking cone would ever see the obstacle.  The shoulder half-width
# must stay within scan_clip_core's HEAD_KEEP_HALF_RAD (1.05) or the gate
# would ask for distances the clip already rewrote to range_max
# (test_scan_clip_sectors.py locks that ordering).
SHOULDER_HALF_RAD = 1.05
# Turn gate threshold: above the 0.22 clip blind zone plus margin, and not
# above LIDAR_STOP_M -- a turn may proceed where forward drive would brake,
# never the other way around.
TURN_STOP_M = 0.28

# Runtime AMCL drift guard.  These are heartbeat-scale samples, not the 10 Hz
# command loop: three consecutive increases of at least 4 cm indicate that a
# positive forward command is moving the estimated pose away from its goal.
DRIFT_REQUIRED_INCREASES = 3
DRIFT_MIN_STEP_M = 0.04


class DistanceDriftTracker(object):
    """Pure state machine for detecting sustained AMCL distance divergence."""

    def __init__(self, required_increases=DRIFT_REQUIRED_INCREASES,
                 min_step_m=DRIFT_MIN_STEP_M):
        if required_increases < 1 or min_step_m <= 0.0:
            raise ValueError("invalid AMCL drift thresholds")
        self.required_increases = int(required_increases)
        self.min_step_m = float(min_step_m)
        self.previous = None
        self.consecutive_increases = 0

    def reset(self):
        """Break a run when the robot is not actively driving forward."""
        self.previous = None
        self.consecutive_increases = 0

    def observe(self, distance_m):
        """Return true after sustained, meaningful distance increases."""
        distance = float(distance_m)
        if math.isnan(distance) or math.isinf(distance) or distance < 0.0:
            self.previous = None
            self.consecutive_increases = 0
            return False
        if self.previous is not None:
            if distance - self.previous >= self.min_step_m:
                self.consecutive_increases += 1
            else:
                self.consecutive_increases = 0
        self.previous = distance
        return self.consecutive_increases >= self.required_increases


def amcl_drift_detected(distances, required_increases=DRIFT_REQUIRED_INCREASES,
                        min_step_m=DRIFT_MIN_STEP_M):
    """Return whether a finite distance sequence contains sustained drift."""
    tracker = DistanceDriftTracker(required_increases, min_step_m)
    return any(tracker.observe(distance) for distance in distances)


def forward_sign():
    """Sign of /cmd_vel.linear.x that moves the AMCL pose toward goals."""
    return FORWARD_SIGN


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quat(z, w):
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def angle_is_in_front_cone(scan_angle,
                           laser_yaw=LASER_YAW_OFFSET,
                           half_width=FRONT_CONE_HALF_RAD):
    """True when a scan angle points at the robot head, not tail/side."""
    return abs(wrap_angle(scan_angle + laser_yaw)) < half_width


def front_cone_min(ranges, angle_min, angle_increment, range_min, range_max):
    """Nearest valid return inside the head cone; 99.0 when cone is clear.

    Bag 17 died because the stop check used the 360-degree minimum and a
    sofa leg 28 cm BEHIND the robot froze the whole tour.  Only returns
    aimed at the head may brake forward motion.
    """
    cap = min(float(range_max), FRONT_RANGE_CAP_M)
    floor = max(float(range_min), FRONT_RANGE_FLOOR_M)
    nearest = None
    angle = float(angle_min)
    increment = float(angle_increment)
    for value in ranges:
        sample = float(value)
        if (not math.isnan(sample) and not math.isinf(sample) and
                floor <= sample <= cap and angle_is_in_front_cone(angle)):
            if nearest is None or sample < nearest:
                nearest = sample
        angle += increment
    return nearest if nearest is not None else 99.0


def turn_sweep_min(ranges, angle_min, angle_increment, range_min, range_max,
                   turn_sign):
    """Nearest valid return in the shoulder an in-place turn sweeps into.

    ``turn_sign`` follows cmd_vel.angular.z: positive (CCW/left) sweeps the
    head toward positive base bearings, negative toward negative ones, zero
    means no turn and returns 99.0.  Only the forward shoulder strictly
    between bearing 0 and +-SHOULDER_HALF_RAD gates the turn: bag 17's sofa
    leg at bearing +-pi (28 cm behind) never appears here, so this gate can
    never re-freeze a whole tour the way the old 360-degree minimum did.
    """
    if turn_sign == 0.0:
        return 99.0
    cap = min(float(range_max), FRONT_RANGE_CAP_M)
    floor = max(float(range_min), FRONT_RANGE_FLOOR_M)
    nearest = None
    angle = float(angle_min)
    increment = float(angle_increment)
    for value in ranges:
        sample = float(value)
        if (not math.isnan(sample) and not math.isinf(sample) and
                floor <= sample <= cap):
            bearing = wrap_angle(angle + LASER_YAW_OFFSET)
            if turn_sign > 0.0:
                in_sweep = 0.0 < bearing < SHOULDER_HALF_RAD
            else:
                in_sweep = -SHOULDER_HALF_RAD < bearing < 0.0
            if in_sweep and (nearest is None or sample < nearest):
                nearest = sample
        angle += increment
    return nearest if nearest is not None else 99.0


def goal_reached(px, py, gx, gy, tolerance=GOAL_TOL_M):
    return math.hypot(gx - px, gy - py) <= tolerance


def steer_command(px, py, yaw, gx, gy):
    """Return (vx, wz, state) steering the pose toward the goal.

    Pure pursuit-lite: rotate in place until the heading error is inside
    YAW_ALIGN_RAD, then drive FORWARD_VX with proportional yaw trim.
    """
    desired = math.atan2(gy - py, gx - px)
    err = wrap_angle(desired - yaw)
    if abs(err) > YAW_ALIGN_RAD:
        return 0.0, (TURN_WZ if err > 0.0 else -TURN_WZ), "turn"
    return FORWARD_VX, max(-0.25, min(0.25, 0.8 * err)), "drive"
