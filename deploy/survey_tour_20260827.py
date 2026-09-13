#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Drive AMCL survey waypoints with cmd_vel.  No move_base / TEB.

Forward sign convention (decided 2026-08-27, do not "fix" elsewhere)
--------------------------------------------------------------------
POSITIVE /cmd_vel.linear.x is physical forward (robot head, REP-103).

Evidence:
  * Bag 11: TEB published +x and the AMCL pose genuinely advanced through
    6/27 waypoints on the map.  Odometry comes from the MCU (jetbot.cpp),
    not from cmd_vel integration, so this is ground truth.
  * Bag 18: this script drove -0.12 and the AMCL pose drifted AWAY from
    every waypoint (start (-1.95, 0.35) -> (0.79, 4.19)).
  * The camera mount is at base x=+0.105: the head is on +x.

joystick.yaml's ``scale_linear: -0.15`` is a *gamepad axis* correction
(stick pushed forward reports a negative axis value on this pad), NOT a
chassis inversion.  Copying that sign here was the bag-18 root cause.
The convention lives in ``forward_sign()`` below and is unit-tested by
``src/jetbot_pro/tests/test_survey_tour_logic.py``.

Lidar gating: base_footprint -> laser_frame yaw is pi, so the robot head
sits near +-pi in scan angles.  Braking uses only that head cone
(``front_cone_min()``); a sofa leg 28 cm behind the robot must never
freeze the tour again (bag 17 root cause).  In-place turns are gated by
``turn_sweep_min()`` on the forward shoulder the head is about to sweep
into -- waiting, like a blocked front, never skips the whole tour.
"""
from __future__ import print_function

import math
import sys

from survey_tour_logic import (FORWARD_VX, GOAL_TOL_M, LIDAR_STOP_M,
                               TURN_STOP_M, TURN_WZ, YAW_ALIGN_RAD,
                               DistanceDriftTracker, forward_sign,
                               front_cone_min, goal_reached, steer_command,
                               turn_sweep_min, wrap_angle, yaw_from_quat)

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

WAYPOINTS = (
    (-1.975, -0.425),
    (-2.775, -0.425),
    (-3.575, -0.425),
    (-1.175, -0.425),
    (-1.175, -1.225),
    (-1.175, -2.025),
    (-1.175, -2.825),
    (-1.175, -3.625),
    (-0.375, -5.225),
    (-1.175, -5.225),
    (-0.375, -6.025),
)

SEED_POSE = (-1.975, 0.375, -1.57)
GOAL_TIMEOUT_S = 50.0
RECORD_WAIT_S = 90.0
HEARTBEAT_S = 2.0


def _yaw_quat(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _recording_started(payload):
    text = getattr(payload, "data", "") or ""
    return '"state":"started"' in text.replace(" ", "")


class OpenloopTour(object):
    def __init__(self):
        self.pose = None
        self.safety_stop = True
        self.min_front = 99.0
        self.min_sweep_ccw = 99.0
        self.min_sweep_cw = 99.0
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self._on_pose)
        rospy.Subscriber("/safety_stop", Bool, self._on_safety)
        rospy.Subscriber("/scan", LaserScan, self._on_scan)

    def _on_pose(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_from_quat(o.z, o.w))

    def _on_safety(self, msg):
        self.safety_stop = bool(msg.data)

    def _on_scan(self, msg):
        args = (msg.ranges, float(msg.angle_min), float(msg.angle_increment),
                float(msg.range_min), float(msg.range_max))
        self.min_front = front_cone_min(*args)
        self.min_sweep_ccw = turn_sweep_min(*(args + (1.0,)))
        self.min_sweep_cw = turn_sweep_min(*(args + (-1.0,)))

    def stop(self):
        self.cmd_pub.publish(Twist())

    def drive(self, vx, wz):
        """Publish one command; forward motion and in-place turns are gated.

        Forward drive brakes on the head cone (front_cone_min).  An
        in-place turn brakes on the shoulder it sweeps into
        (turn_sweep_min): the head crosses that sector before the braking
        cone would see the obstacle.  Both gates WAIT, they never skip.
        """
        if self.safety_stop:
            self.stop()
            return False
        if abs(vx) > 0.02 and self.min_front < LIDAR_STOP_M:
            self.stop()
            return False
        if abs(vx) <= 0.02 and abs(wz) > 0.02:
            sweep = self.min_sweep_ccw if wz > 0.0 else self.min_sweep_cw
            if sweep < TURN_STOP_M:
                self.stop()
                return False
        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)
        return True

    def wait_ready(self):
        deadline = rospy.Time.now() + rospy.Duration(RECORD_WAIT_S)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                status = rospy.wait_for_message(
                    "/survey/recording_status", String, timeout=2.0)
            except rospy.ROSException:
                continue
            if _recording_started(status):
                rospy.loginfo("survey bag is recording; starting openloop tour")
                break
        else:
            raise RuntimeError("survey recording did not start")
        deadline = rospy.Time.now() + rospy.Duration(20.0)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.pose is not None and not self.safety_stop:
                rospy.loginfo("safety_stop is clear; pose x=%.3f y=%.3f",
                              self.pose[0], self.pose[1])
                return
            rospy.sleep(0.2)
        raise RuntimeError("no AMCL pose or safety_stop stayed active")

    def seed_pose(self):
        pub = rospy.Publisher(
            "/initialpose", PoseWithCovarianceStamped, queue_size=1, latch=True)
        rospy.sleep(0.4)
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = SEED_POSE[0]
        msg.pose.pose.position.y = SEED_POSE[1]
        qz, qw = _yaw_quat(SEED_POSE[2])
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        cov = [0.0] * 36
        cov[0] = 0.40
        cov[7] = 0.40
        cov[35] = 0.12
        msg.pose.covariance = cov
        pub.publish(msg)
        rospy.loginfo("seeded /initialpose x=%.3f y=%.3f",
                      SEED_POSE[0], SEED_POSE[1])
        rospy.sleep(2.0)

    def go_to(self, index, total, x, y, timeout_s):
        """Steer toward (x, y).  Blocked front means WAIT, not skip.

        The bounded timeout is the only way a goal is abandoned; within it
        the tour keeps station in front of an obstacle (people/pets move).
        A heartbeat log every HEARTBEAT_S makes 'waiting' visibly different
        from 'dead' -- the bag-18 operator could not tell them apart.
        """
        deadline = rospy.Time.now() + rospy.Duration(timeout_s)
        rate = rospy.Rate(10)
        next_beat = rospy.Time.now()
        state = "init"
        vx = 0.0
        drift = DistanceDriftTracker()
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.pose is None:
                state = "no-pose"
                drift.reset()
                rate.sleep()
                continue
            px, py, yaw = self.pose
            dist = math.hypot(x - px, y - py)
            if goal_reached(px, py, x, y):
                self.stop()
                return True
            vx, wz, state = steer_command(px, py, yaw, x, y)
            if not self.drive(vx, wz):
                if self.safety_stop:
                    state = "safety-stop"
                elif state == "turn":
                    state = "turn-blocked"
                else:
                    state = "front-blocked"
                vx = 0.0
            if rospy.Time.now() >= next_beat:
                if state == "drive" and vx > 0.0:
                    if drift.observe(dist):
                        self.stop()
                        rospy.logwarn(
                            "amcl-drift goal %d/%d: forward vx=%.2f "
                            "distance increased for %d heartbeats (dist=%.2f)",
                            index, total, vx, drift.consecutive_increases,
                            dist)
                        return "drift"
                else:
                    drift.reset()
                rospy.loginfo(
                    "goal %d/%d [%s] pose=(%.2f,%.2f,%.2f) dist=%.2f "
                    "min_front=%.2f vx=%.2f",
                    index, total, state, px, py, yaw, dist,
                    self.min_front, vx)
                next_beat = rospy.Time.now() + rospy.Duration(HEARTBEAT_S)
            rate.sleep()
        self.stop()
        if self.pose is None:
            return False
        return goal_reached(self.pose[0], self.pose[1], x, y)

    def complete(self):
        rospy.wait_for_service("/offline_recorder/complete", timeout=10.0)
        response = rospy.ServiceProxy("/offline_recorder/complete", Trigger)()
        rospy.loginfo("complete: success=%s message=%s",
                      response.success, response.message)
        return bool(response.success)


def _abort_recorder(reason):
    """Fail-close the recording transaction when the tour cannot finish.

    Best effort only: if the recorder is already gone (roslaunch teardown),
    its own _on_shutdown has fail-closed the bag and this call just logs.
    """
    try:
        rospy.wait_for_service("/offline_recorder/fail", timeout=5.0)
        response = rospy.ServiceProxy("/offline_recorder/fail", Trigger)()
        rospy.logwarn("recorder abort (%s): success=%s message=%s",
                      reason, response.success, response.message)
    except Exception as error:
        rospy.logwarn("recorder abort skipped (%s): %s", reason, error)


def _crash_stop(tour):
    """Repeat the zero-twist so a lost message cannot leave wheels spinning."""
    try:
        for _ in range(5):
            tour.stop()
            rospy.sleep(0.1)
    except Exception:
        pass


def _tour_main(tour):
    tour.wait_ready()
    tour.seed_pose()
    reached = 0
    skipped = []
    total = len(WAYPOINTS)
    for index, (x, y) in enumerate(WAYPOINTS):
        if rospy.is_shutdown():
            tour.stop()
            return 1
        rospy.loginfo("goal %d/%d x=%.3f y=%.3f min_front=%.2f",
                      index + 1, total, x, y, tour.min_front)
        result = tour.go_to(index + 1, total, x, y, GOAL_TIMEOUT_S)
        if result == "drift":
            raise RuntimeError("amcl-drift: pose moved away from waypoint")
        if result:
            reached += 1
            rospy.loginfo("goal %d reached", index + 1)
        else:
            skipped.append(index + 1)
            rospy.logwarn("goal %d timed out after %.0fs; moving on",
                          index + 1, GOAL_TIMEOUT_S)
    tour.stop()
    rospy.sleep(1.0)
    rospy.loginfo("tour finished reached=%d/%d skipped=%s",
                  reached, total, skipped or "-")
    # A tour that reached zero waypoints still recorded real sensor data
    # under a healthy AMCL gate; completing the bag preserves evidence for
    # diagnosis instead of raising the bag-17 FATAL that hid it.
    if reached < 1:
        rospy.logwarn("tour reached 0 waypoints; completing bag as evidence")
    if not tour.complete():
        raise RuntimeError("offline_recorder complete failed")
    return 0 if reached >= 1 else 2


def main():
    rospy.init_node("survey_tour")
    assert forward_sign() > 0, "forward sign regressed; see module docstring"
    tour = OpenloopTour()
    try:
        return _tour_main(tour)
    except rospy.ROSInterruptException:
        # Operator interrupt / roslaunch teardown: leave the transaction to
        # the recorder's own shutdown handling so rerun_tour.sh can rerun
        # the tour against a recording that is still healthy.
        raise
    except Exception as error:
        rospy.logfatal("tour crashed: %s", error)
        _crash_stop(tour)
        _abort_recorder("survey tour crashed before completion: %s" % error)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(1)
    except Exception as error:
        # Only reachable for failures before the tour object exists
        # (forward-sign assert, node init, constructor); main() handles
        # everything after that and returns instead of re-raising.
        rospy.logfatal("%s", error)
        _abort_recorder("survey tour failed before starting: %s" % error)
        sys.exit(1)
