#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class JoyTeleop(object):
    def __init__(self):
        self.axis_linear = int(rospy.get_param('~axis_linear', 1))
        self.axis_angular = int(rospy.get_param('~axis_angular', 0))

        self.scale_linear = float(rospy.get_param('~scale_linear', 0.35))
        self.scale_angular = float(rospy.get_param('~scale_angular', 1.2))
        self.scale_linear_turbo = float(rospy.get_param('~scale_linear_turbo', 0.65))
        self.scale_angular_turbo = float(rospy.get_param('~scale_angular_turbo', 2.0))

        require_enable = rospy.get_param('~require_enable_button', True)
        if isinstance(require_enable, bool):
            self.require_enable_button = require_enable
        else:
            self.require_enable_button = str(require_enable).lower() in (
                '1', 'true', 'yes', 'on')
        self.enable_button = int(rospy.get_param('~enable_button', 4))
        self.enable_turbo_button = int(rospy.get_param('~enable_turbo_button', 5))

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber('/joy', Joy, self.joy_callback, queue_size=10)
        rospy.on_shutdown(self.stop)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def _axis(self, axes, index):
        if index < 0 or index >= len(axes):
            return 0.0
        return axes[index]

    def _button(self, buttons, index):
        if index < 0 or index >= len(buttons):
            return 0
        return buttons[index]

    def joy_callback(self, msg):
        enabled = 1 if not self.require_enable_button else self._button(msg.buttons, self.enable_button)
        turbo = self._button(msg.buttons, self.enable_turbo_button)

        twist = Twist()

        if enabled or turbo:
            linear_scale = self.scale_linear_turbo if turbo else self.scale_linear
            angular_scale = self.scale_angular_turbo if turbo else self.scale_angular

            twist.linear.x = self._axis(msg.axes, self.axis_linear) * linear_scale
            twist.angular.z = self._axis(msg.axes, self.axis_angular) * angular_scale

        self.cmd_pub.publish(twist)


if __name__ == '__main__':
    rospy.init_node('joy_teleop', anonymous=False)
    JoyTeleop()
    rospy.spin()
