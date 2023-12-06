#!/usr/bin/python3

import rospy
from geometry_msgs.msg import PointStamped


FRAME_NAME = "tag_map"
WAYPOINT_TOPIC = "/tag_map_waypoint"


def get_float_input(prompt):
    while True:
      try:
        value = float(input(prompt))
        return value
      except ValueError:
          print("Please enter a valid floating-point number.")


def user_input_waypoints():
    print(f"Enter XYZ for the waypoint in the {FRAME_NAME} frame:")
    x = get_float_input("Enter X: ")
    y = get_float_input("Enter Y: ")
    z = get_float_input("Enter Z: ")
    return x, y, z


if __name__ == "__main__":
    rospy.init_node("user_input_waypoints")

    pub = rospy.Publisher(WAYPOINT_TOPIC, PointStamped, queue_size=1)

    while not rospy.is_shutdown():
      x, y, z = user_input_waypoints()

      print(f"Publishing waypoint at {x:.1f} {y:.1f} {z:.1f} in the {FRAME_NAME} frame")

      waypoint_point_stamped_msg = PointStamped()
      waypoint_point_stamped_msg.header.stamp = rospy.Time.now()
      waypoint_point_stamped_msg.header.frame_id = FRAME_NAME
      waypoint_point_stamped_msg.point.x = x
      waypoint_point_stamped_msg.point.y = y
      waypoint_point_stamped_msg.point.z = z

      pub.publish(waypoint_point_stamped_msg)
