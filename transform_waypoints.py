#!/usr/bin/python3

import numpy as np
import rospy
import tf
from geometry_msgs.msg import PointStamped


ODOM_FRAME_NAME = "odom"
MAP_FRAME_NAME = "map"
TAG_MAP_FRAME_NAME = "tag_map"

WAYPOINT_PUB_TOPIC = "/mp_waypoint"
WAYPOINT_SUB_TOPIC = "/tag_map_waypoint"


def lookupLatestTFMat(listener, target_frame, source_frame, wait_duration=0.5):
    try:
        listener.waitForTransform(
            target_frame, source_frame, rospy.Time(), rospy.Duration(wait_duration)
        )

        trans, quat = listener.lookupTransform(
            target_frame, source_frame, rospy.Time(0)
        )
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        raise Exception(
            "Could not get the latest TF from {} to {}".format(
                source_frame, target_frame
            )
        )

    tf_mat = tf.transformations.quaternion_matrix(quat)
    tf_mat[:3, 3] = trans

    return tf_mat


class WaypointTransformer:
    def __init__(
        self,
        odom_frame,
        map_frame,
        tag_map_frame,
        waypoint_pub_topic,
        waypoint_sub_topic,
    ):
        self.odom_frame = odom_frame
        self.map_frame = map_frame
        self.tag_map_frame = tag_map_frame

        self.tf_listener = tf.TransformListener()

        # Lookup fixed transform from tag map to map frame
        self.tag_map_to_map_mat = lookupLatestTFMat(
            self.tf_listener, map_frame, tag_map_frame
        )

        # Initialize transform from map to odom frame
        self.map_to_odom_mat = lookupLatestTFMat(
            self.tf_listener, odom_frame, map_frame
        )

        # ROS timer which updates the transform from map to odom frame
        self.update_map_to_odom_timer = rospy.Timer(
            rospy.Duration(0.5), self.update_map_to_odom
        )

        self.sub = rospy.Subscriber(
            waypoint_sub_topic, PointStamped, self.set_waypoint_cb
        )
        self.pub = rospy.Publisher(waypoint_pub_topic, PointStamped, queue_size=1)

        rospy.loginfo("Started waypoint transformer")

    def update_map_to_odom(self, event):
        try:
            self.map_to_odom_mat = lookupLatestTFMat(
                self.tf_listener, self.odom_frame, self.map_frame, wait_duration=0.1
            )
        except Exception as e:
            rospy.logwarn(
                "Could not update map to odom frame TF".format(
                    self.map_frame, self.odom_frame, e
                )
            )
            return

        self.tag_map_to_odom_mat = self.map_to_odom_mat @ self.tag_map_to_map_mat
        self.publish_waypoint()

    def set_waypoint_cb(self, msg):
        self.waypoint_tag_map = np.array([msg.point.x, msg.point.y, msg.point.z, 1])
        rospy.loginfo(
            f"received waypoint new waypoint: {self.waypoint_tag_map[0]:.1f}, {self.waypoint_tag_map[1]:.1f}, {self.waypoint_tag_map[2]:.1f}"
        )
        self.publish_waypoint()

    def publish_waypoint(self):
        if not hasattr(self, "waypoint_tag_map"):
            return
        
        waypoint_odom = self.tag_map_to_odom_mat @ self.waypoint_tag_map
        point_stamped_msg = PointStamped()
        point_stamped_msg.header.stamp = rospy.Time.now()
        point_stamped_msg.header.frame_id = self.odom_frame
        point_stamped_msg.point.x = waypoint_odom[0]
        point_stamped_msg.point.y = waypoint_odom[1]
        point_stamped_msg.point.z = waypoint_odom[2]
        self.pub.publish(point_stamped_msg)


if __name__ == "__main__":
    rospy.init_node("waypoint_transformer")

    wp_transformer = WaypointTransformer(
        odom_frame=ODOM_FRAME_NAME,
        map_frame=MAP_FRAME_NAME,
        tag_map_frame=TAG_MAP_FRAME_NAME,
        waypoint_pub_topic=WAYPOINT_PUB_TOPIC,
        waypoint_sub_topic=WAYPOINT_SUB_TOPIC,
    )

    rospy.spin()
