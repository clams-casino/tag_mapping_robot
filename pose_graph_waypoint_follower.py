#!/usr/bin/python3

import numpy as np
import yaml

import rospy
import tf
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from pose_graph import PoseGraph


def lookupLatestTFMat(listener, target_frame, source_frame, wait_duration):
    try:
        listener.waitForTransform(
            target_frame, source_frame, rospy.Time(0), rospy.Duration(wait_duration)
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


class WaypointBuffer:
    def __init__(self, waypoints):
        self.waypoints = [np.array(wp) for wp in waypoints]

    def get_next_waypoint(self):
        try:
            return self.waypoints.pop(0)
        except IndexError:
            return None


class PoseGraphWaypointFollower:
    def __init__(
        self,
        params,
    ):
        # Store params
        self.params = params

        # Load pose graph
        self.pose_graph = PoseGraph.load(params["pose_graph_path"])

        # Store list of waypoints of the current path
        self.waypoint_buffer = WaypointBuffer([])

        # Flag to indicate if the goal has been reached
        self.goal_reached = False

        # Store frame names
        self.odom_frame = params["odom_frame_name"]
        self.map_frame = params["map_frame_name"]
        self.tag_map_frame = params["tag_map_frame_name"]

        # Setup TFs
        self.tf_listener = tf.TransformListener()

        self.tag_map_to_map_mat = lookupLatestTFMat(
            self.tf_listener, self.map_frame, self.tag_map_frame, wait_duration=5.0
        )

        self.map_to_odom_mat = lookupLatestTFMat(
            self.tf_listener, self.odom_frame, self.map_frame, wait_duration=5.0
        )

        # ROS timer which updates the transform from map to odom frame
        self.update_map_to_odom_timer = rospy.Timer(
            rospy.Duration(0.5), self.update_map_to_odom
        )

        # ROS subscribers and publishers
        self.odom_sub = rospy.Subscriber(
            params["odom_sub_topic"], PoseStamped, self.odom_callback
        )
        self.goal_sub = rospy.Subscriber(
            params["goal_sub_topic"], PointStamped, self.goal_callback
        )

        self.current_waypoint_pub = rospy.Publisher(
            params["waypoint_pub_topic"], PointStamped, queue_size=1
        )

        self.waypoint_markers_pub = rospy.Publisher(
            "/waypoint_markers", MarkerArray, queue_size=1
        )

        rospy.loginfo("Started pose graph waypoint follower")

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
        self.odom_to_tag_map_mat = np.linalg.inv(self.tag_map_to_odom_mat)
        self.publish_current_waypoint()

    def goal_callback(self, msg):
        if not hasattr(self, "current_loc_tag_map"):
            rospy.logwarn("Received goal before receiving current location")
            return

        goal_tag_map = np.array([msg.point.x, msg.point.y, msg.point.z])
        rospy.loginfo(
            f"Received new goal at {goal_tag_map[0]:.2f} {goal_tag_map[1]:.2f} {goal_tag_map[2]:.2f}"
        )

        nearest_goal_node = self.pose_graph.closest_node(goal_tag_map)
        start_node = self.pose_graph.closest_node(self.current_loc_tag_map)

        path = self.pose_graph.shortest_path(start_node, nearest_goal_node)

        waypoints_tag_map = self.pose_graph.nodes[path].tolist()
        waypoints_tag_map.append(goal_tag_map)
        self.waypoint_buffer = WaypointBuffer(waypoints_tag_map)

        self.current_waypoint_tag_map = self.waypoint_buffer.get_next_waypoint()
        self.goal_reached = False

        rospy.loginfo(
            f"Computed path with the following waypoints in the tag map frame:\n"
        )
        for i, waypoint in enumerate(self.waypoints_tag_map):
            rospy.loginfo(f"{i}: {waypoint[0]:.2f} {waypoint[1]:.2f} {waypoint[2]:.2f}")
        self.publish_waypoint_markers(waypoints_tag_map)

    def odom_callback(self, msg):
        assert (
            msg.header.frame_id == self.odom_frame
        ), f"odom_callback received message with frame_id {msg.header.frame_id}, expected {self.odom_frame}"

        if not hasattr(self, "odom_to_tag_map_mat"):
            return

        loc_odom = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 1]
        )
        self.current_loc_tag_map = (self.odom_to_tag_map_mat @ loc_odom)[:3]

        if not hasattr(self, "current_waypoint_tag_map"):
            return

        if (
            np.linalg.norm(self.current_loc_tag_map - self.current_waypoint_tag_map)
            < self.params["reached_waypoint_tolerance"]
        ):
            self.current_waypoint_tag_map = self.waypoint_buffer.get_next_waypoint()
            if self.current_waypoint_tag_map == None:
                if self.goal_reached:
                    return
                else:
                    rospy.loginfo("Reached goal")
                    self.goal_reached = True
            else:
                rospy.loginfo("Reached current waypoint, publishing next waypoint")
                self.publish_current_waypoint()

    def publish_current_waypoint(self):
        if not hasattr(self, "current_waypoint_tag_map"):
            rospy.logwarn("No current waypoint to publish")
            return

        waypoint_odom = self.tag_map_to_odom_mat @ self.current_waypoint_tag_map
        point_stamped_msg = PointStamped()
        point_stamped_msg.header.stamp = rospy.Time.now()
        point_stamped_msg.header.frame_id = self.odom_frame
        point_stamped_msg.point.x = waypoint_odom[0]
        point_stamped_msg.point.y = waypoint_odom[1]
        point_stamped_msg.point.z = waypoint_odom[2]
        self.current_waypoint_pub.publish(point_stamped_msg)

    def publish_waypoint_markers(self, waypoints_tag_map):
        marker_array_msg = MarkerArray()
        stamp = rospy.Time.now()
        for i, wp in enumerate(waypoints_tag_map):
            marker_msg = Marker()
            marker_msg.header.frame_id = self.tag_map_frame
            marker_msg.header.stamp = stamp
            marker_msg.ns = "waypoints"
            marker_msg.id = i
            marker_msg.type = Marker.SPHERE
            marker_msg.action = Marker.ADD
            marker_msg.pose.position.x = wp[0]
            marker_msg.pose.position.y = wp[1]
            marker_msg.pose.position.z = wp[2]
            marker_msg.pose.orientation.w = 1.0
            marker_msg.scale.x = 0.1
            marker_msg.scale.y = 0.1
            marker_msg.scale.z = 0.1
            marker_msg.color.a = 1.0
            marker_msg.color.r = 0.0
            marker_msg.color.g = 1.0
            marker_msg.color.b = 0.0
            marker_array_msg.markers.append(marker_msg)
        self.waypoint_markers_pub.publish(marker_array_msg)


if __name__ == "__main__":
    rospy.init_node("pose_graph_waypoint_follower")

    PARAMS_PATH = (
        "/home/rsl_admin/tag_mapping_robot/config/waypoint_follower_params.yaml"
    )

    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)

    wp_transformer = PoseGraphWaypointFollower(params)

    rospy.spin()
