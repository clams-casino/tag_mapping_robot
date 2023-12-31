#!/usr/bin/python3

import argparse
import yaml
import numpy as np

import rospy
import tf
from ros_numpy.point_cloud2 import array_to_pointcloud2
from geometry_msgs.msg import (
    PointStamped,
    PoseStamped,
    PoseWithCovarianceStamped,
)
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2

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

        self.pgn_array = np.zeros(
            len(self.pose_graph.nodes),
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
            ],
        )
        self.pgn_array["x"] = self.pose_graph.nodes[:, 0]
        self.pgn_array["y"] = self.pose_graph.nodes[:, 1]
        self.pgn_array["z"] = self.pose_graph.nodes[:, 2]

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

        self.tag_map_to_odom_mat = self.map_to_odom_mat @ self.tag_map_to_map_mat
        self.odom_to_tag_map_mat = np.linalg.inv(self.tag_map_to_odom_mat)

        # ROS timer which updates the transform from map to odom frame
        self.update_map_to_odom_timer = rospy.Timer(
            rospy.Duration(0.5), self.update_map_to_odom
        )

        # ROS timer which refreshes pose graph node visualization
        self.update_pose_graph_nodes_viz_timer = rospy.Timer(
            rospy.Duration(2.0), self.refresh_pose_graph_nodes_viz
        )

        # ROS subscribers and publishers
        self.odom_sub = rospy.Subscriber(
            params["odom_sub_topic"],
            PoseWithCovarianceStamped,
            self.odom_callback,
            queue_size=1,
        )
        self.goal_sub = rospy.Subscriber(
            params["goal_sub_topic"], PointStamped, self.goal_callback, queue_size=1
        )
        self.rviz_nav_goal_sub = rospy.Subscriber(
            params["rviz_nav_goal_sub"],
            PoseStamped,
            self.rviz_nav_goal_callback,
            queue_size=1,
        )

        self.current_waypoint_pub = rospy.Publisher(
            params["waypoint_pub_topic"], PointStamped, queue_size=1
        )

        self.waypoint_path_viz_pub = rospy.Publisher(
            params["waypoint_path_viz_pub_topic"], Path, queue_size=1
        )
        self.pose_graph_nodes_viz_pub = rospy.Publisher(
            params["pose_graph_nodes_viz_pub_topic"], PointCloud2, queue_size=1
        )

        rospy.loginfo("Started pose graph waypoint follower")

    def set_goal(self, goal_tag_map):
        if not hasattr(self, "current_loc_tag_map"):
            rospy.logwarn("Cannot set goal before receiving current location")
            return

        nearest_goal_node = self.pose_graph.closest_node(goal_tag_map)
        start_node = self.pose_graph.closest_node(self.current_loc_tag_map)

        path = self.pose_graph.shortest_path(start_node, nearest_goal_node)

        if len(path) == 1:
            # If the nearest node in the pose graph is the same for the start and goal,
            # then we might as well just go straight to the goal
            # We know for sure that the free space connectivity of the pose graph does not help here
            waypoints_tag_map = [goal_tag_map]
        else:
            waypoints_tag_map = self.pose_graph.nodes[path].tolist() + [goal_tag_map]
        self.waypoint_buffer = WaypointBuffer(waypoints_tag_map)

        self.current_waypoint_tag_map = self.waypoint_buffer.get_next_waypoint()
        self.goal_reached = False

        rospy.loginfo(
            f"Computed path with the following waypoints in the tag map frame:\n"
        )
        for i, waypoint in enumerate(waypoints_tag_map):
            rospy.loginfo(f"{i}: {waypoint[0]:.2f} {waypoint[1]:.2f} {waypoint[2]:.2f}")
        self.publish_waypoints_path_viz(waypoints_tag_map)

    def goal_callback(self, msg: PointStamped):
        goal_tag_map = np.array([msg.point.x, msg.point.y, msg.point.z])
        rospy.loginfo(
            f"Received new goal at {goal_tag_map[0]:.2f} {goal_tag_map[1]:.2f} {goal_tag_map[2]:.2f}"
        )
        self.set_goal(goal_tag_map)

    def rviz_nav_goal_callback(self, msg: PoseStamped):
        if not msg.header.frame_id == self.odom_frame:
            rospy.logwarn(
                f"rviz_nav_goal_callback received message with frame_id {msg.header.frame_id}, expected {self.odom_frame}"
            )
            return

        goal_odom = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                1,
            ]
        )
        goal_tag_map = (self.odom_to_tag_map_mat @ goal_odom)[:3]

        rospy.loginfo(
            f"Received new goal from RViz at {goal_tag_map[0]:.2f} {goal_tag_map[1]:.2f} {goal_tag_map[2]:.2f}"
        )
        self.set_goal(goal_tag_map)

    def odom_callback(self, msg: PoseWithCovarianceStamped):
        assert (
            msg.header.frame_id == self.odom_frame
        ), f"odom_callback received message with frame_id {msg.header.frame_id}, expected {self.odom_frame}"

        loc_odom = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                1,
            ]
        )
        self.current_loc_tag_map = (self.odom_to_tag_map_mat @ loc_odom)[:3]

        if not hasattr(self, "current_waypoint_tag_map"):
            return

        if self.goal_reached:
            return

        dist_to_current_waypoint = np.linalg.norm(
            # only consider distance in x and y
            (self.current_loc_tag_map - self.current_waypoint_tag_map)[:2]
        )
        if dist_to_current_waypoint < self.params["reached_waypoint_tolerance"]:
            # update the waypoint path visualization
            self.publish_waypoints_path_viz(self.waypoint_buffer.waypoints)

            self.current_waypoint_tag_map = self.waypoint_buffer.get_next_waypoint()
            if self.current_waypoint_tag_map is None:
                rospy.loginfo("Reached goal")
                self.goal_reached = True
            else:
                rospy.loginfo("Reached current waypoint, publishing next waypoint")
                self.publish_current_waypoint()

    def update_map_to_odom(self, event):
        try:
            self.map_to_odom_mat = lookupLatestTFMat(
                self.tf_listener, self.odom_frame, self.map_frame, wait_duration=0.01
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

    def publish_current_waypoint(self):
        if not hasattr(self, "current_waypoint_tag_map"):
            rospy.logwarn("No current waypoint to publish")
            return

        if self.goal_reached:
            return

        waypoint_odom = self.tag_map_to_odom_mat @ np.array(
            [
                self.current_waypoint_tag_map[0],
                self.current_waypoint_tag_map[1],
                self.current_waypoint_tag_map[2],
                1,
            ]
        )
        point_stamped_msg = PointStamped()
        point_stamped_msg.header.stamp = rospy.Time.now()
        point_stamped_msg.header.frame_id = self.odom_frame
        point_stamped_msg.point.x = waypoint_odom[0]
        point_stamped_msg.point.y = waypoint_odom[1]
        point_stamped_msg.point.z = waypoint_odom[2]
        self.current_waypoint_pub.publish(point_stamped_msg)

    def publish_waypoints_path_viz(self, waypoints_tag_map):
        path_msg = Path()
        path_msg.header.frame_id = self.tag_map_frame
        path_msg.header.stamp = rospy.Time.now()
        for i, wp in enumerate(waypoints_tag_map):
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self.tag_map_frame
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x = wp[0]
            pose_msg.pose.position.y = wp[1]
            pose_msg.pose.position.z = wp[2]
            path_msg.poses.append(pose_msg)
        self.waypoint_path_viz_pub.publish(path_msg)

    def refresh_pose_graph_nodes_viz(self, event):
        self.pose_graph_nodes_viz_pub.publish(
            array_to_pointcloud2(
                self.pgn_array,
                stamp=rospy.Time.now(),
                frame_id=self.tag_map_frame,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = yaml.safe_load(f)

    rospy.init_node("pose_graph_waypoint_follower")
    wp_transformer = PoseGraphWaypointFollower(params)
    rospy.spin()
