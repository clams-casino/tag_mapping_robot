#!/usr/bin/python3

import numpy as np
import rospy
import tf
from tf.transformations import quaternion_matrix, quaternion_from_matrix, translation_matrix, translation_from_matrix
import tf2_ros
import geometry_msgs.msg


MAP_FRAME_NAME = "map"
TAG_FRAME_NAME = "tag"
TAG_MAP_FRAME_NAME = "tag_map"


def lookupLatestTransform(listener, target_frame, source_frame):
    try:
        # Wait a bit since after starting this node it could be a bit until the frames will be seen to exist
        listener.waitForTransform(
            target_frame, source_frame, rospy.Time(), rospy.Duration(4.0)
        )

        trans, rot = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        raise Exception(
            "Could not get the latest TF from {} to {}".format(
                source_frame, target_frame
            )
        )
    return trans, rot


def generateTransformStampedMsg(trans, quat, target_frame, source_frame):
    # Set the header timestamp right before broadcasting
    tf_stamped = geometry_msgs.msg.TransformStamped()
    tf_stamped.header.frame_id = target_frame
    tf_stamped.child_frame_id = source_frame

    tf_stamped.transform.translation.x = trans[0]
    tf_stamped.transform.translation.y = trans[1]
    tf_stamped.transform.translation.z = trans[2]

    tf_stamped.transform.rotation.x = quat[0]
    tf_stamped.transform.rotation.y = quat[1]
    tf_stamped.transform.rotation.z = quat[2]
    tf_stamped.transform.rotation.w = quat[3]

    return tf_stamped


if __name__ == "__main__":
    rospy.init_node("tag_map_robot_calibration")
    tf_listener = tf.TransformListener()

    """ TF from tag map to map frame """
    tag_to_map_trans, tag_to_map_quat = lookupLatestTransform(
        tf_listener, MAP_FRAME_NAME, TAG_FRAME_NAME
    )

    tag_map_to_map_trans = tag_to_map_trans

    tag_map_to_tag_R = np.array([
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
    ])

    tag_to_map_R = quaternion_matrix(tag_to_map_quat)

    tag_map_to_map_quat = quaternion_from_matrix(
        tag_to_map_R @ tag_map_to_tag_R
    )
    

    # print("tag_map_to_map_trans: ", tag_map_to_map_trans)
    # print("tag_map_to_map_quat: ", tag_map_to_map_quat)

    """ Publish static transform """
    print("\n\nPublishing static TFs")
    static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

    tag_map_to_map_transform_stamped = generateTransformStampedMsg(
        tag_map_to_map_trans, tag_map_to_map_quat, MAP_FRAME_NAME, TAG_MAP_FRAME_NAME
    )

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        tag_map_to_map_transform_stamped.header.stamp = rospy.Time.now()
        static_tf_broadcaster.sendTransform(tag_map_to_map_transform_stamped)
        rate.sleep()
