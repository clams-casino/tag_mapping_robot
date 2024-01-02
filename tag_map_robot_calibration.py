#!/usr/bin/python3

import argparse
import numpy as np

import rospy
import tf
from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    translation_matrix,
    translation_from_matrix,
)
import tf2_ros
import geometry_msgs.msg


""" START OF of definitions """
MAP_FRAME_NAME = "map"
TAG_MAP_FRAME_NAME = "tag_map"

# fmt: off
# stores transform from the april tag to the tag map frame
TAG_CALIBRATIONS = {
    "tag_0": np.array(
        [
            [ 0, 0, -1, 0],
            [-1, 0,  0, 0],
            [ 0, 1,  0, 0],
            [ 0, 0,  0, 1],
        ]
        
    ),
    "tag_1": np.array(
        [
            [-0.008594831, 0.029007451, 0.999542244, -18.480030055],
            [ 0.999899035, 0.011062191, 0.008918931,   2.195683623],
            [-0.011315843, 0.999517982, 0.028909445,   0.048892966],
            [ 0,           0,           0,             1          ],
        ]
    ),
    "tag_2": np.array(
        [
            [ 0.999898036, -0.007735447, -0.012003354, -6.507963383],
            [-0.011841968,  0.020581819, -0.999718039, -5.083118818],
            [ 0.007980317,  0.999758247,  0.020488118,  0.531296805],
            [ 0,            0,            0,            1          ],
        ]
    ),
    "tag_3": np.array(
        [
            [ 0.000453336, -0.001409876,  0.999998903, -42.286504164],
            [ 0.999983617,  0.005706876, -0.000445283,  -6.869720079],
            [-0.005706242,  0.999982722,  0.00141244,   -0.778408466],
            [ 0,            0,            0,             1          ],
        ]
    ),
    "tag_5": np.array(
        [
            [ 0.007563754, -0.013764611,  0.999876655, -28.237872945],
            [ 0.999900326,  0.012025085, -0.007398392,  -1.050738206],
            [-0.011921766,  0.999832952,  0.013854194,  -0.225190027],
            [ 0,            0,            0,             1          ],
        ]
    ),
}
# fmt: on

""" END OF of definitions """


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag_name", choices=TAG_CALIBRATIONS.keys(), required=True)
    args = parser.parse_args()

    at_frame_name = args.tag_name
    tag_map_to_at = np.linalg.inv(TAG_CALIBRATIONS[args.tag_name])

    rospy.init_node("tag_map_robot_calibration")
    tf_listener = tf.TransformListener()

    """ TF from tag map to map frame """
    at_to_map_trans, at_to_map_quat = lookupLatestTransform(
        tf_listener, MAP_FRAME_NAME, at_frame_name
    )

    at_to_map = quaternion_matrix(at_to_map_quat)
    at_to_map[:3, -1] = at_to_map_trans

    tag_map_to_map = at_to_map @ tag_map_to_at

    tag_map_to_map_trans = tag_map_to_map[:3, -1]
    tag_map_to_map_quat = quaternion_from_matrix(tag_map_to_map)

    # print("tag map to map:\n", tag_map_to_map)

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
