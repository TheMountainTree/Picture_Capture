#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge

import cv2
import cv2.aruco as aruco
import numpy as np

from scipy.spatial.transform import Rotation

class ArucoPose(Node):
    def __init__(self):
        super().__init__('aruco_pose_node')

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            '/camera1/color/image_raw',
            self.image_callback,
            10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera1/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.pose_pub = self.create_publisher(
            PoseArray,
            '/aruco_poses',
            10
        )

        self.camera_matrix = None
        self.dist_coeffs = None

        self.marker_length = 0.12  # meters
        self.target_id = 26

        self.aruco_dict = aruco.getPredefinedDictionary(
            aruco.DICT_ARUCO_ORIGINAL
        )
        if hasattr(aruco, "DetectorParameters"):
            self.aruco_params = aruco.DetectorParameters()
        else:
            self.aruco_params = aruco.DetectorParameters_create()


        self.get_logger().info('Aruco Pose Node started')

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is None:
            return

        pose_array = PoseArray()
        pose_array.header = msg.header
        pose_array.header.frame_id = "camera_color_optical_frame"

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id != self.target_id:
                continue

            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                [corners[i]],
                self.marker_length,
                self.camera_matrix,
                self.dist_coeffs
            )

            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            print(tvec)

            cv2.drawFrameAxes(
                frame,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                0.03
            )

            pose = Pose()
            pose.position.x = float(tvec[0])
            pose.position.y = float(tvec[1])
            pose.position.z = float(tvec[2])

            # R, _ = cv2.Rodrigues(rvec)
            # qw = np.sqrt(1 + np.trace(R)) / 2.0
            # qx = (R[2, 1] - R[1, 2]) / (4 * qw)
            # qy = (R[0, 2] - R[2, 0]) / (4 * qw)
            # qz = (R[1, 0] - R[0, 1]) / (4 * qw)

            # 替换原四元数计算部分
            rot = Rotation.from_rotvec(rvec.flatten())
            qx, qy, qz, qw = rot.as_quat()  # returns [x, y, z, w]

            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw

            pose_array.poses.append(pose)

        if len(pose_array.poses) > 0:
            self.pose_pub.publish(pose_array)

        cv2.imshow('Aruco Pose', frame)
        cv2.waitKey(1)



def main():
    rclpy.init()
    node = ArucoPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
