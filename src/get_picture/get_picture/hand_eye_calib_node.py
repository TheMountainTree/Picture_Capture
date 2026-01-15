#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

# === 配置参数 ===
ARUCO_DICT_TYPE = aruco.DICT_ARUCO_ORIGINAL
TARGET_ID = 26
MARKER_LENGTH = 0.12  # meters
ROBOT_BASE_FRAME = 'base_link'  # 机械臂基座坐标系
ROBOT_EE_FRAME = 'tool0'        # 机械臂末端坐标系 (根据你的ur5e urdf可能是 tool0 或 ee_link)
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), 'data')) # 假设在工作空间根目录运行

class HandEyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('hand_eye_calib_node')
        
        self.declare_parameter('camera_topic', '/camera1/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera1/color/camera_info')
        
        # ROS 通信
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.info_callback, 10)
            
        # Aruco 设置
        self.aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.aruco_params = aruco.DetectorParameters()
        
        # 内部状态
        self.camera_matrix = None
        self.dist_coeffs = None
        self.current_frame = None
        self.samples = [] # 存储 {'R_g2b':..., 't_g2b':..., 'R_t2c':..., 't_t2c':...}
        
        self.get_logger().info(f"Hand-Eye Calibration Node Started.")
        self.get_logger().info(f"Keys: [Space] Capture | [S] Solve | [Q] Quit")
        self.get_logger().info(f"Saving data to: {DATA_DIR}")

        # 确保数据目录存在
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera Intrinsics Received.")

    def get_robot_pose(self):
        """
        获取机械臂末端相对于基座的位姿 (T_gripper^base)
        对应 AX=XB 中的 A 部分的来源
        """
        try:
            # 查找最近的 TF 变换
            trans = self.tf_buffer.lookup_transform(
                ROBOT_BASE_FRAME, 
                ROBOT_EE_FRAME, 
                rclpy.time.Time())
            
            # 提取平移
            t = np.array([trans.transform.translation.x,
                          trans.transform.translation.y,
                          trans.transform.translation.z])
            
            # 提取旋转 (四元数 -> 矩阵)
            q = [trans.transform.rotation.x,
                 trans.transform.rotation.y,
                 trans.transform.rotation.z,
                 trans.transform.rotation.w]
            R_mat = Rotation.from_quat(q).as_matrix()
            
            return R_mat, t
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF Error: {e}")
            return None, None

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.current_frame = frame.copy()
        
        # 1. Aruco 检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        tvec_vis = None
        rvec_vis = None
        detected = False

        if ids is not None:
            # 筛选目标 ID
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == TARGET_ID:
                    # 获取 T_target^cam
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                        [corners[i]], MARKER_LENGTH, self.camera_matrix, self.dist_coeffs)
                    
                    rvec_vis = rvecs[0][0]
                    tvec_vis = tvecs[0][0]
                    detected = True
                    
                    # 绘制坐标轴
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, 
                                    rvec_vis, tvec_vis, 0.05)
                    break
        
        # 绘制 UI 信息
        cv2.putText(frame, f"Samples: {len(self.samples)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if detected:
            cv2.putText(frame, "Target Detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Target Missing", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Hand-Eye Calibration', frame)
        
        # === 键盘控制逻辑 ===
        key = cv2.waitKey(1) & 0xFF
        
        # [Space]: 采集数据
        if key == 32: 
            if detected:
                R_g2b, t_g2b = self.get_robot_pose()
                if R_g2b is not None:
                    # 处理 Aruco 位姿 (rvec -> Matrix)
                    R_t2c, _ = cv2.Rodrigues(rvec_vis)
                    t_t2c = tvec_vis
                    
                    self.samples.append({
                        'R_gripper2base': R_g2b,
                        't_gripper2base': t_g2b.reshape(3, 1),
                        'R_target2cam': R_t2c,
                        't_target2cam': t_t2c.reshape(3, 1)
                    })
                    self.get_logger().info(f"Sample {len(self.samples)} captured.")
                else:
                    self.get_logger().warn("Failed to get robot pose from TF.")
            else:
                self.get_logger().warn("Cannot capture: Aruco marker not visible.")

        # [S]: 开始标定
        elif key == ord('s'):
            if len(self.samples) < 3:
                self.get_logger().warn("Need at least 3 samples to calibrate.")
            else:
                self.solve_calibration()

        # [Q]: 退出
        elif key == ord('q'):
            self.get_logger().info("Quitting...")
            rclpy.shutdown()
            cv2.destroyAllWindows()

    def solve_calibration(self):
        self.get_logger().info("--- Solving Hand-Eye Calibration (Eye-to-Hand) ---")
        
        R_gripper2base = [s['R_gripper2base'] for s in self.samples]
        t_gripper2base = [s['t_gripper2base'] for s in self.samples]
        R_target2cam = [s['R_target2cam'] for s in self.samples]
        t_target2cam = [s['t_target2cam'] for s in self.samples]
        
        try:
            # OpenCV calibrateHandEye
            # 默认求解 AX=XB
            # 对应 Eye-to-Hand (相机固定): 
            # 输入: T_gripper^base 和 T_target^cam
            # 输出: T_cam^base (相机在基座坐标系下的位姿)
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # 构造齐次变换矩阵 4x4
            T_cam2base = np.eye(4)
            T_cam2base[:3, :3] = R_cam2base
            T_cam2base[:3, 3] = t_cam2base.flatten()
            
            print("\n" + "="*40)
            print("CALIBRATION RESULT (T_cam_to_base):")
            print(T_cam2base)
            print("="*40 + "\n")
            
            # 保存结果到 txt
            output_file = os.path.join(DATA_DIR, f'calibration_result_{int(time.time())}.txt')
            np.savetxt(output_file, T_cam2base, fmt='%.8f', 
                      header="Eye-to-Hand Calibration Result (T_cam_base)")
            self.get_logger().info(f"Result saved to {output_file}")
            
        except Exception as e:
            self.get_logger().error(f"Calibration Calculation Failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
