#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import os
import threading
from rclpy.qos import qos_profile_sensor_data

# 尝试导入 sensor_msgs_py 用于处理点云
try:
    from sensor_msgs_py import point_cloud2
except ImportError:
    point_cloud2 = None

class Kinect2Node(Node):
    def __init__(self):
        super().__init__('kinect2_save_node')

        self.bridge = CvBridge()

        # 数据暂存
        self.latest_color_img = None
        self.latest_depth_msg = None
        self.latest_cloud_msg = None

        # 状态标志位
        self.is_saving = False
        self.last_save_time = 0
        self.save_status_text = ""

        # 创建保存目录
        self.save_dir = os.path.join(os.getcwd(), 'kinect2_data')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self.get_logger().info(f'Created save directory: {self.save_dir}')

        # --- 订阅配置 (针对 Kinect2) ---
        # 说明：Kinect2 通常提供 sd (512x424), qhd (960x540), hd (1920x1080) 三种分辨率
        # 这里使用 'sd' 以匹配深度相机的原生分辨率，保证点云和深度图一一对应且带宽占用最小。
        # 如果需要更高清的彩色图，可以将 'sd' 改为 'qhd' 或 'hd' (前提是驱动开启了对应分辨率)。
        
        # 1. 彩色图 (使用 rectified 校正后的图)
        self.create_subscription(
            Image, 
            '/kinect2/qhd/image_color_rect', 
            self.color_callback, 
            10
        )
        
        # 2. 深度图 (使用 rectified 校正后的图)
        self.create_subscription(
            Image, 
            '/kinect2/qhd/image_depth_rect', 
            self.depth_callback, 
            qos_profile_sensor_data # 深度图有时数据量大，使用 sensor_data 策略更保险
        )
        
        # 3. 点云 (必须使用 sensor_data QoS，否则可能收不到数据)
        self.create_subscription(
            PointCloud2, 
            '/kinect2/qhd/points', 
            self.point_cloud_callback, 
            qos_profile_sensor_data
        )

        self.get_logger().info('Kinect2 Node started. Waiting for data...')
        self.get_logger().info('Press "s" in the "Kinect2 Color" window to save.')

    def color_callback(self, msg):
        try:
            # Kinect2 的 image_color_rect 通常是 BGR8 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_color_img = cv_image
            
            # --- UI 反馈逻辑 ---
            if self.is_saving:
                cv2.putText(cv_image, "Saving... Please wait", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif (time.time() - self.last_save_time) < 2.0 and self.save_status_text:
                cv2.putText(cv_image, self.save_status_text, (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Kinect2 Color", cv_image)
            
            # --- 按键检测 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if not self.is_saving:
                    threading.Thread(target=self.save_thread_func).start()
                else:
                    self.get_logger().warn("Already saving! Please wait.")

        except Exception as e:
            self.get_logger().error(f"Failed to process color image: {e}")

    def depth_callback(self, msg):
        try:
            # Kinect2 深度通常是 16UC1 (mm)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_msg = depth_image
            
            # 可视化处理
            depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = np.uint8(depth_display)
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            
            cv2.imshow("Kinect2 Depth", depth_colormap)
            cv2.waitKey(1)
        except Exception as e:
            pass

    def point_cloud_callback(self, msg):
        self.latest_cloud_msg = msg

    def save_thread_func(self):
        self.is_saving = True
        self.get_logger().info("Start saving Kinect2 data...")
        
        # 锁定快照
        color_snap = self.latest_color_img
        depth_snap = self.latest_depth_msg
        cloud_snap = self.latest_cloud_msg
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. 保存彩色图
            if color_snap is not None:
                cv2.imwrite(os.path.join(self.save_dir, f'kinect2_color_{timestamp}.png'), color_snap)

            # 2. 保存深度图
            if depth_snap is not None:
                # 保存原始 16-bit 数据
                cv2.imwrite(os.path.join(self.save_dir, f'kinect2_depth_raw_{timestamp}.png'), depth_snap)
                # 保存可视化图
                depth_vis = cv2.normalize(depth_snap, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = np.uint8(depth_vis)
                depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(self.save_dir, f'kinect2_depth_vis_{timestamp}.jpg'), depth_vis_color)

            # 3. 保存点云
            if cloud_snap is not None:
                self.save_point_cloud_to_ply(cloud_snap, os.path.join(self.save_dir, f'kinect2_cloud_{timestamp}.ply'))
            else:
                self.get_logger().warn("No point cloud captured! Check topic /kinect2/sd/points")
            
            self.save_status_text = f"Saved: {timestamp}"
            self.get_logger().info(f"All data saved: {timestamp}")

        except Exception as e:
            self.get_logger().error(f"Error during save: {e}")
            self.save_status_text = "Save Failed!"
        finally:
            self.last_save_time = time.time()
            self.is_saving = False

    def save_point_cloud_to_ply(self, cloud_msg, filename):
        if point_cloud2 is None:
            self.get_logger().error("sensor_msgs_py not installed")
            return
        
        # 注意：Kinect2 点云通常包含 RGB 信息，这里我们只提取 XYZ
        # 如果你需要保存带颜色的点云，需要处理 'rgb' 字段并解包 float -> int
        points = list(point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
        
        if not points:
            return

        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

def main(args=None):
    rclpy.init(args=args)
    node = Kinect2Node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()