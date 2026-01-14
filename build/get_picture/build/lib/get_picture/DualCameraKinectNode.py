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

class DualCameraKinectNode(Node):
    def __init__(self):
        super().__init__('dual_camera_kinect_node')
        self.bridge = CvBridge()

        # --- 状态标志位 ---
        self.is_saving = False
        self.last_save_time = 0
        self.save_status_text = ""

        # --- 数据暂存 (Main Kinect) ---
        self.main_color = None
        self.main_depth = None
        self.main_cloud = None

        # --- 数据暂存 (Assistant Kinect) ---
        self.assi_color = None
        self.assi_depth = None
        self.assi_cloud = None

        # --- 目录配置 ---
        # 获取当前工作目录，并指向 data 文件夹
        self.base_dir = os.path.join(os.getcwd(), 'data')
        self.main_dir = os.path.join(self.base_dir, 'kinect2_main')
        self.assi_dir = os.path.join(self.base_dir, 'kinect2_assi')
        
        for d in [self.main_dir, self.assi_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
                self.get_logger().info(f'Created directory: {d}')

        # --- 订阅 Kinect2_main ---
        self.create_subscription(Image, '/kinect2_main/hd/image_color_rect', self.main_color_callback, 10)
        self.create_subscription(Image, '/kinect2_main/hd/image_depth_rect', self.main_depth_callback, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/kinect2_main/qhd/points', self.main_cloud_callback, qos_profile_sensor_data)

        # --- 订阅 Kinect2_assi ---
        self.create_subscription(Image, '/kinect2_assi/hd/image_color_rect', self.assi_color_callback, 10)
        self.create_subscription(Image, '/kinect2_assi/hd/image_depth_rect', self.assi_depth_callback, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/kinect2_assi/qhd/points', self.assi_cloud_callback, qos_profile_sensor_data)

        self.get_logger().info('Dual Kinect Node started. Press "s" in any color window to save both.')

    # --- 回调函数：Main Kinect ---
    def main_color_callback(self, msg):
        self.main_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.show_image("Kinect2 Main Color", self.main_color)

    def main_depth_callback(self, msg):
        self.main_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.show_depth("Kinect2 Main Depth", self.main_depth)

    def main_cloud_callback(self, msg):
        self.main_cloud = msg

    # --- 回调函数：Assistant Kinect ---
    def assi_color_callback(self, msg):
        self.assi_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.show_image("Kinect2 Assi Color", self.assi_color)

    def assi_depth_callback(self, msg):
        self.assi_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.show_depth("Kinect2 Assi Depth", self.assi_depth)

    def assi_cloud_callback(self, msg):
        self.assi_cloud = msg

    # --- 可视化辅助 ---
    def show_image(self, win_name, img):
        if img is None: return
        disp = img.copy()
        if self.is_saving:
            cv2.putText(disp, "SAVING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        elif (time.time() - self.last_save_time) < 2.0:
            cv2.putText(disp, "SAVED!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow(win_name, disp)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            self.trigger_save()

    def show_depth(self, win_name, depth_img):
        if depth_img is None: return
        depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(np.uint8(depth_vis), cv2.COLORMAP_JET)
        cv2.imshow(win_name, depth_vis)
        cv2.waitKey(1)

    # --- 保存逻辑 ---
    def trigger_save(self):
        if self.is_saving:
            self.get_logger().warn("Save in progress, please wait...")
            return
        # 启动后台线程
        threading.Thread(target=self.save_all_data_thread).start()

    def save_all_data_thread(self):
        self.is_saving = True
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.get_logger().info(f"Sync Save Started: {timestamp}")

        # 1. 抓取快照
        snaps = {
            'main': (self.main_color, self.main_depth, self.main_cloud),
            'assi': (self.assi_color, self.assi_depth, self.assi_cloud)
        }

        # 2. 保存 Main Kinect
        self.save_camera_data('kinect2_main', self.main_dir, snaps['main'], timestamp)
        
        # 3. 保存 Assistant Kinect
        self.save_camera_data('kinect2_assi', self.assi_dir, snaps['assi'], timestamp)

        self.get_logger().info(f"Sync Save Completed: {timestamp}")
        self.last_save_time = time.time()
        self.is_saving = False

    def save_camera_data(self, prefix, path, data, ts):
        color, depth, cloud = data
        
        # 保存彩色图
        if color is not None:
            cv2.imwrite(os.path.join(path, f"{prefix}_color_{ts}.png"), color)
        
        # 保存深度图 (Raw & Vis)
        if depth is not None:
            cv2.imwrite(os.path.join(path, f"{prefix}_depth_{ts}.png"), depth)
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(os.path.join(path, f"{prefix}_depth_vis_{ts}.jpg"), np.uint8(depth_vis))

        # 保存点云
        if cloud is not None:
            filename = os.path.join(path, f"{prefix}_cloud_{ts}.ply")
            self.save_ply(cloud, filename)

    def save_ply(self, cloud_msg, filename):
        if point_cloud2 is None: return
        points = list(point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
        if not points: return

        with open(filename, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for p in points:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

def main(args=None):
    rclpy.init(args=args)
    node = DualCameraKinectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
