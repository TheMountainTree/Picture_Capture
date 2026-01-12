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

class DualCameraSaveNode(Node):
    def __init__(self):
        super().__init__('dual_camera_save_node')
        self.bridge = CvBridge()

        # --- 状态标志位 ---
        self.is_saving = False
        self.last_save_time = 0
        self.save_status_text = ""

        # --- 数据暂存 (Orbbec) ---
        self.orb_color = None
        self.orb_depth = None
        self.orb_cloud = None

        # --- 数据暂存 (Kinect2) ---
        self.kin_color = None
        self.kin_depth = None
        self.kin_cloud = None

        # --- 目录配置 ---
        # 获取当前工作目录，并指向 data 文件夹
        self.base_dir = os.path.join(os.getcwd(), 'data')
        self.orb_dir = os.path.join(self.base_dir, 'orbbec')
        self.kin_dir = os.path.join(self.base_dir, 'kinect2')
        
        for d in [self.orb_dir, self.kin_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
                self.get_logger().info(f'Created directory: {d}')

        # --- 订阅 Orbbec (基于 get_picture_node.py) ---
        self.create_subscription(Image, 'camera/color/image_raw', self.orb_color_callback, 10)
        self.create_subscription(Image, 'camera/depth/image_raw', self.orb_depth_callback, 10)
        self.create_subscription(PointCloud2, '/camera/depth/points', self.orb_cloud_callback, qos_profile_sensor_data)

        # --- 订阅 Kinect2 (基于 get_kinect2_node.py) ---
        self.create_subscription(Image, '/kinect2/hd/image_color_rect', self.kin_color_callback, 10)
        self.create_subscription(Image, '/kinect2/hd/image_depth_rect', self.kin_depth_callback, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/kinect2/qhd/points', self.kin_cloud_callback, qos_profile_sensor_data)

        self.get_logger().info('Dual Camera Node started. Press "s" in any color window to save both.')

    # --- 回调函数：Orbbec ---
    def orb_color_callback(self, msg):
        self.orb_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.show_image("Orbbec Color", self.orb_color)

    def orb_depth_callback(self, msg):
        self.orb_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.show_depth("Orbbec Depth", self.orb_depth)

    def orb_cloud_callback(self, msg):
        self.orb_cloud = msg

    # --- 回调函数：Kinect2 ---
    def kin_color_callback(self, msg):
        self.kin_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.show_image("Kinect2 Color", self.kin_color)

    def kin_depth_callback(self, msg):
        self.kin_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.show_depth("Kinect2 Depth", self.kin_depth)

    def kin_cloud_callback(self, msg):
        self.kin_cloud = msg

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
            'orb': (self.orb_color, self.orb_depth, self.orb_cloud),
            'kin': (self.kin_color, self.kin_depth, self.kin_cloud)
        }

        # 2. 保存 Orbbec
        self.save_camera_data('orbbec', self.orb_dir, snaps['orb'], timestamp)
        
        # 3. 保存 Kinect2
        self.save_camera_data('kinect2', self.kin_dir, snaps['kin'], timestamp)

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
    node = DualCameraSaveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()