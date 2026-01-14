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

class DualCameraOrbbecNode(Node):
    def __init__(self):
        super().__init__('dual_camera_orbbec_node')
        self.bridge = CvBridge()

        # --- 状态标志位 ---
        self.is_saving = False
        self.last_save_time = 0
        self.save_status_text = ""

        # --- 数据暂存 (Camera1) ---
        self.cam1_color = None
        self.cam1_depth = None
        self.cam1_cloud = None

        # --- 数据暂存 (Camera2) ---
        self.cam2_color = None
        self.cam2_depth = None
        self.cam2_cloud = None

        # --- 目录配置 ---
        # 获取当前工作目录，并指向 data 文件夹
        self.base_dir = os.path.join(os.getcwd(), 'data')
        self.cam1_dir = os.path.join(self.base_dir, 'camera1')
        self.cam2_dir = os.path.join(self.base_dir, 'camera2')
        
        for d in [self.cam1_dir, self.cam2_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
                self.get_logger().info(f'Created directory: {d}')

        # --- 订阅 Camera1 ---
        self.create_subscription(Image, 'camera1/color/image_raw', self.cam1_color_callback, 10)
        self.create_subscription(Image, 'camera1/depth/image_raw', self.cam1_depth_callback, 10)
        self.create_subscription(PointCloud2, 'camera1/depth/points', self.cam1_cloud_callback, qos_profile_sensor_data)

        # --- 订阅 Camera2 ---
        self.create_subscription(Image, 'camera2/color/image_raw', self.cam2_color_callback, 10)
        self.create_subscription(Image, 'camera2/depth/image_raw', self.cam2_depth_callback, 10)
        self.create_subscription(PointCloud2, 'camera2/depth/points', self.cam2_cloud_callback, qos_profile_sensor_data)

        self.get_logger().info('Dual Camera Orbbec Node started. Press "s" in any color window to save both.')

    # --- 回调函数：Camera1 ---
    def cam1_color_callback(self, msg):
        self.cam1_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.show_image("Camera1 Color", self.cam1_color)

    def cam1_depth_callback(self, msg):
        self.cam1_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.show_depth("Camera1 Depth", self.cam1_depth)

    def cam1_cloud_callback(self, msg):
        self.cam1_cloud = msg

    # --- 回调函数：Camera2 ---
    def cam2_color_callback(self, msg):
        self.cam2_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.show_image("Camera2 Color", self.cam2_color)

    def cam2_depth_callback(self, msg):
        self.cam2_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.show_depth("Camera2 Depth", self.cam2_depth)

    def cam2_cloud_callback(self, msg):
        self.cam2_cloud = msg

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
            'cam1': (self.cam1_color, self.cam1_depth, self.cam1_cloud),
            'cam2': (self.cam2_color, self.cam2_depth, self.cam2_cloud)
        }

        # 2. 保存 Camera1
        self.save_camera_data('camera1', self.cam1_dir, snaps['cam1'], timestamp)
        
        # 3. 保存 Camera2
        self.save_camera_data('camera2', self.cam2_dir, snaps['cam2'], timestamp)

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
    node = DualCameraOrbbecNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
