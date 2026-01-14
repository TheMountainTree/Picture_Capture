#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import os
import struct
import threading  # <--- 新增：引入多线程模块
from rclpy.qos import qos_profile_sensor_data

# 尝试导入 sensor_msgs_py 用于处理点云
try:
    from sensor_msgs_py import point_cloud2
except ImportError:
    point_cloud2 = None

class OrbbecNode(Node):
    def __init__(self):
        super().__init__('orbbec_node')

        self.bridge = CvBridge()

        # 数据暂存
        self.latest_color_img = None
        self.latest_depth_msg = None # 注意：这里变量名保持一致
        self.latest_cloud_msg = None

        # 状态标志位
        self.is_saving = False      # 是否正在保存中
        self.last_save_time = 0     # 用于控制UI显示提示的持续时间
        self.save_status_text = ""  # UI上显示的提示文字

        # 创建保存目录
        self.save_dir = os.path.join(os.getcwd(), 'saved_data')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self.get_logger().info(f'Created save directory: {self.save_dir}')

        # 订阅话题
        self.create_subscription(Image, 'camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(Image, 'camera/depth/image_raw', self.depth_callback, 10)
        self.create_subscription(PointCloud2, '/camera/depth/points', self.point_cloud_callback, qos_profile_sensor_data)

        self.get_logger().info('Node started. Press "s" to save (Non-blocking).')

    def color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_color_img = cv_image
            
            # --- UI 反馈逻辑 ---
            # 如果正在保存，或者距离上次保存结束不到2秒，在画面上显示文字
            if self.is_saving:
                cv2.putText(cv_image, "Saving... Please wait", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif (time.time() - self.last_save_time) < 2.0 and self.save_status_text:
                # 保存完成后显示 2秒 的 "Saved!"
                cv2.putText(cv_image, self.save_status_text, (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Color Image", cv_image)
            
            # --- 按键检测 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if not self.is_saving:
                    # 启动一个新线程去执行保存，不阻塞当前主线程
                    threading.Thread(target=self.save_thread_func).start()
                else:
                    self.get_logger().warn("Already saving! Please wait.")

        except Exception as e:
            self.get_logger().error(f"Failed to process color image: {e}")

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_msg = depth_image
            
            # 显示用的深度图
            depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = np.uint8(depth_display)
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            
            cv2.imshow("Depth Image", depth_colormap)
            cv2.waitKey(1)
        except Exception as e:
            pass

    def point_cloud_callback(self, msg):
        self.latest_cloud_msg = msg

    # --- 新增：线程执行函数 ---
    def save_thread_func(self):
        """这个函数会在后台线程运行，不会卡住画面"""
        self.is_saving = True
        self.get_logger().info("Start saving in background...")
        
        # 锁定当前时刻的数据快照（防止保存过程中数据被新的一帧覆盖）
        color_snap = self.latest_color_img
        depth_snap = self.latest_depth_msg
        cloud_snap = self.latest_cloud_msg
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. 保存彩色图
            if color_snap is not None:
                cv2.imwrite(os.path.join(self.save_dir, f'color_{timestamp}.png'), color_snap)

            # 2. 保存深度图
            if depth_snap is not None:
                cv2.imwrite(os.path.join(self.save_dir, f'depth_{timestamp}.png'), depth_snap)
                # 可选：保存可视化深度图
                depth_vis = cv2.normalize(depth_snap, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite(os.path.join(self.save_dir, f'depth_vis_{timestamp}.jpg'), np.uint8(depth_vis))
                # depth_norm = cv2.normalize(depth_snap, None, 0, 255, cv2.NORM_MINMAX)
                # depth_norm = np.uint8(depth_norm)
                # # 加上这一句，变成伪彩色
                # depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET) 
                # cv2.imwrite(os.path.join(self.save_dir, f'depth_vis_{timestamp}.jpg'), depth_color)

            # 3. 保存点云 (最耗时的部分)
            if cloud_snap is not None:
                self.save_point_cloud_to_ply(cloud_snap, os.path.join(self.save_dir, f'cloud_{timestamp}.ply'))
            
            # 更新状态
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
            return
        
        # 使用生成器读取点
        points = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # 优化：为了避免把所有点一次性读入内存列表（如果点很多会慢），我们可以直接写文件
        # 但为了获取点的数量写header，这里还是得先读出来，或者分两步走
        # 简单优化：只做必要的转换
        points_list = list(points) # 这一步依然稍慢，但在后台线程无所谓
        
        if not points_list:
            return

        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_list)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points_list:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n") # 格式化减少一点字符串长度

def main(args=None):
    rclpy.init(args=args)
    node = OrbbecNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()