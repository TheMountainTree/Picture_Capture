#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from rclpy.qos import qos_profile_sensor_data

import open3d as o3d
import numpy as np
import threading
import time

class DualCamVisualizer(Node):
    def __init__(self):
        super().__init__('dual_cam_visualizer')
        
        # --- 配置话题 ---
        # 请确保这些话题名与你实际运行的一致
        self.topic_orbbec = '/camera/depth/points'
        self.topic_kinect = '/kinect2/qhd/points' 

        # --- 数据存储与锁 ---
        self.lock = threading.Lock()
        
        # Orbbec 数据
        self.pcd_orbbec_msg = None
        self.new_orbbec = False
        
        # Kinect 数据
        self.pcd_kinect_msg = None
        self.new_kinect = False

        # --- 订阅 ---
        self.create_subscription(PointCloud2, self.topic_orbbec, self.callback_orbbec, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.topic_kinect, self.callback_kinect, qos_profile_sensor_data)
        
        self.get_logger().info(f'Listening to: {self.topic_orbbec} & {self.topic_kinect}')

    def callback_orbbec(self, msg):
        with self.lock:
            self.pcd_orbbec_msg = msg
            self.new_orbbec = True

    def callback_kinect(self, msg):
        with self.lock:
            self.pcd_kinect_msg = msg
            self.new_kinect = True

    def convert_ros_to_o3d(self, ros_cloud, remove_nans=True):
        """极速转换 ROS PointCloud2 -> Open3D PointCloud"""
        if ros_cloud is None:
            return None

        # 1. 简单检查字段
        field_names = [field.name for field in ros_cloud.fields]
        if 'x' not in field_names:
            return None

        # 2. 读取二进制数据
        cloud_data = np.frombuffer(ros_cloud.data, dtype=np.uint8)
        
        # 3. 解析 XYZ
        point_step = ros_cloud.point_step
        num_points = ros_cloud.width * ros_cloud.height
        
        if len(cloud_data) != num_points * point_step:
            return None

        float_data = cloud_data.view(np.float32)
        stride = point_step // 4
        
        xs = float_data[0::stride]
        ys = float_data[1::stride]
        zs = float_data[2::stride]
        
        points = np.column_stack((xs, ys, zs))
        
        if remove_nans:
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]
        
        if len(points) == 0:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

def main():
    rclpy.init()
    node = DualCamVisualizer()

    # 1. 后台运行 ROS 通信
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # 2. 初始化两个独立的可视化窗口
    # 窗口1: Orbbec
    vis_orbbec = o3d.visualization.Visualizer()
    vis_orbbec.create_window(window_name="Orbbec Stream", width=800, height=600, left=0, top=50)
    
    # 窗口2: Kinect (位置向右偏移，避免重叠)
    vis_kinect = o3d.visualization.Visualizer()
    vis_kinect.create_window(window_name="Kinect Stream", width=800, height=600, left=820, top=50)
    
    # 初始化点云对象
    pcd_orbbec = o3d.geometry.PointCloud()
    pcd_kinect = o3d.geometry.PointCloud()
    
    added_orbbec = False
    added_kinect = False

    # 设置渲染选项 (背景色等)
    opt_o = vis_orbbec.get_render_option()
    opt_o.background_color = np.asarray([0.1, 0.1, 0.1])
    opt_o.point_size = 2.0
    
    opt_k = vis_kinect.get_render_option()
    opt_k.background_color = np.asarray([0.1, 0.1, 0.1])
    opt_k.point_size = 2.0

    print("双窗口模式已启动。")
    print("窗口 1: Orbbec 相机")
    print("窗口 2: Kinect 相机")
    print("按 'Q' 退出任意窗口即可结束程序。")

    try:
        # 主循环同时轮询两个窗口
        while vis_orbbec.poll_events() and vis_kinect.poll_events():
            
            # --- 窗口 1 更新逻辑 (Orbbec) ---
            msg_o = None
            with node.lock:
                if node.new_orbbec:
                    msg_o = node.pcd_orbbec_msg
                    node.new_orbbec = False
            
            if msg_o:
                temp_pcd = node.convert_ros_to_o3d(msg_o)
                if temp_pcd:
                    pcd_orbbec.points = temp_pcd.points
                    # Orbbec 保持原色或设为特定颜色 (这里设为红色以便区分，也可去掉这行显示原色)
                    pcd_orbbec.paint_uniform_color([1, 0.4, 0.4]) 
                    
                    if not added_orbbec:
                        vis_orbbec.add_geometry(pcd_orbbec)
                        vis_orbbec.reset_view_point(True)
                        added_orbbec = True
                    else:
                        vis_orbbec.update_geometry(pcd_orbbec)
            
            vis_orbbec.update_renderer()

            # --- 窗口 2 更新逻辑 (Kinect) ---
            msg_k = None
            with node.lock:
                if node.new_kinect:
                    msg_k = node.pcd_kinect_msg
                    node.new_kinect = False
            
            if msg_k:
                temp_pcd = node.convert_ros_to_o3d(msg_k)
                if temp_pcd:
                    pcd_kinect.points = temp_pcd.points
                    # Kinect 设为蓝色
                    pcd_kinect.paint_uniform_color([0.4, 0.4, 1])
                    
                    if not added_kinect:
                        vis_kinect.add_geometry(pcd_kinect)
                        vis_kinect.reset_view_point(True)
                        added_kinect = True
                    else:
                        vis_kinect.update_geometry(pcd_kinect)

            vis_kinect.update_renderer()
            
            # 控制帧率
            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        # 销毁窗口
        vis_orbbec.destroy_window()
        vis_kinect.destroy_window()
        rclpy.shutdown()

if __name__ == '__main__':
    main()