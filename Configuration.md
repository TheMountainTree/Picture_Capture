# Kinect2：

https://github.com/krepa098/kinect2\_ros2

## 安装依赖

### 1\. 安装 depth\_image\_proc

打开终端，运行以下命令来安装这个标准图像处理包：

```text
sudo apt update
sudo apt install ros-humble-depth-image-proc
```

### 2\. 检查是否需要安装其他依赖

```text
sudo apt install ros-humble-image-proc
```

## 设置udev规则

### 1\. 进入 libfreenect2 目录 (假设路径如下，如果不同请自行修改)

```text
cd ~/libfreenect2
```

### 2\. 将规则文件复制到系统配置目录

```text
sudo cp platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
```

### 3\. 重新加载 udev 规则

```text
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## 启动

### 1\. Build

### 2\. source

### 3\. 启动

Launch the kinect2\_bridge to receive color, depth, and mono images as well as the pointcloud.

```text
ros2 launch kinect2_bridge kinect2_bridge_launch.yaml
```

You can also launch rtabmap to create 3D scans:

```text
ros2 launch kinect2_bridge rtabmap.launch.py
```

### 4\. 相机参数

-   获取内参
    

```text
ros2 topic echo /kinect2/hd/camera_info --once
```

# orbbec: femto bolt

https://github.com/orbbec/OrbbecSDK\_ROS2

## 启动

### 1\. source

```text
source ~/ros2_ws/install/setup.bash
```

### 2\. 连接

```text
ros2 launch orbbec_camera femto_bolt.launch.py
```

### 3\. echo

-   List topics / services/ parameters ( On terminal 3)
    

```text
ros2 topic list
ros2 service list
ros2 param list
```

-   Echo a topic
    

```text
ros2 topic echo /camera/depth/camera_info
```

-   Call a service
    

```text
ros2 service call /camera/get_sdk_version orbbec_camera_msgs/srv/GetString '{}'
```

### 4\. 启用点云

启用普通深度点云（无颜色）

```text
ros2 launch orbbec_camera femto_bolt.launch.py enable_point_cloud:=true
```

启用彩色点云（RGB + Depth，看起来像真实世界）

```text
ros2 launch orbbec_camera femto_bolt.launch.py enable_colored_point_cloud:=true
```

### 5\. 获取相机自身变换阵

获取RGB和深度相机之间的变换

```text
ros2 run tf2_ros tf2_echo camera_depth_optical_frame camera_color_optical_frame
```
