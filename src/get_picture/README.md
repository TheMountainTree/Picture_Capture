# get_picture

A ROS 2 package designed for capturing, saving, and visualizing data (Color, Depth, PointCloud) from Orbbec and Kinect2 cameras.

## Features

- **Single Camera Capture**: Nodes to capture data individually from Orbbec or Kinect2 cameras.
- **Dual Camera Capture**: Synchronized capture node to save data from both cameras simultaneously.
- **Visualization**: A dedicated Open3D-based tool to visualize point clouds from both cameras in real-time.
- **Data Saving**: Saves RGB images, Raw Depth images (16-bit), Visualized Depth maps, and Point Clouds (.ply format).

## Nodes

### 1. Orbbec Node (`get_picture_node.py`)
Connects to an Orbbec camera and allows saving data on demand.
- **Topics**: 
  - `camera/color/image_raw`
  - `camera/depth/image_raw`
  - `/camera/depth/points`
- **Output Directory**: `./saved_data/`

### 2. Kinect2 Node (`get_kinect2_node.py`)
Connects to a Kinect2 camera (using QHD resolution topics).
- **Topics**: 
  - `/kinect2/qhd/image_color_rect`
  - `/kinect2/qhd/image_depth_rect`
  - `/kinect2/qhd/points`
- **Output Directory**: `./kinect2_data/`

### 3. Dual Camera Save Node (`DualCameraSaveNode.py`)
Subscribes to both cameras and triggers a simultaneous save for both when requested.
- **Output Directory**: 
  - Orbbec data: `./data/orbbec/`
  - Kinect2 data: `./data/kinect2/`

### 4. Dual Camera Visualizer (`dual_cam_view.py`)
Uses Open3D to render live point clouds from both cameras in separate windows.
- **Topics**: 
  - `/camera/depth/points`
  - `/kinect2/qhd/points`

## Prerequisites

- **ROS 2** (Humble, Foxy, etc.)
- **Python 3 Libraries**:
  - `opencv-python`
  - `numpy`
  - `open3d` (for visualization)
  - `sensor_msgs_py` (for point cloud processing)
- **Camera Drivers**:
  - Ensure Orbbec and Kinect2 ROS 2 drivers are running and publishing to the topics listed above.

## Installation

1. Clone this repository into your ROS 2 workspace `src` directory.
2. Build the package:
   ```bash
   colcon build --packages-select get_picture
   ```
3. Source the workspace:
   ```bash
   source install/setup.bash
   ```

## Usage

### Running Capture Nodes
Launch the node you need using `ros2 run`.

**Orbbec:**
```bash
ros2 run get_picture get_picture_node.py
```

**Kinect2:**
```bash
ros2 run get_picture get_kinect2_node.py
```

**Dual Camera Mode:**
```bash
ros2 run get_picture DualCameraSaveNode.py
```

**Controls:**
- **`s`**: Press 's' on the OpenCV image window to save the current frame (Color, Depth, and Point Cloud).
- **UI Feedback**: "Saving..." text will appear on the image.

### Running Visualization
```bash
ros2 run get_picture dual_cam_view.py
```
- Opens two Open3D windows.
- Press **`Q`** in the visualization window to exit.

## Output Files

Saved files follow a timestamped naming convention `YYYYMMDD_HHMMSS`:

- `*_color_*.png`: RGB Image.
- `*_depth_*.png`: Raw 16-bit Depth Image (values in mm).
- `*_depth_vis_*.jpg`: False-color Depth Image for visualization.
- `*_cloud_*.ply`: Point Cloud data (ASCII PLY format).

## Troubleshooting

- **Missing `sensor_msgs_py`**: If point clouds are not saving, ensure `sensor_msgs_py` is installed.
- **Topics not found**: Check if your camera drivers are publishing to the correct topics. You may need to modify the topic names in the python scripts if they differ.
- **Directories**: The scripts create save directories in the current working directory where the node is launched.
