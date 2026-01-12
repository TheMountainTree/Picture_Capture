# Dual-Camera Extrinsic Calibration Project

This project provides a complete workflow for performing high-precision extrinsic calibration between an **Orbbec (Femto Bolt)** camera and a **Kinect V2** camera. 

The core objective is to compute the homogeneous transformation matrix ($T_{O \to K}$) that aligns the coordinate systems of the two cameras, enabling data fusion and multi-camera applications.

## Project Structure

```
ros2_ws/
├── src/
│   ├── get_picture/           # ROS 2 package for synchronized data capture
│   ├── kinect2_ros2/          # Drivers for Kinect V2
│   └── OrbbecSDK_ROS2/        # Drivers for Orbbec cameras
├── data/                      # Data storage directory
│   ├── orbbec/                # Captured Orbbec data
│   ├── kinect2/               # Captured Kinect data
│   ├── merge/                 # Fused point clouds
│   ├── camera_parameters.txt  # Intrinsic & configuration parameters
│   └── calibration_final.txt  # Final calibration result
├── calibration_pipeline.py    # Main calibration script (PnP + ICP)
├── transform.py               # Verification and fusion script
└── README.md                  # Project documentation
```

## 1. Configuration (`data/camera_parameters.txt`)

The project relies on accurate intrinsic parameters for both cameras. Key parameters are stored in **`data/camera_parameters.txt`**, including:
- **Intrinsics (K)**: Camera matrix for Orbbec and Kinect.
- **Distortion (D)**: Distortion coefficients.
- **Extrinsics (Depth-to-Color)**: Transformation from depth frame to color frame (specifically for Orbbec).
- **Calibration Target**: Dimensions of the chessboard (Rows: 8, Cols: 11, Square Size: 20mm).

## 2. Data Capture

We use the custom `get_picture` ROS 2 package to capture synchronized frames from both cameras.

**Workflow:**
1.  Launch camera drivers:
    ```bash
    # Orbbec
    ros2 launch orbbec_camera gemini330_series.launch.py
    # Kinect
    ros2 launch kinect2_bridge kinect2_bridge_launch.yaml
    ```
2.  Run the capture node:
    ```bash
    ros2 run get_picture DualCameraSaveNode.py
    ```
3.  Capture data:
    -   Place a chessboard in the overlapping field of view.
    -   Press **`s`** to save a synchronized snapshot (Color images + Point Clouds).
    -   Files are saved to `data/orbbec/` and `data/kinect2/`.

## 3. Calibration (`calibration_pipeline.py`)

The calibration process is automated by the `calibration_pipeline.py` script, which executes a two-stage algorithm:

**Stage 1: Symmetric PnP Calibration**
-   Detects chessboard corners in synchronized color images.
-   Reconstructs 3D points using ray-plane intersection.
-   Computes the relative pose for each frame using `cv2.solvePnP`.
-   Calculates a robust average transformation matrix ($T_{PnP}$).

**Stage 2: ICP Refinement**
-   Uses $T_{PnP}$ as the initial guess.
-   Loads the corresponding point clouds from both cameras.
-   Performs **Iterative Closest Point (ICP)** registration to fine-tune the alignment.
-   Outputs the final high-precision transformation matrix.

**Usage:**
```bash
python3 calibration_pipeline.py
```
**Output:**
-   `data/calibration_final.txt`: The final 4x4 homogenous transformation matrix.

## 4. Verification & Fusion (`transform.py`)

To verify the calibration accuracy, use `transform.py` to fuse the point clouds from both cameras using the computed calibration matrix.

**Features:**
-   Loads `data/calibration_final.txt`.
-   Transforms Orbbec point clouds into the Kinect coordinate system.
-   Merges them into a single point cloud.
-   **Visualization**: Orbbec points are colored **Red**, Kinect points are colored **Green**.

**Usage:**
```bash
python3 transform.py
```
**Output:**
-   Merged point clouds are saved in `data/merge/`.
