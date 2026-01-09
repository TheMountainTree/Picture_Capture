import numpy as np
import cv2
import open3d as o3d
import glob
import os

# ================= Configuration =================
DATA_DIR = "/home/themountaintree/workspace/ros2_ws/data"
DIR_ORBBEC = os.path.join(DATA_DIR, "orbbec")
DIR_KINECT = os.path.join(DATA_DIR, "kinect2")

# Intrinsics from calibration.py
# K1 for Orbbec
K1 = np.array([[745.9627, 0.0, 638.2298], 
               [0.0, 745.2563, 360.5474], 
               [0.0, 0.0, 1.0]])

# K2 for Kinect2
K2 = np.array([[540.6860, 0.0, 479.75], 
               [0.0, 540.6860, 269.75], 
               [0.0, 0.0, 1.0]])

CHESSBOARD_SIZE = (11, 8)
# =============================================

def segment_roi(ply_path, img_path, K, output_path):
    print(f"Processing: {os.path.basename(ply_path)}")
    
    # 1. Load Image
    color_image = cv2.imread(img_path)
    if color_image is None:
        print(f"  Error: Failed to load image {img_path}")
        return False

    # 2. Detect Chessboard
    # Try default flags first (matches calibration.py)
    ret, corners = cv2.findChessboardCorners(color_image, CHESSBOARD_SIZE, None)
    
    if not ret:
        # Try adaptive threshold
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(color_image, CHESSBOARD_SIZE, flags)
    
    if not ret:
        # Try swapped dimensions with default
        ret, corners = cv2.findChessboardCorners(color_image, (CHESSBOARD_SIZE[1], CHESSBOARD_SIZE[0]), None)
        
    if not ret:
        # Try swapped dimensions with adaptive
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(color_image, (CHESSBOARD_SIZE[1], CHESSBOARD_SIZE[0]), flags)
        
    if not ret:
        print(f"  No chessboard detected in {os.path.basename(img_path)} (Size: {color_image.shape})")
        return False
    
    # Optimize corners
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    # 3. Create ROI Mask
    hull = cv2.convexHull(corners)
    mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [hull.astype(int)], -1, (255), -1)

    # 4. Load Point Cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        print(f"  Error: Point cloud is empty {ply_path}")
        return False

    # 5. Project 3D to 2D
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    valid_z = Z > 0
    
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    
    u[valid_z] = (X[valid_z] * fx / Z[valid_z]) + cx
    v[valid_z] = (Y[valid_z] * fy / Z[valid_z]) + cy
    
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)
    
    h, w = mask.shape
    in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    
    valid_points_mask = valid_z & in_image
    
    # Filter points in ROI
    valid_u = u[valid_points_mask]
    valid_v = v[valid_points_mask]
    
    in_roi_mask = mask[valid_v, valid_u] > 0
    
    final_indices = np.where(valid_points_mask)[0][in_roi_mask]
    
    print(f"  Points kept: {len(final_indices)} / {len(points)}")

    # 6. Save ROI Cloud
    if len(final_indices) == 0:
        print("  Warning: No points in ROI.")
        return False
        
    pcd_roi = pcd.select_by_index(final_indices)
    o3d.io.write_point_cloud(output_path, pcd_roi)
    print(f"  Saved to {os.path.basename(output_path)}")
    return True

def process_folder(folder_path, K, prefix):
    # Find all PLY files
    # Pattern: {prefix}_cloud_{timestamp}.ply
    ply_files = sorted(glob.glob(os.path.join(folder_path, f"{prefix}_cloud_*.ply")))
    
    print(f"Found {len(ply_files)} ply files in {folder_path}")
    
    for ply_path in ply_files:
        # Avoid processing already generated ROI files if running multiple times
        if "_roi.ply" in ply_path:
            continue
            
        filename = os.path.basename(ply_path)
        # Extract timestamp: prefix_cloud_YYYYMMDD_HHMMSS.ply
        # timestamp is everything after prefix_cloud_ and before .ply
        timestamp_part = filename.replace(f"{prefix}_cloud_", "").replace(".ply", "")
        
        # Construct expected image path
        img_filename = f"{prefix}_color_{timestamp_part}.png"
        img_path = os.path.join(folder_path, img_filename)
        
        if not os.path.exists(img_path):
            print(f"  Missing image for {filename}: {img_path}")
            continue
            
        # Construct output path
        # User requested consistency. Let's use suffix _roi.ply
        output_filename = filename.replace(".ply", "_roi.ply")
        output_path = os.path.join(folder_path, output_filename)
        
        segment_roi(ply_path, img_path, K, output_path)

def main():
    if os.path.exists(DIR_ORBBEC):
        print("Processing Orbbec Data...")
        process_folder(DIR_ORBBEC, K1, "orbbec")
    else:
        print(f"Directory not found: {DIR_ORBBEC}")

    if os.path.exists(DIR_KINECT):
        print("\nProcessing Kinect2 Data...")
        process_folder(DIR_KINECT, K2, "kinect2")
    else:
        print(f"Directory not found: {DIR_KINECT}")

if __name__ == "__main__":
    main()
