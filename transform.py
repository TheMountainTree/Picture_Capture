import numpy as np
import open3d as o3d
import glob
import os
import copy

# ===============================
# Configuration
# ===============================
POSSIBLE_BASE_DIRS = [
    "/home/themountaintree/workspace/Picture_Capture/data",
    os.path.join(os.getcwd(), "data"),
    os.path.join(os.path.dirname(os.getcwd()), "Picture_Capture/data"), 
    "/data"
]

BASE_DIR = "/home/themountaintree/workspace/Picture_Capture/data"
for p in POSSIBLE_BASE_DIRS:
    if os.path.exists(os.path.join(p, "orbbec")):
        BASE_DIR = p
        break

DIR_ORBBEC = os.path.join(BASE_DIR, "orbbec")
DIR_KINECT = os.path.join(BASE_DIR, "kinect2")
DIR_MERGE = os.path.join(BASE_DIR, "merge")

# depth -> color (Femto Bolt) - copied from calibration_pnp_icp_refine.py
T_depth2color = np.array([
    [ 1.000, -0.004, -0.004,  0.032],
    [ 0.004,  0.994, -0.105,  0.001],
    [ 0.004,  0.105,  0.994, -0.002],
    [ 0.000,  0.000,  0.000,  1.000]
])

def get_timestamp(filename, prefix):
    base = os.path.basename(filename)
    if "cloud" in base:
        return base.replace(f"{prefix}_cloud_", "").replace(".ply", "")
    elif "color" in base:
        return base.replace(f"{prefix}_color_", "").replace(".png", "")
    return ""

def load_and_prepare_cloud_orbbec(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    # Homogeneous coordinates
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    # Apply depth-to-color transform
    pts_color = (T_depth2color @ pts_h.T).T[:, :3]
    pcd.points = o3d.utility.Vector3dVector(pts_color)
    return pcd

def main():
    # 1. Create merge directory
    if not os.path.exists(DIR_MERGE):
        os.makedirs(DIR_MERGE)
        print(f"Created merge directory: {DIR_MERGE}")

    # 2. Load Calibration Matrix
    calib_path = os.path.join(BASE_DIR, "calibration_final.txt")
    if not os.path.exists(calib_path):
        print(f"Calibration file not found: {calib_path}")
        return
    
    T_calib = np.loadtxt(calib_path)
    print(f"Loaded calibration matrix from {calib_path}")
    print(T_calib)

    # 3. Process files
    files_o = sorted(glob.glob(os.path.join(DIR_ORBBEC, "orbbec_cloud_*.ply")))
    print(f"Found {len(files_o)} Orbbec files.")

    count = 0
    for f_o in files_o:
        ts = get_timestamp(f_o, "orbbec")
        f_k = os.path.join(DIR_KINECT, f"kinect2_cloud_{ts}.ply")
        
        if not os.path.exists(f_k):
            continue
            
        print(f"Processing timestamp: {ts}")
        
        # Load clouds
        pcd_o = load_and_prepare_cloud_orbbec(f_o)
        pcd_k = o3d.io.read_point_cloud(f_k)
        
        # Transform Orbbec to Kinect frame
        pcd_o.transform(T_calib)
        
        # Colorize
        # Orbbec = Red [1, 0, 0]
        pcd_o.paint_uniform_color([1, 0, 0])
        # Kinect = Green [0, 1, 0]
        pcd_k.paint_uniform_color([0, 1, 0])
        
        # Merge
        merged = pcd_o + pcd_k
        
        # Save
        out_name = f"merge_{ts}.ply"
        out_path = os.path.join(DIR_MERGE, out_name)
        o3d.io.write_point_cloud(out_path, merged)
        print(f"  Saved {out_path}")
        count += 1

    print(f"Done. Merged {count} pairs.")

if __name__ == "__main__":
    main()
