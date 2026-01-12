import numpy as np
import open3d as o3d
import cv2
import glob
import os
import copy
from scipy.spatial.transform import Rotation as R_scipy

# 先运行calibration_ICP.py获取初始PnP结果，然后用本脚本进行ICP细化

# ===============================
# Configuration
# ===============================
POSSIBLE_BASE_DIRS = [
    "/home/themountaintree/workspace/ros2_ws/data",
    os.path.join(os.getcwd(), "data"),
    os.path.join(os.path.dirname(os.getcwd()), "ros2_ws/data"), 
    "/data"
]

BASE_DIR = "/home/themountaintree/workspace/ros2_ws/data"
for p in POSSIBLE_BASE_DIRS:
    if os.path.exists(os.path.join(p, "orbbec")):
        BASE_DIR = p
        break

DIR_ORBBEC = os.path.join(BASE_DIR, "orbbec")
DIR_KINECT = os.path.join(BASE_DIR, "kinect2")

# Intrinsics
K_O = np.array([[745.9627, 0.0, 638.2298],
                [0.0, 745.2563, 360.5474],
                [0.0, 0.0, 1.0]])
D_O = np.array([0.07809, -0.10810, -0.00012, -0.00006, 0.04469])

K_K = np.array([[540.6860, 0.0, 479.75],
                [0.0, 540.6860, 269.75],
                [0.0, 0.0, 1.0]])
D_K = np.zeros(5)

# depth -> color (Femto Bolt)
T_depth2color = np.array([
    [ 1.000, -0.004, -0.004,  0.032],
    [ 0.004,  0.994, -0.105,  0.001],
    [ 0.004,  0.105,  0.994, -0.002],
    [ 0.000,  0.000,  0.000,  1.000]
])

# ===============================
# Utility
# ===============================
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
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_color = (T_depth2color @ pts_h.T).T[:, :3]
    pcd.points = o3d.utility.Vector3dVector(pts_color)
    return pcd

def preprocess_cloud(pcd, voxel=0.01):
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=30)
    )
    return pcd

# ===============================
# ICP refinement (核心)
# ===============================
def refine_with_icp(
    pcd_orbbec,
    pcd_kinect,
    T_init,
    max_corr_dist=0.05
):
    """
    PnP + ICP refinement
    - T_init: Orbbec -> Kinect (PnP result)
    - ICP only does small correction
    """

    src = copy.deepcopy(pcd_orbbec)
    tgt = copy.deepcopy(pcd_kinect)

    src = preprocess_cloud(src, voxel=0.01)
    tgt = preprocess_cloud(tgt, voxel=0.01)

    result = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_corr_dist,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50
        )
    )

    print("\n[ICP refinement]")
    print(f"  fitness: {result.fitness:.4f}")
    print(f"  inlier RMSE: {result.inlier_rmse:.4f}")

    return result.transformation

# ===============================
# Main
# ===============================
def main():
    # -------------------------------------------------
    # 1. Load PnP result (your existing output)
    # -------------------------------------------------
    # NOTE: Assuming the file is in BASE_DIR
    pnp_result_path = os.path.join(BASE_DIR, "calibration_symmetric_pnp.txt")
    if not os.path.exists(pnp_result_path):
        print(f"PnP result not found at {pnp_result_path}")
        return

    T_pnp = np.loadtxt(pnp_result_path)

    print("\nInitial PnP result:")
    print(T_pnp)

    # -------------------------------------------------
    # 2. Load one representative frame for ICP
    # -------------------------------------------------
    files_o = sorted(glob.glob(os.path.join(DIR_ORBBEC, "orbbec_cloud_*.ply")))
    
    pcd_o_path = None
    pcd_k_path = None
    
    print(f"Searching for a valid pair in {len(files_o)} Orbbec files...")
    
    for f_o in files_o:
        ts = get_timestamp(f_o, "orbbec")
        f_k = os.path.join(DIR_KINECT, f"kinect2_cloud_{ts}.ply")
        if os.path.exists(f_k):
            pcd_o_path = f_o
            pcd_k_path = f_k
            print(f"Found pair: {os.path.basename(pcd_o_path)} and {os.path.basename(pcd_k_path)}")
            break
            
    if pcd_o_path is None or pcd_k_path is None:
        print("No valid Orbbec-Kinect cloud pair found.")
        return

    pcd_o = load_and_prepare_cloud_orbbec(pcd_o_path)
    pcd_k = o3d.io.read_point_cloud(pcd_k_path)

    # -------------------------------------------------
    # 3. ICP refinement
    # -------------------------------------------------
    T_refined = refine_with_icp(
        pcd_o,
        pcd_k,
        T_pnp,
        max_corr_dist=0.05
    )

    print("\nRefined (PnP + ICP) result:")
    print(T_refined)

    # -------------------------------------------------
    # 4. Compare delta
    # -------------------------------------------------
    delta = np.linalg.inv(T_pnp) @ T_refined
    r = R_scipy.from_matrix(delta[:3, :3])
    angle = np.linalg.norm(r.as_rotvec()) * 180 / np.pi
    t = delta[:3, 3]

    print("\n[Delta introduced by ICP]")
    print(f"  rotation change: {angle:.3f} deg")
    print(f"  translation change: {np.linalg.norm(t)*100:.1f} cm")
    print(f"  delta t = {t}")

    # -------------------------------------------------
    # 5. Save
    # -------------------------------------------------
    out_path = os.path.join(BASE_DIR, "calibration_pnp_icp.txt")
    np.savetxt(out_path, T_refined, fmt="%.8f")
    print(f"\nSaved refined calibration to: {out_path}")

    # -------------------------------------------------
    # 6. Visual verification
    # -------------------------------------------------
    pcd_o.transform(T_refined)
    pcd_o.paint_uniform_color([1, 0, 0])
    pcd_k.paint_uniform_color([0, 1, 0])

    merged = pcd_o + pcd_k
    vis_path = os.path.join(BASE_DIR, "verify_pnp_icp.ply")
    o3d.io.write_point_cloud(vis_path, merged)
    print(f"Saved merged cloud: {vis_path}")

if __name__ == "__main__":
    main()
