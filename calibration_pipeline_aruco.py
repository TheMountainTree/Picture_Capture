import numpy as np
import cv2
import glob
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ===============================
# Configuration
# ===============================
BASE_DIR = "/home/themountaintree/workspace/ros2_ws/data"
DIR_ORBBEC = os.path.join(BASE_DIR, "orbbec")
DIR_KINECT = os.path.join(BASE_DIR, "kinect2")

# ⚠️ ARUCO CONFIGURATION
# 请根据实际使用的 ArUco 字典和 ID/尺寸进行修改
# 常用字典: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, etc.
ARUCO_DICT_ENUM = cv2.aruco.DICT_5X5_100
MARKER_ID = 22
MARKER_LENGTH = 0.100  # 物理尺寸 (meter)

K_O = np.array([[745.9627, 0.0, 638.2298],
                [0.0, 745.2563, 360.5474],
                [0.0, 0.0, 1.0]])
D_O = np.array([0.07809, -0.10810, -0.00012, -0.00006, 0.04469])

K_K = np.array([[540.6860, 0.0, 479.75],
                [0.0, 540.6860, 269.75],
                [0.0, 0.0, 1.0]])
D_K = np.zeros(5)

# ===============================
# Utilities
# ===============================
def generate_board():
    # ArUco Marker 4 corners in 3D (Local frame)
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # Coordinate system: X-Right, Y-Up (standard ArUco), Z-Out
    half = MARKER_LENGTH / 2.0
    objp = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)
    return objp

def detect_corners(path):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ENUM)
    parameters = cv2.aruco.DetectorParameters()
    # 启用亚像素角点优化
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        ids = ids.flatten()
        if MARKER_ID in ids:
            index = list(ids).index(MARKER_ID)
            # 返回该 ID 的 4 个角点 (4, 2)
            return corners[index].reshape(4, 2)
            
    return None

def solve_pnp(board, corners, K, D):
    # board: (4, 3), corners: (4, 2)
    ok, rvec, tvec = cv2.solvePnP(board, corners, K, D)
    if not ok:
        return None
    Rm,_ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = Rm
    T[:3,3] = tvec.flatten()
    return T

def reproj_error(board, corners, T, K, D):
    rvec,_ = cv2.Rodrigues(T[:3,:3])
    proj,_ = cv2.projectPoints(board, rvec, T[:3,3], K, D)
    return np.mean(np.linalg.norm(proj.reshape(-1,2) - corners, axis=1))

def relative_motion(T1, T2):
    return T2 @ np.linalg.inv(T1)

# ===============================
# Hand–Eye (robust)
# ===============================
def run_handeye():
    board = generate_board()

    Ts_o, Ts_k = [], []

    files_o = sorted(glob.glob(os.path.join(DIR_ORBBEC, "orbbec_color_*.png")))

    print(f"Searching for ArUco ID {MARKER_ID} (Dict={ARUCO_DICT_ENUM}) in {len(files_o)} files...")

    for f_o in files_o:
        ts = os.path.basename(f_o).replace("orbbec_color_","").replace(".png","")
        f_k = os.path.join(DIR_KINECT, f"kinect2_color_{ts}.png")
        if not os.path.exists(f_k):
            continue

        co = detect_corners(f_o)
        ck = detect_corners(f_k)
        if co is None or ck is None:
            # print(f"Skipping {ts}: Marker not found")
            continue

        T_bo = solve_pnp(board, co, K_O, D_O)
        T_bk = solve_pnp(board, ck, K_K, D_K)
        if T_bo is None or T_bk is None:
            continue

        eo = reproj_error(board, co, T_bo, K_O, D_O)
        ek = reproj_error(board, ck, T_bk, K_K, D_K)

        # Relaxed constraint for ArUco (usually very accurate, but keep similar thresholds or tighter)
        if eo > 5.0 or ek > 8.0:
            print(f"Skipping {ts}: Error too high (O={eo:.2f}, K={ek:.2f})")
            continue
        
        print(f"Frame {ts}: O={eo:.2f}, K={ek:.2f}")

        Ts_o.append(T_bo)
        Ts_k.append(T_bk)

    print(f"PnP 有效帧数: {len(Ts_o)}")
    if len(Ts_o) < 5:
        raise RuntimeError("PnP 帧数不足")

    # 构造相对运动
    A_R, A_t, B_R, B_t = [], [], [], []

    for i in range(len(Ts_o)-1):
        # A = Kinect (as Gripper)
        A = relative_motion(Ts_k[i], Ts_k[i+1])
        # B = Orbbec (as Target)
        B = relative_motion(Ts_o[i], Ts_o[i+1])

        rot_mag = np.linalg.norm(R.from_matrix(A[:3,:3]).as_rotvec())
        if rot_mag < np.deg2rad(5):  # 激励不足
            continue

        A_R.append(A[:3,:3])
        A_t.append(A[:3,3])
        B_R.append(B[:3,:3])
        B_t.append(B[:3,3])

    print(f"Hand–Eye 使用帧对数: {len(A_R)}")
    if len(A_R) < 3:
        raise RuntimeError("运动激励不足，Hand–Eye 病态")

    R_oc, t_oc = cv2.calibrateHandEye(
        A_R, A_t, B_R, B_t,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T = np.eye(4)
    T[:3,:3] = R_oc
    T[:3,3] = t_oc.flatten()
    return T

# ===============================
# ICP Refinement (Level 4)
# ===============================
def refine_icp_multiframe(T_init, n_frames=5):
    print(f"\n[ICP] Refining with {n_frames} frames...")
    
    files_o = sorted(glob.glob(os.path.join(DIR_ORBBEC, "orbbec_cloud_*.ply")))
    
    # Filter valid pairs first
    valid_pairs = []
    for f_o in files_o:
        ts = os.path.basename(f_o).replace("orbbec_cloud_", "").replace(".ply", "")
        f_k = os.path.join(DIR_KINECT, f"kinect2_cloud_{ts}.ply")
        if os.path.exists(f_k):
            valid_pairs.append((f_o, f_k))
            
    if len(valid_pairs) == 0:
        print("[ICP] No point cloud pairs found. Skipping refinement.")
        return T_init

    # Select frames (equally spaced)
    if len(valid_pairs) > n_frames:
        indices = np.linspace(0, len(valid_pairs)-1, n_frames, dtype=int)
        selected_pairs = [valid_pairs[i] for i in indices]
    else:
        selected_pairs = valid_pairs

    deltas_R = []
    deltas_t = []
    
    T_inv_init = np.linalg.inv(T_init)

    for f_o, f_k in selected_pairs:
        print(f"  - ICP on {os.path.basename(f_o)} ...")
        
        src = o3d.io.read_point_cloud(f_o)
        tgt = o3d.io.read_point_cloud(f_k)
        
        if src.is_empty() or tgt.is_empty():
            print("    Empty point cloud, skipping.")
            continue

        # Preprocessing
        src = src.voxel_down_sample(0.005) 
        tgt = tgt.voxel_down_sample(0.005)
        src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        
        # ICP (Coarse-to-Fine)
        # Stage 1: Coarse (0.2m)
        reg_coarse = o3d.pipelines.registration.registration_icp(
            src, tgt, 0.2, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        # Stage 2: Fine (0.02m)
        reg_fine = o3d.pipelines.registration.registration_icp(
            src, tgt, 0.02, reg_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        print(f"    Coarse Fitness: {reg_coarse.fitness:.4f}, Fine Fitness: {reg_fine.fitness:.4f}")

        T_res = reg_fine.transformation
        
        # Compute adjustment delta: T_res = Delta * T_init  =>  Delta = T_res * inv(T_init)
        Delta = T_res @ T_inv_init
        
        deltas_R.append(Delta[:3,:3])
        deltas_t.append(Delta[:3,3])

    if not deltas_R:
        print("[ICP] No successful ICP registrations.")
        return T_init

    # Average Deltas
    # Average rotation using rotation vectors
    r_vecs = [R.from_matrix(r).as_rotvec() for r in deltas_R]
    avg_r_vec = np.mean(r_vecs, axis=0)
    avg_R = R.from_rotvec(avg_r_vec).as_matrix()
    
    avg_t = np.mean(deltas_t, axis=0)
    
    Delta_avg = np.eye(4)
    Delta_avg[:3,:3] = avg_R
    Delta_avg[:3,3] = avg_t
    
    print("  Average ICP Delta (t):", np.linalg.norm(avg_t))
    print("  Average ICP Delta (R deg):", np.linalg.norm(avg_r_vec)*180/np.pi)

    T_final = Delta_avg @ T_init
    return T_final

# ===============================
# Main
# ===============================
def main():
    try:
        T = run_handeye()
        print("\nHand–Eye result:\n", T)
        
        T_ref = refine_icp_multiframe(T)
        print("\nFinal result (after ICP):\n", T_ref)
        
        np.savetxt(os.path.join(BASE_DIR, "calibration_final.txt"), T_ref, fmt="%.8f")
        
    except Exception as e:
        print(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
