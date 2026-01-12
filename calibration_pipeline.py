import numpy as np
import open3d as o3d
import cv2
import glob
import os
import copy
from scipy.spatial.transform import Rotation as R_scipy

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

BOARD_SIZE = (11, 8)
SQUARE_SIZE = 0.020

# ===============================
# Shared Utilities
# ===============================
def get_timestamp(filename, prefix):
    base = os.path.basename(filename)
    if "cloud" in base:
        return base.replace(f"{prefix}_cloud_", "").replace(".ply", "")
    elif "color" in base:
        return base.replace(f"{prefix}_color_", "").replace(".png", "")
    return ""

# ===============================
# PnP Calibration Functions (From calibration_ICP.py)
# ===============================
def generate_ideal_board(square_size, cols=11, rows=8):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp

def detect_corners(img_path, board_size):
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    if not ret:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if not ret:
        return None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return corners.reshape(-1, 2), img.shape

def extract_board_pointcloud(pcd_path, corners_2d, K, D, img_shape, T_dc):
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        return None

    pts_d = np.asarray(pcd.points)
    pts_h = np.hstack([pts_d, np.ones((pts_d.shape[0], 1))])
    pts_c = (T_dc @ pts_h.T).T[:, :3]

    h, w = img_shape[:2]
    pts_2d, _ = cv2.projectPoints(pts_c, np.zeros(3), np.zeros(3), K, D)
    pts_2d = pts_2d.reshape(-1, 2)

    hull = cv2.convexHull(corners_2d.astype(np.float32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)

    u = np.round(pts_2d[:, 0]).astype(int)
    v = np.round(pts_2d[:, 1]).astype(int)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.where(valid)[0][mask[v[valid], u[valid]] > 0]

    if len(idx) < 50:
        return None

    roi_pts = pts_c[idx]
    z = roi_pts[:, 2]
    roi_pts = roi_pts[(z > 0.1) & (z < 5.0) & (np.abs(z - np.median(z)) < 0.15)]

    if len(roi_pts) < 50:
        return None

    return roi_pts

def fit_plane_ransac(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(0.01, 3, 1000)
    if len(inliers) < len(points) * 0.5:
        return None, []
    return plane_model, inliers

def reconstruct_corners_ray_plane(corners_2d, plane_model, K, D):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])

    corners_u = cv2.undistortPoints(corners_2d, K, D, P=K).reshape(-1, 2)
    pts_3d, valid_idx = [], []

    for i, (u, v) in enumerate(corners_u):
        ray = np.array([(u - K[0,2]) / K[0,0],
                        (v - K[1,2]) / K[1,1],
                        1.0])
        ray /= np.linalg.norm(ray)
        denom = normal @ ray
        if abs(denom) < 1e-6:
            continue
        t = -d / denom
        if 0.1 < t < 5.0:
            pts_3d.append(ray * t)
            valid_idx.append(i)

    if len(pts_3d) < 20:
        return None, None

    return np.array(pts_3d), np.array(valid_idx)

def estimate_actual_square_size(corners_3d, valid_indices, cols=11):
    dists = []
    for idx in valid_indices:
        r, c = idx // cols, idx % cols
        p = np.where(valid_indices == idx)[0][0]
        if c < cols - 1 and idx + 1 in valid_indices:
            q = np.where(valid_indices == idx + 1)[0][0]
            dists.append(np.linalg.norm(corners_3d[p] - corners_3d[q]))
        if r < BOARD_SIZE[1] - 1 and idx + cols in valid_indices:
            q = np.where(valid_indices == idx + cols)[0][0]
            dists.append(np.linalg.norm(corners_3d[p] - corners_3d[q]))
    return np.median(dists) if dists else None

def process_frame(img_o, pcd_o, img_k, ts):
    corners_o, shape_o = detect_corners(img_o, BOARD_SIZE)
    if corners_o is None:
        return None, "Orbbec 角点失败"

    corners_k, _ = detect_corners(img_k, BOARD_SIZE)
    if corners_k is None:
        return None, "Kinect 角点失败"

    board_pts = extract_board_pointcloud(
        pcd_o, corners_o, K_O, D_O, shape_o, T_depth2color
    )
    if board_pts is None:
        return None, "点云提取失败"

    plane, _ = fit_plane_ransac(board_pts)
    if plane is None:
        return None, "平面拟合失败"

    corners_3d, valid = reconstruct_corners_ray_plane(corners_o, plane, K_O, D_O)
    if corners_3d is None:
        return None, "3D 重建失败"

    sq = estimate_actual_square_size(corners_3d, valid, BOARD_SIZE[0])
    if sq is None or not (0.01 < sq < 0.05):
        sq = SQUARE_SIZE

    board = generate_ideal_board(sq, BOARD_SIZE[0], BOARD_SIZE[1])

    ok, r_o, t_o = cv2.solvePnP(board, corners_o, K_O, D_O)
    if not ok:
        return None, "Orbbec PnP 失败"

    ok, r_k, t_k = cv2.solvePnP(board, corners_k, K_K, D_K)
    if not ok:
        return None, "Kinect PnP 失败"

    R_o, _ = cv2.Rodrigues(r_o)
    R_k, _ = cv2.Rodrigues(r_k)

    T_m2o = np.eye(4); T_m2o[:3,:3] = R_o; T_m2o[:3,3] = t_o.flatten()
    T_m2k = np.eye(4); T_m2k[:3,:3] = R_k; T_m2k[:3,3] = t_k.flatten()

    T_o2k = T_m2k @ np.linalg.inv(T_m2o)

    err_o = np.mean(np.linalg.norm(
        corners_o - cv2.projectPoints(board, r_o, t_o, K_O, D_O)[0].reshape(-1,2), axis=1))
    err_k = np.mean(np.linalg.norm(
        corners_k - cv2.projectPoints(board, r_k, t_k, K_K, D_K)[0].reshape(-1,2), axis=1))

    return T_o2k, {
        "reproj_error_o": err_o,
        "reproj_error_k": err_k,
        "actual_square_size": sq
    }


def karcher_mean_rotation(rotations, max_iter=50, tol=1e-6):
    """Karcher Mean for SO(3)"""
    if not rotations:
        return np.eye(3)
    
    R_mean = rotations[0].copy()
    
    for _ in range(max_iter):
        vec_sum = np.zeros(3)
        for R in rotations:
            diff = R_mean.T @ R
            vec_sum += R_scipy.from_matrix(diff).as_rotvec()
        vec_avg = vec_sum / len(rotations)
        
        if np.linalg.norm(vec_avg) < tol:
            break
        
        R_update = R_scipy.from_rotvec(vec_avg).as_matrix()
        R_mean = R_mean @ R_update
    
    return R_mean

def average_transforms_robust(transforms, infos):
    """
    鲁棒平均 (基于重投影误差)
    """
    if not transforms:
        return np.eye(4)
    
    # 使用两个相机的平均重投影误差
    errors = np.array([
        (info['reproj_error_o'] + info['reproj_error_k']) / 2.0 
        for info in infos
    ])
    
    print(f"\n  重投影误差统计:")
    print(f"    中位数: {np.median(errors):.2f} pixels")
    print(f"    范围: {errors.min():.2f} - {errors.max():.2f} pixels")
    
    # 过滤: 误差 > 2.0 pixels 的丢弃
    error_threshold = min(np.median(errors) * 1.5, 2.0)
    valid_mask = errors < error_threshold
    
    if np.sum(valid_mask) < 2:
        print(f"  警告: 过滤后仅剩 {np.sum(valid_mask)} 帧，放宽阈值")
        error_threshold = np.median(errors) * 2.0
        valid_mask = errors < error_threshold
    
    valid_transforms = [T for i, T in enumerate(transforms) if valid_mask[i]]
    valid_errors = errors[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    print(f"  质量过滤: 保留 {len(valid_transforms)}/{len(transforms)} 帧")
    
    if len(valid_transforms) == 0:
        return np.eye(4)
    
    # 检查平移一致性
    ts = np.array([T[:3, 3] for T in valid_transforms])
    t_median = np.median(ts, axis=0)
    t_dists = np.linalg.norm(ts - t_median, axis=1)
    
    print(f"\n  平移一致性:")
    print(f"    中位数: {t_median}")
    print(f"    偏差范围: {t_dists.min():.3f} - {t_dists.max():.3f} m")
    
    # 过滤平移离群值
    t_valid_mask = t_dists < 0.15
    
    if np.sum(t_valid_mask) >= 2:
        valid_transforms = [T for i, T in enumerate(valid_transforms) if t_valid_mask[i]]
        valid_errors = valid_errors[t_valid_mask]
        print(f"    平移过滤: 保留 {len(valid_transforms)} 帧")
    
    # 计算权重
    weights = 1.0 / (valid_errors + 0.1)
    weights = weights / weights.sum()
    
    # 加权平均
    ts = np.array([T[:3, 3] for T in valid_transforms])
    t_avg = np.average(ts, axis=0, weights=weights)
    
    rs = [T[:3, :3] for T in valid_transforms]
    R_avg = karcher_mean_rotation(rs)
    
    T_result = np.eye(4)
    T_result[:3, :3] = R_avg
    T_result[:3, 3] = t_avg
    
    return T_result

# ===============================
# ICP Refinement Functions (From calibration_pnp_icp_refine.py)
# ===============================
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
# Pipeline Stages
# ===============================
def run_pnp_calibration():
    print("="*60)
    print("STEP 1: PnP Calibration (Symmetric PnP)")
    print("="*60)
    print(f"数据目录: {BASE_DIR}")
    print(f"棋盘格: {BOARD_SIZE[0]}x{BOARD_SIZE[1]}, 理论方格: {SQUARE_SIZE*1000}mm\n")
    
    files_o_img = sorted(glob.glob(os.path.join(DIR_ORBBEC, "orbbec_color_*.png")))
    
    print(f"找到 {len(files_o_img)} 张图像\n")
    
    transforms = []
    infos = []
    
    for i, img_o in enumerate(files_o_img):
        ts = get_timestamp(img_o, "orbbec")
        
        img_k = os.path.join(DIR_KINECT, f"kinect2_color_{ts}.png")
        pcd_o = os.path.join(DIR_ORBBEC, f"orbbec_cloud_{ts}.ply")
        
        if not all(os.path.exists(f) for f in [img_k, pcd_o]):
            continue
        
        print(f"[{i+1:2d}/{len(files_o_img)}] {ts}...", end=" ")
        
        T_o2k, info = process_frame(img_o, pcd_o, img_k, ts)
        
        if T_o2k is None:
            print(f"✗ {info}")
            continue
        
        transforms.append(T_o2k)
        infos.append(info)
        
        print(f"✓ 误差: O={info['reproj_error_o']:.2f}px, K={info['reproj_error_k']:.2f}px, "
              f"尺寸={info['actual_square_size']*1000:.1f}mm")
    
    print(f"\n{'='*60}")
    print(f"成功: {len(transforms)}/{len(files_o_img)} 帧")
    print(f"{'='*60}\n")
    
    if len(transforms) == 0:
        print("❌ 没有成功的帧!")
        return None
    
    if len(transforms) < 3:
        print(f"⚠️  仅 {len(transforms)} 帧，建议至少 10 帧\n")
    
    # 鲁棒平均
    T_final = average_transforms_robust(transforms, infos)
    
    # Save intermediate result (optional but good for debugging)
    result_path = os.path.join(BASE_DIR, "calibration_symmetric_pnp.txt")
    np.savetxt(result_path, T_final, fmt='%.8f')
    print(f"\nStep 1 Result Saved to: {result_path}")
    print(T_final)
    return T_final

def run_icp_refinement(T_pnp):
    print("\n" + "="*60)
    print("STEP 2: ICP Refinement")
    print("="*60)
    
    # Find a valid pair for ICP
    files_o = sorted(glob.glob(os.path.join(DIR_ORBBEC, "orbbec_cloud_*.ply")))
    pcd_o_path = None
    pcd_k_path = None
    
    print(f"Searching for a valid point cloud pair in {len(files_o)} Orbbec files...")
    
    for f_o in files_o:
        ts = get_timestamp(f_o, "orbbec")
        f_k = os.path.join(DIR_KINECT, f"kinect2_cloud_{ts}.ply")
        if os.path.exists(f_k):
            pcd_o_path = f_o
            pcd_k_path = f_k
            print(f"Found pair for ICP: {os.path.basename(pcd_o_path)} and {os.path.basename(pcd_k_path)}")
            break
            
    if pcd_o_path is None or pcd_k_path is None:
        print("No valid Orbbec-Kinect cloud pair found for ICP.")
        return None

    pcd_o = load_and_prepare_cloud_orbbec(pcd_o_path)
    pcd_k = o3d.io.read_point_cloud(pcd_k_path)

    # ICP refinement
    T_refined = refine_with_icp(
        pcd_o,
        pcd_k,
        T_pnp,
        max_corr_dist=0.05
    )

    print("\nRefined (PnP + ICP) result:")
    print(T_refined)

    # Compare delta
    delta = np.linalg.inv(T_pnp) @ T_refined
    r = R_scipy.from_matrix(delta[:3, :3])
    angle = np.linalg.norm(r.as_rotvec()) * 180 / np.pi
    t = delta[:3, 3]

    print("\n[Delta introduced by ICP]")
    print(f"  rotation change: {angle:.3f} deg")
    print(f"  translation change: {np.linalg.norm(t)*100:.1f} cm")
    print(f"  delta t = {t}")
    
    return T_refined

# ===============================
# Main
# ===============================
def main():
    # 1. Run PnP
    T_pnp = run_pnp_calibration()
    if T_pnp is None:
        print("PnP Calibration failed. Aborting.")
        return

    # 2. Run ICP Refinement
    T_final = run_icp_refinement(T_pnp)
    if T_final is None:
        print("ICP Refinement failed. Aborting.")
        return

    # 3. Save Final Result
    final_out_path = os.path.join(BASE_DIR, "calibration_final.txt")
    np.savetxt(final_out_path, T_final, fmt="%.8f")
    print(f"\n" + "="*60)
    print(f"FINAL CALIBRATION SAVED TO: {final_out_path}")
    print("="*60)
    print(T_final)

if __name__ == "__main__":
    main()
