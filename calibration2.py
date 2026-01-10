"""
代码原理与流程说明:

1. 原理:
   本程序实现 Orbbec 相机与 Kinect 相机之间的外参标定（Rigid Transformation）。
   采用"混合标定"（Hybrid Calibration）策略，结合了稀疏特征点和稠密点云的优势：
   - 粗标定 (Phase 1): 利用棋盘格角点的稀疏对应关系 (Sparse Correspondences)，通过 RANSAC 算法剔除误匹配，并使用 LM (Levenberg-Marquardt) 优化求解初始变换矩阵。
   - 精标定 (Phase 2): 利用棋盘格区域的稠密点云 (Dense Point Clouds)，通过点对面 ICP (Point-to-Plane ICP) 算法进一步微调变换矩阵，提高标定精度。

2. 流程:
   (1) 环境配置: 设置数据路径、相机内参 (Intrinsics) 和畸变系数 (Distortion)。
   (2) 数据读取与预处理: 遍历数据目录，配对读取 Orbbec 和 Kinect 的 RGB 图像及对应的 PLY 点云文件。
   (3) 特征提取 (process_camera_data):
       - 对每幅图像检测棋盘格角点。
       - 将 2D 角点通过内参和点云深度信息反投影到 3D 空间，获取稀疏 3D 角点。
   (4) 稠密点云提取 (extract_dense_board_cloud):
       - 根据棋盘格角点围成的区域生成 2D Mask。
       - 将点云投影回像素坐标系，提取位于 Mask 区域内的点，并进行深度范围滤波。
   (5) Phase 1 - 稀疏粗配准:
       - 匹配两组相机的 3D 角点。
       - 使用 cv2.estimateAffine3D (RANSAC) 计算初始变换矩阵。
       - 使用 least_squares (LM算法) 对内点进行非线性优化，得到 T_coarse。
   (6) Phase 2 - 稠密精配准:
       - 融合多帧提取的稠密棋盘格点云。
       - 使用 T_coarse 作为初值，运行点对面 ICP 算法，得到最终变换矩阵 T_final。
   (7) 结果输出: 保存变换矩阵到文本文件，并保存配准后的可视化点云以供验证。
"""

import numpy as np
import open3d as o3d
import cv2
import glob
import os
import shutil
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

# --- Configuration ---
# Robust path handling
POSSIBLE_BASE_DIRS = [
    "/home/themountaintree/workspace/ros2_ws/data",
    os.path.join(os.getcwd(), "data"),
    os.path.join(os.path.dirname(os.getcwd()), "ros2_ws/data"), 
    "/data"
]

BASE_DIR = "/home/themountaintree/workspace/ros2_ws/data" # Default
for p in POSSIBLE_BASE_DIRS:
    if os.path.exists(os.path.join(p, "orbbec")):
        BASE_DIR = p
        print(f"Located data directory at: {BASE_DIR}")
        break

DIR_ORBBEC = os.path.join(BASE_DIR, "orbbec")
DIR_KINECT = os.path.join(BASE_DIR, "kinect2")

# Intrinsics
# Orbbec 相机内参和畸变
K_O = np.array([[745.9627, 0.0, 638.2298], [0.0, 745.2563, 360.5474], [0.0, 0.0, 1.0]])
D_O = np.array([0.07809, -0.10810, -0.00012, -0.00006, 0.04469])

# Kinect 相机内参和畸变
K_K = np.array([[540.6860, 0.0, 479.75], [0.0, 540.6860, 269.75], [0.0, 0.0, 1.0]])
D_K = np.zeros(5)

BOARD_SIZE = (11, 8) # 棋盘格内角点数量 (cols, rows)

def get_timestamp(filename, prefix):
    """
    从文件名中解析时间戳。
    
    参数:
        filename (str): 文件路径或文件名。
        prefix (str): 文件名前缀 (如 "orbbec", "kinect2")，用于定位时间戳位置。
        
    返回:
        str: 提取出的时间戳字符串。若无法解析则返回空字符串。
    """
    base = os.path.basename(filename)
    if "cloud" in base:
        return base.replace(f"{prefix}_cloud_", "").replace(".ply", "")
    elif "color" in base:
        return base.replace(f"{prefix}_color_", "").replace(".png", "")
    return ""

def back_project(u, v, z, K):
    """
    利用针孔相机模型，将 2D 像素坐标和深度值反投影为 3D 坐标。
    
    参数:
        u (float/int): 像素横坐标。
        v (float/int): 像素纵坐标。
        z (float): 该像素点的深度值 (单位: 米)。
        K (np.array): 相机内参矩阵 (3x3)。
        
    返回:
        np.array: 对应的 3D 点坐标 [x, y, z]。
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])

def extract_dense_board_cloud(img_shape, corners, pcd_path, K, D):
    """
    根据 2D 角点生成 Mask，提取棋盘格区域的稠密点云。
    
    参数:
        img_shape (tuple): 图像尺寸 (height, width, ...)。
        corners (np.array): 检测到的棋盘格 2D 角点坐标 [N, 2]。
        pcd_path (str): 对应的点云文件路径 (.ply)。
        K (np.array): 相机内参矩阵。
        D (np.array): 畸变系数。
        
    返回:
        np.array or None: 提取出的稠密点云数组 [M, 3]。若提取失败或点数过少，返回 None。
    """
    # 1. 生成 Mask
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(corners.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    # 腐蚀操作，去除边缘噪声点 (Erosion to remove edge noise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    # 2. 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd or len(pcd.points) == 0: return None
    pts_3d = np.asarray(pcd.points)
    
    # 3. 投影 3D 点到 2D 以应用 Mask
    if len(pts_3d) > 0:
        # 这里假设点云坐标系与图像坐标系一致（除了畸变）
        pts_2d_proj, _ = cv2.projectPoints(pts_3d, np.zeros(3), np.zeros(3), K, D)
        pts_2d_proj = pts_2d_proj.reshape(-1, 2)
        
        u = np.round(pts_2d_proj[:, 0]).astype(int)
        v = np.round(pts_2d_proj[:, 1]).astype(int)
        
        h, w = img_shape[:2]
        valid_indices = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        # 筛选 Mask 内的点
        # 使用 numpy 向量化操作加速
        valid_u = u[valid_indices]
        valid_v = v[valid_indices]
        
        # 检查 mask 值
        mask_vals = mask[valid_v, valid_u]
        pts_indices = np.where(valid_indices)[0]
        final_indices = pts_indices[mask_vals > 0]
                
        if len(final_indices) < 100: 
            return None
            
        dense_points = pts_3d[final_indices]
        
        # 深度范围滤波 (假设拍摄距离 0.5m - 3.0m)
        z = dense_points[:, 2]
        z_mean = np.median(z)
        mask_z = (z > 0.5) & (z < 3.0) & (np.abs(z - z_mean) < 0.1)
        dense_points = dense_points[mask_z]
        
        return dense_points
    return None

def process_camera_data(img_path, pcd_path, K, D):
    """
    处理单帧相机数据：检测角点并映射到 3D 空间。
    
    参数:
        img_path (str): RGB 图像路径。
        pcd_path (str): 对应的点云文件路径。
        K (np.array): 相机内参矩阵。
        D (np.array): 畸变系数。
        
    返回:
        tuple: (valid_indices, valid_corners_3d, corners, img_shape)
            - valid_indices (np.array): 有效的（能找到对应深度的）角点索引列表。
            - valid_corners_3d (np.array): 对应的 3D 角点坐标 [N, 3]。
            - corners (np.array): 检测到的所有 2D 角点 [M, 2]。
            - img_shape (tuple): 图像尺寸。
            若处理过程中出现错误或找不到角点，返回 (None, None, None, None)。
    """
    # 1. Load Image
    img = cv2.imread(img_path)
    if img is None: return None, None, None, None
    
    # 2. Detect Corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
    if not ret:
        # 尝试自适应阈值
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, flags)
    if not ret: return None, None, None, None
    
    # 亚像素级角点优化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    corners = corners.reshape(-1, 2)
    
    # 3. 稀疏角点提取 (用于初值计算)
    # 去畸变
    corners_undist = cv2.undistortPoints(corners, K, D, P=K).reshape(-1, 2)
    
    # 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd: return None, None, None, None
    pts_3d_all = np.asarray(pcd.points)
    
    # 将所有点云投影到图像平面，建立 KDTree 用于查找角点对应的深度
    pts_2d_proj, _ = cv2.projectPoints(pts_3d_all, np.zeros(3), np.zeros(3), K, D)
    pts_2d_proj = pts_2d_proj.reshape(-1, 2)
    tree = cKDTree(pts_2d_proj)
    
    valid_corners_3d = []
    valid_indices = []
    
    for i, (u, v) in enumerate(corners):
        # 查找角点周围 2 像素半径内的点云点
        indices = tree.query_ball_point([u, v], r=2.0)
        if len(indices) < 3: continue
        local_pts = pts_3d_all[indices]
        depths = local_pts[:, 2]
        
        # 严格滤波: 去除无效深度和离群值
        mask_val = (depths > 0.5) & (depths < 3.0)
        depths = depths[mask_val]
        if len(depths) < 3: continue
        z_med = np.median(depths)
        mask_cln = np.abs(depths - z_med) < 0.02 # 保留中值附近 2cm 内的点
        depths_cln = depths[mask_cln]
        if len(depths_cln) < 3: continue
        z_fin = np.mean(depths_cln) # 取平均作为该角点深度
        
        # 反投影得到 3D 角点
        u_u, v_u = corners_undist[i]
        valid_corners_3d.append(back_project(u_u, v_u, z_fin, K))
        valid_indices.append(i)
        
    return np.array(valid_indices), np.array(valid_corners_3d), corners, img.shape

def main():
    """
    主函数：执行完整的混合标定流程。
    """
    print("Starting Hybrid Calibration (Sparse Init + Dense Grid Refine)...")
    
    files_o = sorted(glob.glob(os.path.join(DIR_ORBBEC, "orbbec_cloud_*.ply")))
    
    # 存储两类数据
    sparse_matches_A = [] # [N, 3] Orbbec 端角点
    sparse_matches_B = [] # [N, 3] Kinect 端角点
    
    dense_clouds_A = [] # List of PointCloud (Orbbec)
    dense_clouds_B = [] # List of PointCloud (Kinect)
    valid_timestamps = []
    
    print(f"Found {len(files_o)} files.")
    
    for f_o in files_o:
        ts = get_timestamp(f_o, "orbbec")
        f_k = os.path.join(DIR_KINECT, f"kinect2_cloud_{ts}.ply")
        img_o = os.path.join(DIR_ORBBEC, f"orbbec_color_{ts}.png")
        img_k = os.path.join(DIR_KINECT, f"kinect2_color_{ts}.png")
        
        if not (os.path.exists(f_k) and os.path.exists(img_o) and os.path.exists(img_k)): continue
            
        print(f"Processing {ts}...")
        
        # 1. 提取数据 (检测角点并计算3D坐标)
        idx_o, pts_o, corners_o, shape_o = process_camera_data(img_o, f_o, K_O, D_O)
        idx_k, pts_k, corners_k, shape_k = process_camera_data(img_k, f_k, K_K, D_K)
        
        if idx_o is None or idx_k is None: continue
            
        # 2. 稀疏点匹配 (找到同一 ID 的角点)
        common = np.intersect1d(idx_o, idx_k)
        if len(common) < 5: continue
            
        dict_o = {i: p for i, p in zip(idx_o, pts_o)}
        dict_k = {i: p for i, p in zip(idx_k, pts_k)}
        
        for i in common:
            sparse_matches_A.append(dict_o[i])
            sparse_matches_B.append(dict_k[i])
            
        # 3. 提取稠密 Mask 点云 (用于后续精配准)
        dense_pts_o = extract_dense_board_cloud(shape_o, corners_o, f_o, K_O, D_O)
        dense_pts_k = extract_dense_board_cloud(shape_k, corners_k, f_k, K_K, D_K)
        
        if dense_pts_o is not None and dense_pts_k is not None:
            # 转为 Open3D 格式并降采样，提高处理速度
            pcd_o = o3d.geometry.PointCloud()
            pcd_o.points = o3d.utility.Vector3dVector(dense_pts_o)
            pcd_o = pcd_o.voxel_down_sample(voxel_size=0.005) # 5mm
            
            pcd_k = o3d.geometry.PointCloud()
            pcd_k.points = o3d.utility.Vector3dVector(dense_pts_k)
            pcd_k = pcd_k.voxel_down_sample(voxel_size=0.005)
            
            dense_clouds_A.append(pcd_o)
            dense_clouds_B.append(pcd_k)
            valid_timestamps.append(ts)
            
    pts_A_sparse = np.array(sparse_matches_A)
    pts_B_sparse = np.array(sparse_matches_B)
    
    print(f"\nCollected Data: Sparse Pairs={len(pts_A_sparse)}, Dense Frames={len(dense_clouds_A)}")
    
    if len(pts_A_sparse) < 10:
        print("Not enough points found. Exiting.")
        return

    # --- Phase 1: Sparse RANSAC + LM (Get Coarse Init) ---
    print("\nPhase 1: Sparse Coarse Alignment (RANSAC)...")
    
    # 使用 RANSAC 估算刚性变换
    retval, T_ransac, inliers = cv2.estimateAffine3D(pts_A_sparse, pts_B_sparse, ransacThreshold=0.02)
    if not retval:
        print("RANSAC failed.")
        return
        
    print(f"RANSAC Inliers: {np.sum(inliers)}/{len(pts_A_sparse)}")
    T_coarse = np.vstack([T_ransac, [0,0,0,1]])
    
    # LM Optimization (Optional but good practice)
    # 对 RANSAC 的内点进行非线性优化，进一步优化粗配准结果
    pts_A_in = pts_A_sparse[inliers.ravel() > 0]
    pts_B_in = pts_B_sparse[inliers.ravel() > 0]
    
    def loss_func(params, A, B):
        r = params[:3]
        t = params[3:]
        R = R_scipy.from_rotvec(r).as_matrix()
        return (np.dot(A, R.T) + t - B).flatten()
        
    r_init = R_scipy.from_matrix(T_coarse[:3, :3]).as_rotvec()
    t_init = T_coarse[:3, 3]
    res = least_squares(loss_func, np.concatenate([r_init, t_init]), args=(pts_A_in, pts_B_in))
    
    R_lm = R_scipy.from_rotvec(res.x[:3]).as_matrix()
    t_lm = res.x[3:]
    T_coarse = np.eye(4)
    T_coarse[:3,:3] = R_lm
    T_coarse[:3,3] = t_lm
    
    # Calculate Sparse RMSE
    ones = np.ones((len(pts_A_in), 1))
    pts_A_in_h = np.hstack([pts_A_in, ones])
    pts_A_trans = (T_coarse @ pts_A_in_h.T).T[:, :3]
    errors = np.linalg.norm(pts_A_trans - pts_B_in, axis=1)
    rmse_sparse = np.sqrt(np.mean(errors**2))
    
    print("Coarse Transform (LM refined):")
    print(T_coarse)
    print(f"Sparse Alignment RMSE: {rmse_sparse:.6f} m")

    # --- Phase 2: Dense Grid ICP (Refinement) ---
    print("\nPhase 2: Dense Grid ICP Refinement...")
    
    if len(dense_clouds_A) == 0:
        print("No dense clouds extracted. Using coarse result.")
        T_final = T_coarse
    else:
        # Merge all dense clouds (将所有帧的稠密点云合并，以增加约束)
        full_source = o3d.geometry.PointCloud()
        full_target = o3d.geometry.PointCloud()
        
        for p in dense_clouds_A: full_source += p
        for p in dense_clouds_B: full_target += p
        
        print(f"Merged Dense Cloud Size: {len(full_source.points)}")
        
        # Estimate Normals for Point-to-Plane (计算法线)
        print("Estimating normals...")
        full_source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        full_target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        
        # ICP (Point-to-Plane is better for sliding planes)
        # Using T_coarse as initial guess
        print("Running ICP (Point-to-Plane)...")
        
        # Evaluate before ICP
        threshold = 0.03
        eval_init = o3d.pipelines.registration.evaluate_registration(
            full_source, full_target, threshold, T_coarse)
        print(f"Initial Dense Fitness (Before ICP): {eval_init.fitness}")
        print(f"Initial Dense RMSE (Before ICP): {eval_init.inlier_rmse}")
        
        reg_p2l = o3d.pipelines.registration.registration_icp(
            full_source, full_target, threshold, T_coarse,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        
        T_final = reg_p2l.transformation
        print(f"Final Dense Fitness (After ICP): {reg_p2l.fitness}")
        print(f"Final Dense RMSE (After ICP): {reg_p2l.inlier_rmse}")

    print("\n" + "="*50)
    print("Final Calibration Result (Hybrid)")
    print("="*50)
    print(T_final)
    
    # Save result
    np.savetxt(os.path.join(BASE_DIR, "calibration_result_hybrid.txt"), T_final)
    print("Saved matrix to text file.")
    
    # --- Visualization ---
    if len(dense_clouds_A) > 0:
        print("Saving debug visualization...")
        # Use the first captured frame for clear visualization
        vis_source = dense_clouds_A[0]
        vis_target = dense_clouds_B[0]
        vis_source.transform(T_final)
        o3d.io.write_point_cloud(os.path.join(BASE_DIR, "vis_aligned_source.ply"), vis_source)
        o3d.io.write_point_cloud(os.path.join(BASE_DIR, "vis_target.ply"), vis_target)
        print("Saved vis_aligned_source.ply and vis_target.ply")

        # Save full clouds for the first valid frame
        best_ts = valid_timestamps[0]
        print(f"\nSaving full verification clouds using Frame {best_ts}")
        
        o_full_path = os.path.join(DIR_ORBBEC, f"orbbec_cloud_{best_ts}.ply")
        k_full_path = os.path.join(DIR_KINECT, f"kinect2_cloud_{best_ts}.ply")
        
        if os.path.exists(o_full_path) and os.path.exists(k_full_path):
            pcd_o_full = o3d.io.read_point_cloud(o_full_path)
            pcd_k_full = o3d.io.read_point_cloud(k_full_path)
            
            pcd_o_full.transform(T_final)
            
            o_out = os.path.join(BASE_DIR, "final_aligned_orbbec.ply")
            k_out = os.path.join(BASE_DIR, "final_target_kinect.ply")
            o3d.io.write_point_cloud(o_out, pcd_o_full)
            o3d.io.write_point_cloud(k_out, pcd_k_full)
            print(f"Saved {o_out} and {k_out}")

if __name__ == "__main__":
    main()
