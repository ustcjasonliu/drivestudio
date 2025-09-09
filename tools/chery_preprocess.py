
import os
import sys
import glob
import json
import yaml
import numpy as np
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageDraw
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import math
import cv2
import hashlib
from multiprocessing import Pool, cpu_count
import open3d.t.io as tio

# Categories configuration for dynamic masks
# Movable categories include unknown as dynamic; static ones (e.g., cones, barriers) are ignored in masks
MOVABLE_CATEGORIES = {
    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'construction_vehicle',
    'tricycle', 'person', 'pickup_truck', 'unknown'
}
STATIC_CATEGORIES = {'traffic_cone', 'barrier'}

# No fine-grained per-category labels; we only output all/human/vehicle masks.
def calculate_horizontal_fov105_fx(fx, fy, w=3840, h=2160):
    fov105_radians = 105 / 180 * math.pi
    fov105_fx = w / 2 / math.tan(fov105_radians / 2)
    fov105_fy = (fov105_fx / fx) * fy
    return fov105_fx, fov105_fy
def undistort_cam0_to_fov105(K, W, H):
    K = np.float64(K)
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    fov105_fx, fov105_fy = calculate_horizontal_fov105_fx(fx, fy, W, H)
    fov105_intrinsic = np.eye(3)
    fov105_intrinsic[0][0] = fov105_fx
    fov105_intrinsic[1][1] = fov105_fy
    fov105_intrinsic[0][2] = cx
    fov105_intrinsic[1][2] = cy
    new_K = fov105_intrinsic
    return new_K
def undistort_img(image, distort_f, k_9nums, camera, interplt=cv2.INTER_LINEAR):
    assert k_9nums is not None
    assert distort_f is not None
    assert len(distort_f) in [4, 5, 8]  # fisheye 4, pinhole 5 or 8
    #assert camera in CAMERA_TYPE_MAPPING
    k_9nums = np.float64(k_9nums)
    if k_9nums.shape != (3, 3):
        K = k_9nums.reshape((3, 3))     # the intrinsic
    else:
        K = k_9nums
    distCoeffs = np.float64(distort_f)

    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError(f'Wrong value of image, type(image): {type(image)}')
    w, h = img.shape[1], img.shape[0]

    new_size = (w, h)
    new_K = K
    if camera == "camera0":
        new_K = undistort_cam0_to_fov105(K, w, h)
    # use the raw intrinsic in undistortion
    if len(distort_f) in [5, 8]:
        mapx, mapy = cv2.initUndistortRectifyMap(K, distCoeffs, None, new_K, new_size, cv2.CV_32FC1)
        img_undist = cv2.remap(img, mapx, mapy, interpolation=interplt)
    else:
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, distCoeffs, np.eye(3), new_K, new_size, cv2.CV_16SC2)
        img_undist = cv2.remap(img, mapx, mapy, interpolation=interplt, borderMode=cv2.BORDER_CONSTANT)
    
    return img_undist, new_K

def save_image(image: Image.Image, path: str) -> None:
    """保存图像文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def save_camera_intrinsics(intrinsics: np.ndarray, path: str) -> None:
    """保存相机内参"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, intrinsics)


def save_camera_extrinsics(extrinsics: np.ndarray, path: str) -> None:
    """保存相机外参"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, extrinsics)


def create_output_dirs(output_dir: str) -> None:
    """创建输出目录结构"""
    subdirs = [
    'images', 'lidar', 'ego_pose', 'extrinsics',
    'intrinsics', 'dynamic_masks', 'sky_masks', 'lidar_debug'
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


def interpolate_annotations(anno_dir: str, anno_files: List[str], target_ts: int) -> Optional[List[Dict]]:
    """
    根据目标时间戳对3D标注进行插值
    """
    ts_list = []
    anno_list = []
    for anno_file in anno_files:
        try:
            ts = int(anno_file.split('_')[1].split('.')[0])
            with open(os.path.join(anno_dir, anno_file), 'r') as f:
                anno = json.load(f)
                anno = anno.get('3d_city_object_detection_with_fish_eye_annotated_info', anno.get('3d_highway_object_detection_with_fish_eye_annotated_info', anno))

            if 'annotated_info' in anno and '3d_object_detection_info' in anno['annotated_info']:
                ts_list.append(ts)
                anno_list.append(anno['annotated_info']['3d_object_detection_info']['3d_object_detection_anns_info'])
        except Exception as e:
            print(f"Error reading annotation file {anno_file}: {e}")
            continue
    if not ts_list:
        return None
    ts_array = np.array(ts_list)
    idx = np.searchsorted(ts_array, target_ts)
    if idx == 0:
        return anno_list[0]
    elif idx >= len(ts_array):
        return anno_list[-1]
    t0, t1 = ts_array[idx-1], ts_array[idx]
    alpha = (target_ts - t0) / (t1 - t0)
    interpolated = []
    anno0, anno1 = anno_list[idx-1], anno_list[idx]
    track_ids0 = {obj['track_id']: obj for obj in anno0}
    track_ids1 = {obj['track_id']: obj for obj in anno1}
    common_tracks = set(track_ids0.keys()) & set(track_ids1.keys())
    for track_id in common_tracks:
        obj0 = track_ids0[track_id]
        obj1 = track_ids1[track_id]
        pos0 = np.array(obj0['obj_center_pos'])
        pos1 = np.array(obj1['obj_center_pos'])
        pos = pos0 * (1 - alpha) + pos1 * alpha
        rot0 = np.array(obj0['obj_rotation'])
        rot1 = np.array(obj1['obj_rotation'])
        rot = rot0 * (1 - alpha) + rot1 * alpha
        rot = rot / np.linalg.norm(rot)
        obj = obj0.copy()
        obj['obj_center_pos'] = pos.tolist()
        obj['obj_rotation'] = rot.tolist()
        interpolated.append(obj)
    return interpolated


def generate_dynamic_mask(
    image_size: Tuple[int, int],
    annotations: List[Dict],
    camera_id: int,
    chery_clip_dir: str,
    undistort_params: dict = None
) -> Image.Image:
    """
    根据3D标注生成动态目标掩码
    """
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    camera_mapping = get_camera_mapping()
    reverse_mapping = {v: k for k, v in camera_mapping.items() if '_camera' not in k}
    cam_name = reverse_mapping.get(camera_id)
    if not cam_name:
        raise ValueError(f"未知的相机ID: {camera_id}")
    # 使用undistort_params
    if undistort_params and camera_id in undistort_params:
        K_new = undistort_params[camera_id]['K_new']
        img_size = undistort_params[camera_id]['img_size']
        # map1/map2可用于后续mask去畸变（如需）
    else:
        print(f"使用相机 {cam_name} 的原始内参")
        # fallback: 仍然读取原始K
        intr_patterns = [
            os.path.join(chery_clip_dir, 'intrinsics', f'{cam_name}.yaml'),
            os.path.join(chery_clip_dir, 'intrinsics', f'{cam_name}_camera.yaml')
        ]
        intr_file = None
        for pattern in intr_patterns:
            if os.path.exists(pattern):
                intr_file = pattern
                break
        if not intr_file:
            raise FileNotFoundError(f"找不到相机 {cam_name} 的内参文件")
        with open(intr_file, 'r') as f:
            intrinsics = yaml.safe_load(f)
        K_new = np.array(intrinsics['K'])
        img_size = image_size
    cam_name_for_extr = cam_name.replace('_', '')
    extr_path = os.path.join(chery_clip_dir, 'extrinsics', 'lidar2camera', f'lidar2{cam_name_for_extr}.yaml')
    if not os.path.exists(extr_path):
        raise FileNotFoundError(f"找不到相机 {cam_name} 的外参文件：{extr_path}")
    T_lidar2cam = read_transform_yaml(extr_path)
    K = K_new
    for obj in annotations:
        center = np.array(obj['obj_center_pos'])
        size = np.array(obj['size'])
        quat = np.array(obj['obj_rotation'])
        R = Rotation.from_quat([ quat[0], quat[1], quat[2], quat[3]]).as_matrix()
        corners = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]) * (size / 2)
        corners = (R @ corners.T).T + center
        corners_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
        corners_cam = (T_lidar2cam @ corners_h.T).T[:, :3]
        if np.any(corners_cam[:, 2] < 0):
            continue
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u = cx + fx * corners_cam[:, 0] / corners_cam[:, 2]
        v = cy + fy * corners_cam[:, 1] / corners_cam[:, 2]
        points = [(int(u[i]), int(v[i])) for i in range(8)]
        hull = ConvexHull(points)
        hull_points = [points[i] for i in hull.vertices]
        draw.polygon(hull_points, fill=255)
    return mask

def fix_camera_pose(T_orig):
    """
    输入一个 4x4 相机外参矩阵（Z-forward），输出转换到 Waymo 坐标系（X-forward）的版本。
    """
    T_fix = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    R_orig = T_orig[:3, :3]
    t_orig = T_orig[:3, 3]
    R_new = T_fix @ R_orig
    t_new = T_fix @ t_orig
    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_new
    return np.linalg.inv(T_new)

def get_camera_mapping() -> Dict[str, int]:
    """获取相机名称到ID的映射关系"""
    mapping = {
        "front_wide": 0, "front_main": 1, "left_front": 2, "left_rear": 3,
        "right_front": 4, "right_rear": 5, "rear_main": 6,
        "fisheye_left": 7, "fisheye_rear": 8, "fisheye_front": 9, "fisheye_right": 10
    }
    variants = {k + "_camera": v for k, v in mapping.items()}
    mapping.update(variants)
    return mapping

def process_camera_params(chery_clip_dir: str, output_dir: str, undistort_fov: float = 60.0):
    """处理相机参数（内参和外参），并计算去畸变参数，返回每个相机的去畸变映射和新内参"""
    import cv2
    intrinsics_dir = os.path.join(chery_clip_dir, 'intrinsics')
    extrinsics_dir = os.path.join(chery_clip_dir, 'extrinsics')
    cam_names = [f for f in os.listdir(intrinsics_dir) if f.endswith('.yaml')]
    os.makedirs(os.path.join(output_dir, 'intrinsics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'extrinsics'), exist_ok=True)

    camera_mapping = get_camera_mapping()
    undistort_params = {}  # cam_id: dict with 'K_new', 'map1', 'map2', 'img_size'

    for cam_yaml in cam_names:
        cam_name = cam_yaml.replace('.yaml', '')
        try:
            cam_id = camera_mapping[cam_name]
        except KeyError:
            print(f"警告: 未知的相机名称 {cam_name}，跳过处理")
            continue

        intr_path = os.path.join(intrinsics_dir, cam_yaml)
        if os.path.exists(intr_path):
            with open(intr_path, 'r') as f:
                intr = yaml.safe_load(f)
            K = intr['K']  # [f_u, f_v, c_u, c_v]
            D = intr['D']  # [k1, k2, p1, p2] 或更多
            D_pad = D #+ [0.0] * (5 - len(D))
            

            # 计算去畸变参数和新内参（FOV=undistort_fov）
            # 假设图像分辨率由K的主点给出
            img_w = int(K[2] * 2)
            img_h = int(K[3] * 2)
            if 'width' in intr and 'height' in intr:
                img_w = intr['width']
                img_h = intr['height']
                # print(f"相机 {cam_name} 图像分辨率: {img_w}x{img_h}")
            else:
                print(f"警告: 相机 {cam_name} 没有指定图像分辨率，使用默认值 {img_w}x{img_h}")

            # 特殊处理：相机0裁剪掉底部1/3区域
            # if cam_id == 0:
            #     original_h = img_h
            #     img_h = int(img_h * 4 / 5)  # 裁剪掉底部1/3
            #     K[3] = K[3] * 4 / 5  # 调整主点 c_v，保持相对位置不变

            img_size = (img_w, img_h)
            img_size = (img_w, img_h)
            K_cv = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]])
            D_cv = np.array(D_pad[:4])

            # 新实现：直接保存K_new和D，供undistort_img使用
            if cam_id == 0:
                new_K = undistort_cam0_to_fov105(K_cv, img_w, img_h)
            else:
                new_K = K_cv.copy()
            undistort_params[cam_id] = {
                'K_new': new_K,
                'D': D_cv,
                'img_size': img_size,
                'cam_name': cam_name
            }
            intr_array = np.array([new_K[0, 0], new_K[1, 1], new_K[0, 2], new_K[1, 2], 0, 0, 0, 0, 0])
            save_camera_intrinsics(
                intr_array,
                os.path.join(output_dir, 'intrinsics', f'{cam_id}.txt')
            )
        cam_name_for_extr = cam_name.replace('_', '').replace('camera', '')
        extr_path = os.path.join(extrinsics_dir, 'lidar2camera', f'lidar2{cam_name_for_extr}.yaml')
        if os.path.exists(extr_path):
            extr_array = read_transform_yaml(extr_path)
            extr_array = fix_camera_pose(extr_array)  # 转换到 Waymo 坐标系
            save_camera_extrinsics(
                extr_array,
                os.path.join(output_dir, 'extrinsics', f'{cam_id}.txt')
            )
        else:
            print(f"警告: 相机 {cam_id} 的外参文件不存在: {extr_path}")

    return undistort_params

def process_images(sample_path: str, chery_clip_dir: str, output_dir: str, timestep: int, undistort_params=None) -> None:
    """
    处理图像相关数据，包括图片、动态目标掩码和天空掩码
    
    Args:
        sample_path: 当前帧的路径
        chery_clip_dir: 数据根目录，用于查找标注文件
        output_dir: 输出目录
        timestep: 当前帧的序号
    """
    # 获取相机映射关系
    camera_mapping = get_camera_mapping()
    reverse_mapping = {v: k for k, v in camera_mapping.items() if '_camera' not in k}

    # 1. 处理图像
    import cv2
    for cam_id in range(11):
        img_patterns = [
            os.path.join(sample_path, f'camera{cam_id}_*.jpg'),
            os.path.join(sample_path, f'{reverse_mapping[cam_id]}_*.jpg'),
            os.path.join(sample_path, f'{reverse_mapping[cam_id]}_camera_*.jpg'),
        ]
        img_files = []
        for pattern in img_patterns:
            img_files.extend(glob.glob(pattern))

        img_files = [f for f in img_files if 'undist' not in f]  # 排除空字符串
        if not img_files:
            print(f"找不到相机 {cam_id} ({reverse_mapping[cam_id]}) 的图像文件")
            continue

        img_path = img_files[0]
        image = Image.open(img_path)

        # 新去畸变实现
        if undistort_params and cam_id in undistort_params:
            cam_param = undistort_params[cam_id]
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cam_name = cam_param.get('cam_name', f'camera{cam_id}')
            img_undist, _ = undistort_img(img_cv, cam_param['D'], cam_param['K_new'], cam_name)
            image = Image.fromarray(cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB))
        # 裁剪底部 1/3，仅限相机0
        # if cam_id == 0:
        #     w, h = image.size
        #     crop_h = int(h * 4 / 5)
        #     image = image.crop((0, 0, w, crop_h))

        save_image(
            image,
            os.path.join(output_dir, 'images', f'{timestep:03d}_{cam_id}.jpg')
        )

    # 1.2 生成 dynamic masks（all/human/vehicle）
        try:
            img_timestamp = int(os.path.basename(img_path).split('_')[1].split('.')[0])
            anno_root = os.path.join(chery_clip_dir, 'annotation')
            anno_dir = glob.glob(os.path.join(anno_root, '3d_*_object_detection_with_fish_eye'))[0]
            # anno_dir = os.path.join(chery_clip_dir, 'annotation', '3d_city_object_detection_with_fish_eye')
            anno_files = sorted([f for f in os.listdir(anno_dir) if f.startswith('sample_') and f.endswith('.json')])
            annotations = interpolate_annotations(anno_dir, anno_files, img_timestamp)
            if annotations:
                # Normalize categories
                anns_norm = []
                for a in annotations:
                    a = dict(a)
                    if 'category' in a and isinstance(a['category'], str):
                        a['category'] = a['category'].lower()
                    anns_norm.append(a)

                # Filters
                anns_all = [a for a in anns_norm if a.get('category', '') in MOVABLE_CATEGORIES]
                anns_human = [a for a in anns_norm if a.get('category', '') == 'person']
                anns_vehicle = [a for a in anns_norm if (a.get('category', '') in MOVABLE_CATEGORIES and a.get('category', '') != 'person')]

                # All movable (binary)
                if anns_all:
                    dynamic_mask_all = generate_dynamic_mask(
                        image.size,
                        anns_all,
                        cam_id,
                        chery_clip_dir,
                        undistort_params=undistort_params,
                    )
                else:
                    dynamic_mask_all = Image.new('L', image.size, 0)
                save_image(
                    dynamic_mask_all,
                    os.path.join(output_dir, 'dynamic_masks/all', f'{timestep:03d}_{cam_id}.png')
                )

                # Human-only (binary)
                if anns_human:
                    dynamic_mask_human = generate_dynamic_mask(
                        image.size,
                        anns_human,
                        cam_id,
                        chery_clip_dir,
                        undistort_params=undistort_params,
                    )
                else:
                    dynamic_mask_human = Image.new('L', image.size, 0)
                save_image(
                    dynamic_mask_human,
                    os.path.join(output_dir, 'dynamic_masks/human', f'{timestep:03d}_{cam_id}.png')
                )

                # Vehicle-only (binary)
                if anns_vehicle:
                    dynamic_mask_vehicle = generate_dynamic_mask(
                        image.size,
                        anns_vehicle,
                        cam_id,
                        chery_clip_dir,
                        undistort_params=undistort_params,
                    )
                else:
                    dynamic_mask_vehicle = Image.new('L', image.size, 0)
                save_image(
                    dynamic_mask_vehicle,
                    os.path.join(output_dir, 'dynamic_masks/vehicle', f'{timestep:03d}_{cam_id}.png')
                )

                # 不输出细分类（仅保留 all / human / vehicle）
            else:
                # 没有该帧标注时，也输出默认空掩码，保持文件完备
                save_image(
                    Image.new('L', image.size, 0),
                    os.path.join(output_dir, 'dynamic_masks/all', f'{timestep:03d}_{cam_id}.png')
                )
                save_image(
                    Image.new('L', image.size, 0),
                    os.path.join(output_dir, 'dynamic_masks/human', f'{timestep:03d}_{cam_id}.png')
                )
                save_image(
                    Image.new('L', image.size, 0),
                    os.path.join(output_dir, 'dynamic_masks/vehicle', f'{timestep:03d}_{cam_id}.png')
                )
        except Exception as e:
            print(f"Error generating dynamic mask for {img_path}: {e}")

        sky_path = os.path.join(chery_clip_dir, 'annotation', os.path.basename(img_path).replace('.jpg', '_sky.png'))
        if os.path.exists(sky_path):
            save_image(
                Image.open(sky_path),
                os.path.join(output_dir, 'sky_masks', f'{timestep:03d}_{cam_id}.png')
            )
        else:
            pass
def o3d_to_waymo_bin(o3d_pcd, out_bin_path):
    """
    把 tio.read_point_cloud 的结果转成 Waymo 5 通道 bin
    """
    # 1. 取出 tensor 并转 numpy
    pos  = o3d_pcd.point['positions'].numpy().astype(np.float32)
    intensity = o3d_pcd.point['intensity'].numpy().astype(np.float32)

    # 2. 补齐 elongation（Waymo 要求 5 通道）
    elongation = np.zeros((pos.shape[0], 1), dtype=np.float32)

    # 3. 拼成 [x,y,z,i,e]
    waymo_pc = np.hstack([pos, intensity, elongation])  # (N,5)

    # 4. 写二进制
    waymo_pc.tofile(out_bin_path)
    print(f'Waymo bin 已生成：{out_bin_path}  点云数：{waymo_pc.shape[0]}')


# def interpolate_pointcloud(positions, timestamps, ego_positions, lidar_to_ego_transmat):

def process_lidar(sample_path: str, chery_clip_dir: str,
                 ego_positions, output_dir: str, 
                 timestep: int) -> None:
    """处理激光雷达数据"""
    lidar_data = []
    total_points = 0
    
    for lidar_idx in range(4):
        # print(f"\n处理激光雷达 {lidar_idx}")
        lidar_files = glob.glob(os.path.join(sample_path, f'lidar{lidar_idx}_*.pcd'))
        
        if not lidar_files:
            print(f"未找到激光雷达 {lidar_idx} 的数据文件")
            continue
            
        # print(f"找到文件: {lidar_files[0]}")
        
             
        pcd = tio.read_point_cloud(lidar_files[0])
        #print(pcd.point)
        # print(f"点云数量: {len(pcd.points)}")
        if len(pcd.point.positions) == 0:
            print(f"激光雷达 {lidar_idx} 没有有效的点云数据")
            continue

        debug_dir = os.path.join(output_dir, 'lidar_debug')
        pcd_path = os.path.join(debug_dir, f'{timestep:03d}.pcd')
        success = tio.write_point_cloud(pcd_path, pcd)
        trans_mat = get_lidar_transform(chery_clip_dir, lidar_idx)
        print("trans_mat ", trans_mat)
        posistions = pcd.point.positions.numpy().astype(np.float32)
        timetamps = pcd.point.timestamp.numpy().astype(np.float32)
        waymo_data = convert_to_waymo_format(posistions, trans_mat, lidar_idx)

            # print(f"读取到 {pts.shape[0]} 个点")
            
            # 获取转换矩阵
            #
            # print(f"转换矩阵:\n{trans_mat}")
            
            # 转换点云并构造Waymo格式
            
            # print(f"转换后数据: shape={waymo_data.shape}")
            
            #lidar_data.append(waymo_data)
            #total_points += waymo_data.shape[0]
            
    
    # if lidar_data:
    #     # print(f"\n合并所有激光雷达数据，总计 {total_points} 个点")
    #     try:
    #         combined_data = np.concatenate(lidar_data, axis=0)
    #         save_path = os.path.join(output_dir, 'lidar', f'{timestep:03d}.bin')
    #         save_lidar_data(combined_data, save_path)
    #         # 同步输出调试用 PLY（仅 xyz ）
    #         try:
    #             debug_dir = os.path.join(output_dir, 'lidar_debug')
    #             os.makedirs(debug_dir, exist_ok=True)
    #             ply_path = os.path.join(debug_dir, f'{timestep:03d}.ply')
    #             # combined_data 列 3:6 为点坐标 (x,y,z)
    #             points_xyz = combined_data[:, 3:6]
    #             save_point_cloud_ply(points_xyz, ply_path)
    #         except Exception as e:
    #             print(f"保存调试PLY失败: {e}")
    #         # print(f"保存数据到: {save_path}")
    #     except Exception as e:
    #         print(f"保存合并数据时出错: {str(e)}")
    # else:
    #     print("警告：没有有效的激光雷达数据可保存")

def read_transform_yaml(yaml_path: str) -> np.ndarray:
    """读取并解析YAML格式的转换矩阵
    
    Args:
        yaml_path: YAML文件路径
        
    Returns:
        4x4 转换矩阵，如果文件不存在则返回单位矩阵
    """
    if not os.path.exists(yaml_path):
        return np.eye(4)
        
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # 处理不同的数据格式
    if 'transform' in data:
        if isinstance(data['transform'], dict) and 'rotation' in data['transform'] and 'translation' in data['transform']:
            # 处理包含旋转和平移的字典格式
            rot = data['transform']['rotation']
            trans = data['transform']['translation']
            
            # 从四元数创建旋转矩阵
            quat = np.array([rot['x'], rot['y'], rot['z'], rot['w']])
            # 这里主要影响副lidar的数据
            R = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_matrix()
            
            # 创建平移向量
            t = np.array([trans['x'], trans['y'], trans['z']])
            
            # 构建4x4变换矩阵
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t
        else:
            transform = np.array(data['transform'])
    elif 'data' in data:
        transform = np.array(data['data']).reshape(4, 4)
    else:
        # 如果数据直接是矩阵格式
        transform = np.array(data).reshape(4, 4)
        
    return transform

def get_lidar_transform(chery_clip_dir: str, lidar_idx: int) -> np.ndarray:
    """获取LiDAR到主LiDAR的转换矩阵"""
    # 主LiDAR返回单位矩阵
    if lidar_idx == 0:
        return np.eye(4)
    
    # 根据索引获取对应的LiDAR名称
    lidar_names = {
        1: 'left',
        2: 'front',
        3: 'right'
    }
    
    if lidar_idx not in lidar_names:
        print(f"警告: 未知的激光雷达索引 {lidar_idx}，使用单位矩阵")
        return np.eye(4)
    
    extr_path = os.path.join(chery_clip_dir, 'extrinsics', 'lidar2lidar', 
                            f'{lidar_names[lidar_idx]}2mainlidar.yaml')
    return read_transform_yaml(extr_path)

def convert_to_waymo_format(pts: np.ndarray, trans_mat: np.ndarray, 
                          lidar_idx: int) -> np.ndarray:
    """将点云转换为Waymo格式"""
    if pts.shape[0] == 0:
        # 处理空点云的情况
        return np.zeros((0, 10), dtype=np.float32)
    
    # 转换到主LiDAR坐标系
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts_trans = (trans_mat @ pts_h.T).T[:, :3]
    
    # 构造Waymo格式数组
    N = pts_trans.shape[0]
    arr = np.zeros((N, 14), dtype=np.float32)
    # origins: LiDAR安装位置
    lidar_origin = trans_mat[:3, 3]
    arr[:, 0:3] = np.tile(lidar_origin, (N, 1))
    arr[:, 3:6] = pts_trans  # points
    arr[:, 6:10] = 0  # flows
    arr[:, 10] = 0  # ground label
    arr[:, 11] = 0  # intensity
    arr[:, 12] = 0  # elongation
    arr[:, 13] = lidar_idx  # laser_id

    # arr[:, 0:3] = pts
    # arr[:, 3:6] = pts_trans  # points
    # arr[:, 6:10] = 0  # flows
    # arr[:, 6] = 0  # ground label
    # arr[:, 7] = 0  # intensity
    # arr[:, 8] = 0  # elongation
    # arr[:, 9] = lidar_idx  # laser_id
    return arr

def load_poses(chery_clip_dir: str):
    pose_path = os.path.join(chery_clip_dir, 'localization.json')
    if not os.path.exists(pose_path):
        return None
        
    with open(pose_path, 'r') as f:
        poses = json.load(f)
    # import pdb; pdb.set_trace() 
    # 确保poses是列表
    if not isinstance(poses, list):
        if 'poses' in poses:
            poses = poses['poses']
        else:
            print(f"Warning: Unexpected format in {pose_path}")
            return None
    
    # 获取时间戳对应的位姿（dict格式）
    ts_pose = [(p["timestamp"], p["pose"]) for p in poses if "timestamp" in p and "pose" in p]
    return ts_pose

def process_ego_pose(ts_pose, sample_name: str, output_dir: str, 
                    timestep: int) -> None:
   
    # 获取时间戳对应的位姿（dict格式）
    if not ts_pose:
        print(f"No valid pose in {pose_path}")
        return
    # 处理时间戳和插值
    pose_dict = interpolate_pose(ts_pose, sample_name)
    pose_mat = None
    
    if pose_dict is not None:
        # 转换为Waymo格式4x4矩阵
        pose_dict['position']['z'] += 1.801
        pose_mat = convert_pose_dict_to_matrix(pose_dict)
        save_ego_pose(pose_mat, os.path.join(output_dir, 'ego_pose', f'{timestep:03d}.txt'))
    

def interpolate_pose_float(ts_pose: List[Tuple[int, np.ndarray]], 
                    sample_ts: float) -> Optional[np.ndarray]:
    ts_list = [t for t, _ in ts_pose]
    if sample_ts in ts_list:
        return ts_pose[ts_list.index(sample_ts)][1]
    # 找到最近的两个时间戳做线性插值（仅插值position和orientation）
    ts_sorted = sorted(ts_pose, key=lambda x: x[0])
    prev = next = None
    for t, p in ts_sorted:
        if t < sample_ts:
            prev = (t, p)
        elif t > sample_ts and next is None:
            next = (t, p)
            break
    if prev and next:
        alpha = (sample_ts - prev[0]) / (next[0] - prev[0])
        # 插值position
        pos0 = np.array([prev[1]['position']['x'], prev[1]['position']['y'], prev[1]['position']['z']])
        pos1 = np.array([next[1]['position']['x'], next[1]['position']['y'], next[1]['position']['z']])
        pos = pos0 * (1 - alpha) + pos1 * alpha
        # 插值四元数（线性插值，或可用slerp）
        quat0 = np.array([prev[1]['orientation']['qx'], prev[1]['orientation']['qy'], prev[1]['orientation']['qz'], prev[1]['orientation']['qw']])
        quat1 = np.array([next[1]['orientation']['qx'], next[1]['orientation']['qy'], next[1]['orientation']['qz'], next[1]['orientation']['qw']])
        quat = quat0 * (1 - alpha) + quat1 * alpha
        quat = quat / np.linalg.norm(quat)
        # 构造插值后的pose dict
        pose_interp = prev[1].copy()
        pose_interp['position'] = {'x': pos[0], 'y': pos[1], 'z': pos[2]}
        pose_interp['orientation'] = {'qx': quat[0], 'qy': quat[1], 'qz': quat[2], 'qw': quat[3]}
        return pose_interp
    return prev[1] if prev else next[1] if next else ts_sorted[0][1]


def interpolate_pose(ts_pose: List[Tuple[int, np.ndarray]], 
                    sample_name: str) -> Optional[np.ndarray]:
    """根据时间戳插值计算位姿"""
    try:
        sample_ts = int(sample_name.split('_')[1])
        sample_ts = sample_ts / 1000000.0  # 转换为秒
        # print(f"Processing timestamp: {sample_ts}")
    except (IndexError, ValueError):
        print(f"Warning: 无法解析时间戳 {sample_name}，使用第一个位姿")
        return ts_pose[0][1]  # 无法解析时间戳，返回第一个位姿
    
    return interpolate_pose_float(ts_pose, sample_ts)


import numpy as np
from scipy.spatial.transform import Rotation

def convert_pose_dict_to_matrix(pose_dict):
    """
    将 UTM 坐标系下的 pose dict 转换为 Waymo 坐标系下的 4x4 变换矩阵。
    使用坐标系共轭变换方式，保证方向保持一致。
    """

    # 1. 从 pose_dict 构造原始的 4x4 变换矩阵（UTM 坐标系）
    pos = pose_dict['position']
    x, y, z = pos['x'], pos['y'], pos['z']
    quat = pose_dict['orientation']
    qx, qy, qz, qw = quat['qx'], quat['qy'], quat['qz'], quat['qw']
    rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    
    T_utm = np.eye(4)
    T_utm[:3, :3] = rot
    T_utm[:3, 3] = [x, y, z]

    # 2. 定义坐标轴变换矩阵 R（将 ENU → Waymo）
    # ENU: X=East, Y=North, Z=Up
    # Waymo: X=Forward, Y=Left, Z=Up
    # 所以我们需要将 ENU 坐标系旋转到 Waymo 坐标系：
    # ENU (E, N, U) → Waymo (F, L, U) ≈ Waymo (N, -E, U)
    R_axis = np.array([
        [ 0,  1, 0],
        [-1,  0, 0],
        [ 0,  0, 1]
    ])  # 3x3

    R_transform = np.eye(4)
    R_transform[:3, :3] = R_axis

    # 3. 共轭变换
    T_waymo = R_transform @ T_utm @ np.linalg.inv(R_transform)
    # T_waymo = T_utm
    return T_waymo


def _process_frame_worker(args):
    """Worker to process a single frame: images, lidar, ego pose."""
    timestep, sample_name, chery_clip_dir, output_dir, undistort_params = args
    sample_path = os.path.join(chery_clip_dir, sample_name)
    # Heavy I/O & CPU tasks (safe to run in parallel as filenames are unique per timestep)
    #process_images(sample_path, chery_clip_dir, output_dir, timestep, undistort_params=undistort_params)
    ego_poses = load_poses(chery_clip_dir) 
    print("ego poses shape ", np.shape(ego_poses))
    process_ego_pose(ego_poses, sample_name, output_dir, timestep)
   
    process_lidar(sample_path, chery_clip_dir, ego_poses, output_dir, timestep)

    return timestep


def preprocess_chery_clip(chery_clip_dir: str, output_dir: str) -> None:
    """
    预处理Chery数据集中的单个clip
    
    Args:
        chery_clip_dir: Chery数据clip的根目录
        output_dir: 输出目录路径
    """
    create_output_dirs(output_dir)
    # 2. 处理相机参数（内参和外参），并获取去畸变参数
    undistort_params = process_camera_params(chery_clip_dir, output_dir, undistort_fov=71.0)
    # 3. 获取所有sample并按时间戳排序
    sample_dirs = sorted([d for d in os.listdir(chery_clip_dir) if d.startswith('sample_')])
    # 预先准备标注目录和文件列表（供插值与统计使用）
    anno_root = os.path.join(chery_clip_dir, 'annotation')
    anno_dirs = glob.glob(os.path.join(anno_root, '3d_*_object_detection_with_fish_eye'))
    anno_dir = anno_dirs[0] if anno_dirs else None
    anno_files = []
    if anno_dir and os.path.isdir(anno_dir):
        anno_files = sorted([f for f in os.listdir(anno_dir) if f.startswith('sample_') and f.endswith('.json')])

    # 为 instances 输出做收集
    frame_instances: Dict[str, List[int]] = {}
    instances_info: Dict[str, Dict] = {}
    clip_ns = os.path.basename(chery_clip_dir.rstrip('/'))

    # 4.1 重任务并行处理（图像/点云/位姿）
    tasks = [(i, name, chery_clip_dir, output_dir, undistort_params) for i, name in enumerate(sample_dirs)]
    with Pool(min(cpu_count(), 8)) as pool:
        for _ in tqdm(pool.imap_unordered(_process_frame_worker, tasks), total=len(tasks), desc="Frames", unit="frame"):
            pass

    # 4.1.1 统一原点：将第一帧 ego pose 作为世界坐标系原点 (保存为相对矩阵)
    try:
        ego_pose_dir = os.path.join(output_dir, 'ego_pose')
        pose_files = sorted([f for f in os.listdir(ego_pose_dir) if f.endswith('.txt')])
        if pose_files:
            T0 = np.loadtxt(os.path.join(ego_pose_dir, pose_files[0]))
            if T0.shape == (4, 4):
                T0_inv = np.linalg.inv(T0)
                for pf in pose_files:
                    full_path = os.path.join(ego_pose_dir, pf)
                    try:
                        Ti = np.loadtxt(full_path)
                        if Ti.shape == (4, 4):
                            Ti_rel = T0_inv @ Ti
                            np.savetxt(full_path, Ti_rel)
                    except Exception as e:
                        print(f"规范化 ego pose 时读取 {pf} 失败: {e}")
    except Exception as e:
        print(f"规范化第一帧原点失败: {e}")

    # 4.2 串行收集实例信息（确保一致性，避免并发写共享状态）
    for timestep, sample_name in enumerate(sample_dirs):
        try:
            if anno_dir and anno_files:
                ts = int(sample_name.split('_')[1])
                annos = interpolate_annotations(anno_dir, anno_files, ts)
                if annos:
                    tids = []
                    for obj in annos:
                        cat_raw = str(obj.get('category', '')).lower()
                        if cat_raw in STATIC_CATEGORIES:
                            continue
                        mapped_cls = 'Pedestrian' if cat_raw == 'person' else 'Vehicle'

                        if 'track_id' not in obj:
                            continue
                        try:
                            track_id = int(obj['track_id']) - 1
                        except Exception:
                            track_id = obj['track_id']
                        tids.append(track_id)

                        key = str(track_id)
                        if key not in instances_info:
                            uid = hashlib.md5(f"{clip_ns}:{track_id}".encode('utf-8')).hexdigest()
                            instances_info[key] = {
                                'id': uid,
                                'class_name': mapped_cls,
                                'frame_annotations': {
                                    'frame_idx': [],
                                    'obj_to_world': [],
                                    'box_size': []
                                }
                            }

                        center = obj.get('obj_center_pos', [0, 0, 0])
                        quat = obj.get('obj_rotation', [0, 0, 0, 1])
                        size = obj.get('size', [0, 0, 0])

                        R_obj = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_matrix()
                        t_obj = np.array(center, dtype=float)

                        T_obj_in_ego = np.eye(4)
                        T_obj_in_ego[:3, :3] = R_obj
                        T_obj_in_ego[:3, 3] = t_obj

                        # 读取已保存的本帧ego位姿（world_from_ego）并映射到world
                        # ego 已在前面被规范化为相对第一帧的坐标系
                        T_obj_to_world = T_obj_in_ego
                        try:
                            pose_path_txt = os.path.join(output_dir, 'ego_pose', f'{timestep:03d}.txt')
                            if os.path.exists(pose_path_txt):
                                pose_mat = np.loadtxt(pose_path_txt)
                                if pose_mat.shape == (4, 4):
                                    # world_from_ego (relative) * obj_in_ego
                                    T_obj_to_world = pose_mat @ T_obj_in_ego
                        except Exception as e:
                            pass

                        inst = instances_info[key]['frame_annotations']
                        inst['frame_idx'].append(timestep)
                        inst['obj_to_world'].append(T_obj_to_world.tolist())
                        inst['box_size'].append(size)

                    frame_instances[str(timestep)] = tids
        except Exception as e:
            print(f"Error collecting instances for {sample_name}: {e}")

    # 写出 instances 结果（对实例进行连续重映射，确保两份 JSON 一致，且从 0 开始）
    try:
        inst_dir = os.path.join(output_dir, 'instances')
        os.makedirs(inst_dir, exist_ok=True)

        # 1) 构建 old_id -> new_id 的连续映射
        if instances_info:
            ordered_old_ids = sorted(int(k) for k in instances_info.keys())
            id_map = {old_id: new_id for new_id, old_id in enumerate(ordered_old_ids)}

            # 2) 重映射 instances_info 到新 ID（键与内部 'id' 字段）
            remapped_instances_info: Dict[str, Dict] = {}
            for old_id in ordered_old_ids:
                old_key = str(old_id)
                new_id = id_map[old_id]
                item = dict(instances_info[old_key])
                # 使用新的连续 ID，保持简洁
                item['id'] = new_id
                remapped_instances_info[str(new_id)] = item

            # 3) 重映射 frame_instances 中的 track 列表
            remapped_frame_instances: Dict[str, List[int]] = {}
            for frame_key, tids in frame_instances.items():
                new_tids: List[int] = []
                for tid in tids:
                    try:
                        new_tids.append(id_map[int(tid)])
                    except Exception:
                        # 若某些 tid 不在映射中（理论上不会发生，因为已过滤），则忽略
                        continue
                remapped_frame_instances[frame_key] = new_tids
        else:
            remapped_instances_info = instances_info
            remapped_frame_instances = frame_instances

        # 4) 写文件
        with open(os.path.join(inst_dir, 'frame_instances.json'), 'w') as f:
            json.dump(remapped_frame_instances, f)
        with open(os.path.join(inst_dir, 'instances_info.json'), 'w') as f:
            json.dump(remapped_instances_info, f)
    except Exception as e:
        print(f"Error writing instances outputs: {e}")

def read_pcd_xyz(pcd_path):
    """
    只读取pcd文件的xyz坐标，返回(N,3)数组。
    支持ASCII和二进制格式的PCD文件。
    """
    try:
        # 读取文件头部
        header_data = {}
        header_lines = []
        with open(pcd_path, 'rb') as f:
            while True:
                line = f.readline()
                try:
                    decoded_line = line.decode('ascii', errors='ignore').strip()
                    header_lines.append(decoded_line)
                except Exception as e:
                    print(f"警告：解码行时出错: {e}")
                    continue

                if not decoded_line:
                    continue
                    
                if decoded_line.startswith('DATA'):
                    header_data['data_format'] = decoded_line.split()[1]
                    break
                    
                if decoded_line.startswith('#'):
                    continue
                    
                tokens = decoded_line.split()
                if len(tokens) >= 2:
                    header_data[tokens[0]] = tokens[1:]
                    

        # 获取点云信息
        try:
            width = int(header_data.get('WIDTH', ['0'])[0])
            height = int(header_data.get('HEIGHT', ['0'])[0])
            points_num = width * height
            
            fields = header_data.get('FIELDS', ['x', 'y', 'z', ])
            sizes = [int(s) for s in header_data.get('SIZE', ['4'] * len(fields))]
            types = header_data.get('TYPE', ['F'] * len(fields))
            
            # 验证必要的字段是否存在
            if 'x' not in fields or 'y' not in fields or 'z' not in fields:
                print(f"错误：缺少必要的坐标字段")
                return np.zeros((0, 3), dtype=np.float32)
        except Exception as e:
            print(f"错误：解析点云信息时出错: {e}")
            return np.zeros((0, 3), dtype=np.float32)
        
        # 计算字段偏移
        field_offsets = {}
        offset = 0
        for field, size in zip(fields, sizes):
            field_offsets[field] = offset
            offset += size
        point_size = offset
        
        # 检查是否为二进制格式
        is_binary = header_data['data_format'].lower() == 'binary'
        
        if is_binary:
            # 二进制格式
            import struct
            # 计算头部大小
            with open(pcd_path, 'rb') as f:
                header_size = 0
                while True:
                    line = f.readline()
                    header_size += len(line)
                    if b'DATA' in line:
                        break
                
            # 读取点云数据
            pts = []
            with open(pcd_path, 'rb') as f:
                # 跳过头部
                f.seek(header_size)
                
                # 准备解包格式
                fmt = ''
                for t in types[:3]:  # 只读取xyz
                    if t == 'F': fmt += 'f'
                    elif t == 'D': fmt += 'd'
                    elif t == 'I': fmt += 'i'
                    elif t == 'U': fmt += 'I'
                
                # 读取数据
                for _ in range(points_num):
                    try:
                        data = f.read(point_size)
                        if len(data) < point_size:
                            break
                        
                        # 只取xyz字段
                        x = struct.unpack('f', data[field_offsets['x']:field_offsets['x']+4])[0]
                        y = struct.unpack('f', data[field_offsets['y']:field_offsets['y']+4])[0]
                        z = struct.unpack('f', data[field_offsets['z']:field_offsets['z']+4])[0]
                        pts.append([x, y, z])
                    except struct.error:
                        break
        else:
            # ASCII格式
            pts = []
            with open(pcd_path, 'r') as f:
                # 跳过头部
                for line in f:
                    if line.startswith('DATA'):
                        break
                # 读取点数据
                for line in f:
                    try:
                        vals = line.strip().split()
                        if len(vals) >= 3:
                            pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
                    except (ValueError, IndexError):
                        continue
        
        points = np.array(pts, dtype=np.float32)
        if len(points) == 0:
            print(f"警告：PCD文件 {pcd_path} 没有读取到任何点")
            return np.zeros((0, 3), dtype=np.float32)
            
        # 检查数据有效性
        if np.isnan(points).any() or np.isinf(points).any():
            print(f"警告：点云数据中包含 NaN 或 Inf 值")
            # 移除无效点
            valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
            points = points[valid_mask]
            # print(f"清理后的点云数据: shape={points.shape}")
            
        return points
        
    except Exception as e:
        print(f"错误：读取PCD文件 {pcd_path} 失败: {str(e)}")
        return np.zeros((0, 3), dtype=np.float32)



# def process_chery_camera_params(intrinsics_dir, extrinsics_dir, output_dir):
#     """
#     处理 chery 相机内外参，统一保存为 txt 格式
#     """
#     # 相机名列表（可根据实际文件夹内容自动获取）
#     cam_names = [f for f in os.listdir(intrinsics_dir) if f.endswith('.yaml')]
#     os.makedirs(f"{output_dir}/intrinsics", exist_ok=True)
#     os.makedirs(f"{output_dir}/extrinsics", exist_ok=True)

#     for cam_yaml in cam_names:
#         cam_name = cam_yaml.replace('.yaml', '')
#         # 1. 处理内参
#         with open(os.path.join(intrinsics_dir, cam_yaml), 'r') as f:
#             intr = yaml.safe_load(f)
#         # K: [f_u, f_v, c_u, c_v]
#         K = intr['K']
#         # D: [k1, k2, p1, p2] 或更多
#         D = intr['D']
#         # 拼接为 [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
#         # 若 D 长度不足补零
#         D_pad = D + [0.0] * (5 - len(D))
#         intr_array = np.array(K + D_pad[:5])
#         np.savetxt(f"{output_dir}/intrinsics/{cam_name}.txt", intr_array)

#         # 2. 处理外参
#         extr_path = os.path.join(extrinsics_dir, 'lidar2camera', f'lidar2{cam_name}.yaml')
#         if os.path.exists(extr_path):
#             with open(extr_path, 'r') as f:
#                 extr = yaml.safe_load(f)
#             # 假设 extr['transform'] 为 4x4 矩阵
#             if 'transform' in extr:
#                 extr_array = np.array(extr['transform'])
#             else:
#                 # 兼容部分 yaml 格式为 flat list
#                 extr_array = np.array(extr['data']).reshape(4, 4)
#             np.savetxt(f"{output_dir}/extrinsics/{cam_name}.txt", extr_array)
#         else:
#             print(f"Extrinsics not found for {cam_name}")


# IO 操作相关函数
def save_lidar_data(data: np.ndarray, save_path: str) -> None:
    """保存LiDAR数据为二进制文件"""
    data.tofile(save_path)

def save_ego_pose(pose: np.ndarray, save_path: str) -> None:
    """保存ego位姿"""
    np.savetxt(save_path, pose)

def save_point_cloud_ply(points: np.ndarray, save_path: str) -> None:
    """保存点云为简单ASCII PLY文件 (N x 3)."""
    if points.size == 0:
        # 仍然写一个空的 PLY 头，方便调试
        header = [
            'ply',
            'format ascii 1.0',
            'element vertex 0',
            'property float x',
            'property float y',
            'property float z',
            'end_header'\
        ]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('\n'.join(header))
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        # 写入点
        for p in points:
            f.write(f'{p[0]} {p[1]} {p[2]}\n')


if __name__ == "__main__":
    # from multiprocessing import Pool, cpu_count
    # chery_clip_root = "/home/yuhan/yuhan/chery/A车/高速"
    # output_root = "/home/yuhan/yuhan/chery_gs6"
    # clips = [clip_name for clip_name in os.listdir(chery_clip_root) if clip_name.startswith('clip_')]

    # def process_one_clip(clip_name):
    #     chery_clip_dir = os.path.join(chery_clip_root, clip_name)
    #     output_dir = os.path.join(output_root, clip_name)
    #     print(f"Processing clip: {chery_clip_dir} -> {output_dir}")
    #     preprocess_chery_clip(chery_clip_dir, output_dir)
    #     print(f"Finished processing clip: {chery_clip_dir} -> {output_dir}")

    # with Pool(min(cpu_count(), 16)) as pool:
    #     pool.map(process_one_clip, clips)
    # print("所有clips处理完成！")
    # 使用示例
    chery_clip_dir = "/home/yuchen/yuchen/repos/cherry-dataset/A车/城市/clip_1717058483101"
    # chery_clip_dir = "/home/yuhan/yuhan/chery/B车/城市/clip_1744499330800"
    output_dir = "./outputs_mirror_fixed/014"
    # 预处理单个clip
    preprocess_chery_clip(chery_clip_dir, output_dir)
