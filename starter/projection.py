"""LiDAR -> camera projection, overlay và perturb calibration.

Hai hàm có `TODO(CP2)` là phần học viên phải tự cài đặt (bắt buộc cho topic A, C, F;
khuyến khích cho mọi topic vì đây là bài test calibration rẻ nhất).

Chạy thử sau khi cài đặt xong (cùng một code chạy được cho cả KITTI và nuScenes):
    python -m starter.projection --data-root data/synthetic --frame 000000
    python -m starter.projection --data-root data/synthetic --frame 000000 --yaw-deg 1.0
    python -m starter.projection --data-root data/kitti_mini --frame 000011
    python -m starter.projection --data-root data/nuscenes_mini_subset --frame scene-0103_010
    python -m starter.projection --data-root data/nuscenes_mini_subset --frame scene-0103_010 --ignore-ego-motion
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import numpy as np

from starter.datasets import dataset_type, load_frame
from starter.kitti_io import KittiCalib, KittiObject


def velo_to_cam(points_xyz: np.ndarray, calib: KittiCalib) -> np.ndarray:
    """Đưa điểm (N, 3) từ velodyne frame sang rectified camera frame (N, 3).

    TODO(CP2):
      1. Chuyển sang toạ độ đồng nhất (N, 4).
      2. Nhân với calib.T_cam_velo (4x4). Chú ý chiều nhân và transpose.
      3. Trả về 3 cột đầu.
    Tự kiểm: một điểm velodyne (10, 0, 0) phải có z_cam ~ 10 (phía trước camera).
    """
    if len(points_xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts_hom = np.hstack([points_xyz[:, :3], np.ones((len(points_xyz), 1), dtype=points_xyz.dtype)])
    pts_cam = pts_hom @ calib.T_cam_velo.T
    return pts_cam[:, :3]


def cam_to_image(points_cam: np.ndarray, P2: np.ndarray, image_shape: tuple[int, ...],
                 min_depth: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chiếu điểm camera frame (N, 3) lên ảnh bằng P2 (3x4).

    Trả về:
      uv    (M, 2) toạ độ pixel của các điểm hợp lệ
      depth (M,)   z_cam của các điểm hợp lệ
      mask  (N,)   bool, True nếu điểm hợp lệ

    Điểm hợp lệ = depth > min_depth VÀ nằm trong ảnh (0 <= u < W, 0 <= v < H).

    TODO(CP2):
      1. Lọc điểm không hợp lệ (NaN/Inf): dữ liệu thật không bao giờ sạch.
      2. Toạ độ đồng nhất, nhân P2 -> (N, 3) = [s*u, s*v, s].
      3. Chia cho s để có (u, v). Chỉ chia với điểm có depth > min_depth.
      4. Lọc theo kích thước ảnh image_shape[:2] = (H, W).
    """
    N = len(points_cam)
    if N == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)

    # 1. Lọc điểm không hợp lệ (NaN/Inf)
    valid_finite = np.isfinite(points_cam).all(axis=1)

    # 2. Toạ độ đồng nhất (N, 4) và nhân P2 (3x4) -> (N, 3)
    pts_cam_safe = np.where(valid_finite[:, None], points_cam, 0.0)
    pts_cam_hom = np.hstack([pts_cam_safe, np.ones((N, 1), dtype=pts_cam_safe.dtype)])
    pts_2d_hom = pts_cam_hom @ P2.T

    # 3. Chia cho s (depth)
    s = pts_2d_hom[:, 2]
    positive_depth = s > min_depth
    valid_divide = valid_finite & positive_depth

    u = np.zeros(N, dtype=np.float32)
    v = np.zeros(N, dtype=np.float32)
    u[valid_divide] = pts_2d_hom[valid_divide, 0] / s[valid_divide]
    v[valid_divide] = pts_2d_hom[valid_divide, 1] / s[valid_divide]

    # 4. Lọc theo kích thước ảnh image_shape[:2] = (H, W)
    H, W = image_shape[:2]
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    mask = valid_divide & in_bounds

    uv = np.stack([u[mask], v[mask]], axis=1)
    depth = s[mask]
    return uv, depth, mask


def project_velo_to_image(points: np.ndarray, calib: KittiCalib, image_shape: tuple[int, ...]):
    """points (N, >=3) velodyne -> (uv, depth, mask) như `cam_to_image`."""
    return cam_to_image(velo_to_cam(points[:, :3], calib), calib.P2, image_shape)


def overlay_points(image: np.ndarray, uv: np.ndarray, depth: np.ndarray,
                   max_depth: float = 50.0, radius: int = 2) -> np.ndarray:
    """Vẽ điểm lên ảnh, màu theo depth (gần = đỏ, xa = xanh)."""
    out = image.copy()
    d = np.clip(depth / max_depth, 0, 1)
    colors = cv2.applyColorMap((255 * (1 - d)).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
    for (u, v), c in zip(uv.astype(int), colors[:, 0]):
        cv2.circle(out, (int(u), int(v)), radius, tuple(int(x) for x in c), -1)
    return out


def _rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def perturb_extrinsic(calib: KittiCalib, roll_deg: float = 0.0, pitch_deg: float = 0.0,
                      yaw_deg: float = 0.0, t_xyz_m: tuple[float, float, float] = (0, 0, 0)) -> KittiCalib:
    """Giả lập calibration drift: xoay/dịch LiDAR trong chính velodyne frame
    (yaw quanh z-up, pitch quanh y-left, roll quanh x-forward) rồi ghép vào Tr_velo_to_cam."""
    D = np.eye(4)
    D[:3, :3] = _rot(*np.deg2rad([roll_deg, pitch_deg, yaw_deg]))
    D[:3, 3] = t_xyz_m
    Tr = np.eye(4)
    Tr[:3, :] = calib.Tr_velo_to_cam
    out = copy.deepcopy(calib)
    out.Tr_velo_to_cam = (Tr @ D)[:3, :]
    return out


def box3d_corners_cam(obj: KittiObject) -> np.ndarray:
    """8 góc (8, 3) của box KITTI trong rectified camera frame.
    Thứ tự: 4 góc đáy (y=0) rồi 4 góc nóc (y=-h)."""
    h, w, l = obj.dimensions
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (R @ np.vstack([x, y, z])).T + obj.location


def draw_box2d(image: np.ndarray, bbox, color=(0, 255, 0), label: str | None = None) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Chiếu điểm LiDAR lên ảnh camera, tô màu theo độ sâu, vẽ 2D box của label")
    ap.add_argument("--data-root", default="data/synthetic",
                    help="thư mục KITTI (data/synthetic, data/kitti_mini) hoặc nuScenes (data/nuscenes_mini_subset)")
    ap.add_argument("--frame", default="000000", help="KITTI: 000011. nuScenes: scene-0103_010")
    ap.add_argument("--out-dir", default="results/figures", help="nơi lưu ảnh overlay")
    ap.add_argument("--roll-deg", type=float, default=0.0, help="giả lập LiDAR bị xoay quanh trục x (độ)")
    ap.add_argument("--pitch-deg", type=float, default=0.0, help="giả lập LiDAR bị xoay quanh trục y (độ)")
    ap.add_argument("--yaw-deg", type=float, default=0.0, help="giả lập LiDAR bị xoay quanh trục z hướng lên (độ)")
    ap.add_argument("--tx", type=float, default=0.0, help="giả lập LiDAR bị dịch theo trục x của LiDAR (mét)")
    ap.add_argument("--ty", type=float, default=0.0, help="giả lập LiDAR bị dịch theo trục y của LiDAR (mét)")
    ap.add_argument("--tz", type=float, default=0.0, help="giả lập LiDAR bị dịch theo trục z của LiDAR (mét)")
    ap.add_argument("--ignore-ego-motion", action="store_true",
                    help="chỉ cho nuScenes: bỏ bù chuyển động xe giữa thời điểm chụp LiDAR và camera")
    args = ap.parse_args()

    kwargs = {"use_ego_motion": not args.ignore_ego_motion} if dataset_type(args.data_root) == "nuscenes" else {}
    fr = load_frame(args.data_root, args.frame, **kwargs)
    calib = perturb_extrinsic(fr["calib"], args.roll_deg, args.pitch_deg, args.yaw_deg,
                              (args.tx, args.ty, args.tz))
    uv, depth, mask = project_velo_to_image(fr["points"], calib, fr["image"].shape)
    vis = overlay_points(fr["image"], uv, depth)
    for obj in fr["labels"]:
        vis = draw_box2d(vis, obj.bbox, label=obj.type)

    tag = f"r{args.roll_deg}_p{args.pitch_deg}_y{args.yaw_deg}_t{args.tx}_{args.ty}_{args.tz}"
    if args.ignore_ego_motion:
        tag += "_noego"
    out = Path(args.out_dir) / f"overlay_{args.frame}_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    print(f"points={len(mask)} inside_image={int(mask.sum())} ({mask.mean():.1%}) -> {out}")


if __name__ == "__main__":
    main()
