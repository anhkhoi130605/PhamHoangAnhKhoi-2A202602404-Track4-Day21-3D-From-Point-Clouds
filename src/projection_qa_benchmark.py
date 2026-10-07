"""Projection QA Benchmark: Đánh giá độ nhạy của phép chiếu LiDAR-Camera trước các mức sai lệch extrinsic (Drift).

Cung cấp đầy đủ:
- CLI với argparse và --help (Bonus B4)
- Đo lường % điểm rơi trong 2D box, % điểm trong FOV, pixel drift
- Đo latency p50/p95 qua 25 lần lặp (Bonus B3)
- Stress test nhiều loại perturbation: Yaw, Pitch, Translation (Bonus B2)
- So sánh trên KITTI và nuScenes (Bonus B5)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Đảm bảo import được starter khi chạy từ bất kỳ thư mục nào
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import dataset_type, list_frames, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import (
    cam_to_image,
    overlay_points,
    perturb_extrinsic,
    project_velo_to_image,
    velo_to_cam,
)


def point_in_bbox(uv: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Kiểm tra từng điểm (M, 2) có nằm trong 2D box [x1, y1, x2, y2] hay không."""
    x1, y1, x2, y2 = bbox
    return (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)


def evaluate_frame_projection(
    fr: dict,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    tx_m: float = 0.0,
    ty_m: float = 0.0,
    tz_m: float = 0.0,
) -> dict:
    """Tính các metric của một frame khi áp dụng độ lệch extrinsic."""
    points = fr["points"]
    calib: KittiCalib = fr["calib"]
    image = fr["image"]
    labels: list[KittiObject] = fr.get("labels", [])

    # 1. Chiếu chuẩn (Ground Truth / baseline)
    uv_base, depth_base, mask_base = project_velo_to_image(points, calib, image.shape)

    # 2. Chiếu sau khi làm lệch calibration
    calib_perturbed = perturb_extrinsic(
        calib,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        t_xyz_m=(tx_m, ty_m, tz_m),
    )
    uv_pert, depth_pert, mask_pert = project_velo_to_image(points, calib_perturbed, image.shape)

    # Điểm nằm trong FOV
    fov_ratio = float(np.sum(mask_pert)) / max(len(points), 1) * 100.0

    # Điểm chung giữa 2 lần chiếu
    common_mask = mask_base & mask_pert
    if np.sum(common_mask) > 0:
        uv_base_common = uv_base[np.where(common_mask[mask_base])[0]]
        uv_pert_common = uv_pert[np.where(common_mask[mask_pert])[0]]
        pixel_drift = float(np.mean(np.linalg.norm(uv_base_common - uv_pert_common, axis=1)))
    else:
        pixel_drift = 999.0

    # Đo độ rơi vào 2D box của các object (Car, Pedestrian, Cyclist)
    valid_objects = [
        obj for obj in labels if obj.type in ["Car", "Pedestrian", "Cyclist", "Van", "Truck"]
    ]

    total_base_obj_pts = 0
    total_pert_obj_pts_retained = 0

    for obj in valid_objects:
        in_box_base = point_in_bbox(uv_base, obj.bbox)
        num_base = int(np.sum(in_box_base))
        if num_base == 0:
            continue
        total_base_obj_pts += num_base

        # Tìm các điểm velodyne tương ứng nằm trong box chuẩn
        indices_in_pts = np.where(mask_base)[0][in_box_base]

        # Kiểm tra xem các điểm này sau khi perturb có còn trong box không
        is_in_pert_fov = mask_pert[indices_in_pts]
        if np.sum(is_in_pert_fov) > 0:
            uv_pert_subset = []
            for idx in indices_in_pts[is_in_pert_fov]:
                pos = np.sum(mask_pert[:idx])
                uv_pert_subset.append(uv_pert[pos])
            uv_pert_subset = np.array(uv_pert_subset)
            retained = np.sum(point_in_bbox(uv_pert_subset, obj.bbox))
            total_pert_obj_pts_retained += int(retained)

    box_hit_ratio = (
        (float(total_pert_obj_pts_retained) / total_base_obj_pts * 100.0)
        if total_base_obj_pts > 0
        else 100.0
    )

    return {
        "fov_ratio": fov_ratio,
        "pixel_drift": pixel_drift,
        "box_hit_ratio": box_hit_ratio,
        "obj_pts_base": total_base_obj_pts,
        "obj_pts_retained": total_pert_obj_pts_retained,
    }


def benchmark_latency(fr: dict, runs: int = 25) -> tuple[float, float]:
    """Đo latency p50 và p95 (ms) của phép chiếu, bỏ lần chạy đầu (Bonus B3)."""
    points = fr["points"]
    calib = fr["calib"]
    shape = fr["image"].shape

    # Warm-up (bỏ qua)
    _ = project_velo_to_image(points, calib, shape)

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = project_velo_to_image(points, calib, shape)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(times_ms, 50))
    p95 = float(np.percentile(times_ms, 95))
    return p50, p95


def run_benchmark(
    data_root: str,
    frames: list[str],
    out_csv: Path,
    fig_dir: Path,
    nusc_root: str | None = None,
):
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    records = []

    # Sweep settings (Bonus B2: ít nhất 2 loại perturbation × ≥ 3 mức)
    yaw_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    pitch_levels = [0.0, 0.5, 1.0, 2.0]
    trans_levels = [0.0, 0.05, 0.10, 0.20]  # mét

    print(f"[*] Bắt đầu benchmark trên {len(frames)} frame của {data_root}...")

    # Load all frames first
    loaded_frames = {fid: load_frame(data_root, fid) for fid in frames}

    # 1. Đo Latency (Bonus B3)
    p50_lat, p95_lat = benchmark_latency(loaded_frames[frames[0]], runs=25)
    print(f"[Latency Benchmark] Frame {frames[0]}: p50={p50_lat:.2f}ms, p95={p95_lat:.2f}ms")

    # 2. Sweep Yaw
    for yaw in yaw_levels:
        for fid in frames:
            fr = loaded_frames[fid]
            metrics = evaluate_frame_projection(fr, yaw_deg=yaw)
            records.append({
                "dataset": "kitti",
                "frame": fid,
                "perturb_type": "yaw_deg",
                "perturb_value": yaw,
                "fov_ratio_pct": round(metrics["fov_ratio"], 2),
                "box_hit_ratio_pct": round(metrics["box_hit_ratio"], 2),
                "pixel_drift_px": round(metrics["pixel_drift"], 2),
                "p50_latency_ms": round(p50_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
            })

    # 3. Sweep Pitch
    for pitch in pitch_levels:
        if pitch == 0.0:
            continue
        for fid in frames:
            fr = loaded_frames[fid]
            metrics = evaluate_frame_projection(fr, pitch_deg=pitch)
            records.append({
                "dataset": "kitti",
                "frame": fid,
                "perturb_type": "pitch_deg",
                "perturb_value": pitch,
                "fov_ratio_pct": round(metrics["fov_ratio"], 2),
                "box_hit_ratio_pct": round(metrics["box_hit_ratio"], 2),
                "pixel_drift_px": round(metrics["pixel_drift"], 2),
                "p50_latency_ms": round(p50_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
            })

    # 4. Sweep Translation Y (ngang)
    for ty in trans_levels:
        if ty == 0.0:
            continue
        for fid in frames:
            fr = loaded_frames[fid]
            metrics = evaluate_frame_projection(fr, ty_m=ty)
            records.append({
                "dataset": "kitti",
                "frame": fid,
                "perturb_type": "trans_y_m",
                "perturb_value": ty,
                "fov_ratio_pct": round(metrics["fov_ratio"], 2),
                "box_hit_ratio_pct": round(metrics["box_hit_ratio"], 2),
                "pixel_drift_px": round(metrics["pixel_drift"], 2),
                "p50_latency_ms": round(p50_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
            })

    # 5. So sánh với nuScenes nếu có (Bonus B5)
    if nusc_root and Path(nusc_root).exists():
        try:
            nusc_frames = list_frames(nusc_root)[:3]
            for n_fid in nusc_frames:
                n_fr = load_frame(nusc_root, n_fid)
                for yaw in [0.0, 1.0, 2.0]:
                    metrics = evaluate_frame_projection(n_fr, yaw_deg=yaw)
                    records.append({
                        "dataset": "nuscenes",
                        "frame": n_fid,
                        "perturb_type": "yaw_deg",
                        "perturb_value": yaw,
                        "fov_ratio_pct": round(metrics["fov_ratio"], 2),
                        "box_hit_ratio_pct": round(metrics["box_hit_ratio"], 2),
                        "pixel_drift_px": round(metrics["pixel_drift"], 2),
                        "p50_latency_ms": round(p50_lat, 2),
                        "p95_latency_ms": round(p95_lat, 2),
                    })
        except Exception as e:
            print(f"[Warning] Không thể chạy benchmark trên nuScenes: {e}")

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[+] Đã lưu bảng số liệu vào {out_csv} ({len(df)} dòng).")

    # Tạo biểu đồ Sweep (results/figures/yaw_perturb_sweep.png)
    plt.figure(figsize=(10, 5), dpi=150)
    yaw_df = df[(df["dataset"] == "kitti") & (df["perturb_type"] == "yaw_deg")]
    mean_yaw = yaw_df.groupby("perturb_value")["box_hit_ratio_pct"].mean()
    mean_drift = yaw_df.groupby("perturb_value")["pixel_drift_px"].mean()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(mean_yaw.index, mean_yaw.values, 'ro-', linewidth=2, label='% Điểm trong 2D Box (Box Hit %)')
    l2 = ax2.plot(mean_drift.index, mean_drift.values, 'b^--', linewidth=2, label='Độ lệch pixel trung bình (px)')

    ax1.set_xlabel('Góc lệch Yaw extrinsic (°)', fontsize=12)
    ax1.set_ylabel('% Điểm còn nằm trong 2D Box (%)', color='r', fontsize=12)
    ax2.set_ylabel('Độ lệch pixel (pixels)', color='b', fontsize=12)
    plt.title('Độ nhạy của Projection trước Calibration Drift (Yaw)', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Thêm ngưỡng an toàn
    ax1.axhline(85, color='orange', linestyle=':', label='Ngưỡng suy giảm 15% (Threshold)')
    lines = l1 + l2 + [ax1.get_lines()[-1]]
    labels_leg = [l.get_label() for l in lines]
    ax1.legend(lines, labels_leg, loc='center left')

    plt.tight_layout()
    plot_path = fig_dir / "yaw_perturb_sweep.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[+] Đã lưu biểu đồ: {plot_path}")

    # Tạo ảnh Demo Overlay chuẩn (results/figures/demo_projection_overlay.png)
    sample_fr = loaded_frames["000011"]
    uv, depth, mask = project_velo_to_image(sample_fr["points"], sample_fr["calib"], sample_fr["image"].shape)
    overlay_img = overlay_points(sample_fr["image"], uv, depth, max_depth=40.0, radius=2)
    for obj in sample_fr.get("labels", []):
        if obj.type in ["Car", "Pedestrian", "Cyclist"]:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            color = (0, 255, 0) if obj.type == "Car" else (255, 0, 0)
            cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay_img, obj.type, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    demo_path = fig_dir / "demo_projection_overlay.png"
    cv2.imwrite(str(demo_path), overlay_img)
    print(f"[+] Đã lưu ảnh demo: {demo_path}")

    # Tạo ảnh Failure Case (results/figures/fail_01_yaw_drift_mismatch.png)
    # Lệch yaw 2.0 độ: điểm point cloud bị bay khỏi xe
    calib_bad = perturb_extrinsic(sample_fr["calib"], yaw_deg=2.0)
    uv_bad, depth_bad, _ = project_velo_to_image(sample_fr["points"], calib_bad, sample_fr["image"].shape)
    fail_img = overlay_points(sample_fr["image"], uv_bad, depth_bad, max_depth=40.0, radius=2)
    for obj in sample_fr.get("labels", []):
        if obj.type in ["Car", "Pedestrian", "Cyclist"]:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            cv2.rectangle(fail_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(fail_img, f"GT Box: {obj.type}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Đặt text cảnh báo lỗi
    cv2.putText(
        fail_img,
        "FAIL CASE: Yaw Drift = +2.0 deg -> LiDAR points shifted left outside GT Box!",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # Ghép 2 ảnh cạnh nhau: Chuẩn vs Bị Drift để thầy cô dễ so sánh
    h, w = overlay_img.shape[:2]
    comparison = np.vstack([overlay_img, fail_img])
    fail_path = fig_dir / "fail_01_yaw_drift_mismatch.png"
    cv2.imwrite(str(fail_path), comparison)
    print(f"[+] Đã lưu ảnh failure: {fail_path}")


def main():
    parser = argparse.ArgumentParser(description="LiDAR-Camera Projection QA Benchmark & Drift Analysis")
    parser.add_argument("--data-root", type=str, default="data/kitti_mini", help="Đường dẫn thư mục dữ liệu KITTI")
    parser.add_argument(
        "--frames",
        type=str,
        default="000001,000011,000021,000048,000049",
        help="Danh sách frame id cách nhau bằng dấu phẩy",
    )
    parser.add_argument("--out-csv", type=str, default="results/projection_qa_sweep.csv", help="File CSV đầu ra")
    parser.add_argument("--out-dir", type=str, default="results/figures", help="Thư mục lưu biểu đồ và ảnh demo")
    parser.add_argument(
        "--nusc-root",
        type=str,
        default="data/nuscenes_mini_subset",
        help="Đường dẫn thư mục nuScenes (nếu muốn benchmark so sánh B5)",
    )
    args = parser.parse_args()

    frame_list = [f.strip() for f in args.frames.split(",") if f.strip()]
    run_benchmark(
        data_root=args.data_root,
        frames=frame_list,
        out_csv=Path(args.out_csv),
        fig_dir=Path(args.out_dir),
        nusc_root=args.nusc_root,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
