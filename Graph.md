<HoVaTen>-<MSSV>-Track4-Day21/
├── README.md, TOPICS.md, CHECKPOINTS.md, RUBRIC.md, SUBMISSION.md, RULES.md   # tài liệu đề bài
├── requirements.txt
├── starter/                 # code khởi đầu, đã viết sẵn
│   ├── datasets.py          #   đọc KITTI và nuScenes theo cùng một cách
│   ├── kitti_io.py          #   đọc file KITTI: velodyne, calib, ảnh, label
│   ├── nuscenes_io.py       #   đọc nuScenes, chuyển sang cùng cấu trúc với KITTI
│   ├── projection.py        #   chiếu điểm lên ảnh, vẽ overlay, làm lệch calibration  ← CÓ 2 HÀM TODO
│   ├── data_health.py       #   thống kê point cloud ra CSV
│   └── perturb.py           #   các phép làm suy giảm dữ liệu cho topic C
├── data/                    # dữ liệu, KHÔNG sửa
├── src/                     # ← BẠN VIẾT: toàn bộ code của bạn đặt ở đây
├── results/                 # ← BẠN TẠO: CSV số liệu, thư mục figures/ chứa ảnh
├── report/
│   └── REPORT.md            # ← BẠN ĐIỀN: thông tin học viên + báo cáo 6 mục
└── tools/                   # check_submission.py, verify_data.py, script tải thêm dữ liệu (tuỳ chọn)