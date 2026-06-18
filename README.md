🚀 Dự án Nhận Diện Cảm Xúc Khuôn Mặt - Đã Hoàn Thành!
Toàn bộ hệ thống (từ huấn luyện mô hình đến giao diện web testing) đã được xây dựng thành công. Dưới đây là kiến trúc dự án và hướng dẫn sử dụng.

📂 Cấu trúc dự án mới
text

emotion_ai/
├── archive/              (Tập dữ liệu FER2013 của bạn)
│   ├── train/
│   └── test/
├── models/               (Nơi chứa mô hình sau khi train)
├── src/                  (Mã nguồn huấn luyện AI)
│   ├── dataset.py        (Xử lý, Data Augmentation)
│   ├── model.py          (Định nghĩa kiến trúc CNN)
│   ├── train.py          (Script huấn luyện)
│   └── export_tflite.py  (Script chuyển đổi h5 -> tflite cho Mobile)
├── static/               (Tài nguyên Web UI)
│   ├── style.css         (Giao diện Dark mode + Glassmorphism cực đẹp)
│   └── script.js         (Xử lý logic Webcam và Upload File)
├── templates/
│   └── index.html        (Giao diện web)
├── app.py                (Backend FastAPI chạy server Web Test)
├── requirements.txt      (Các thư viện cần thiết)
└── README.md
🛠 Hướng dẫn Khởi chạy
IMPORTANT

Trước khi chạy, hãy đảm bảo bạn cài đặt các thư viện cần thiết trong môi trường Python của bạn:

bash

pip install -r requirements.txt
1. Huấn luyện Mô hình
Nếu bạn chưa có mô hình đã train, hãy mở terminal và chạy lệnh sau (Quá trình này có thể mất một thời gian tùy vào GPU/CPU của máy bạn):

bash

python src/train.py
TIP

Mô hình tốt nhất sẽ tự động được lưu vào models/emotion_model.h5.

2. Xuất Mô hình cho Mobile (TFLite)
Khi đã có file .h5, bạn có thể tối ưu hóa và xuất ra định dạng .tflite để dùng cho Android/iOS:

bash

python src/export_tflite.py
3. Khởi chạy Môi trường Web Test
Để chạy thử nghiệm mô hình thực tế qua giao diện web siêu xịn, hãy chạy server backend:

bash

uvicorn app:app --reload
Sau đó mở trình duyệt và truy cập vào địa chỉ: http://127.0.0.1:8000

✨ Điểm nhấn của UI
Trang web thử nghiệm được thiết kế với phong cách hiện đại:

Glassmorphism: Giao diện mờ kính sang trọng, kết hợp với các hiệu ứng gradient và đổ bóng.
Micro-animations: Chuyển đổi tab mượt mà, hiệu ứng hover nút bấm.
Live Webcam & Upload: Hỗ trợ vẽ Bounding Box trực tiếp lên luồng video webcam để hiển thị Cảm Xúc & Độ Tự Tin (Confidence) theo thời gian thực.# emotion-ai
