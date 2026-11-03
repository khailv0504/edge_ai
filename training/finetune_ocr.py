from ultralytics import YOLO

def main():
    # 1. Khởi tạo mô hình YOLOv8 pretrained (có thể dùng yolov8n.pt hoặc yolov8s.pt)
    model = YOLO("yolov26n.pt")

    # 2. Đường dẫn tới file data.yaml của tập dataset OCR ký tự
    data_yaml_path = "/dataset_ocr/data.yaml"

    # 3. Bắt đầu quá trình fine-tune cho OCR
    model.train(
        data=data_yaml_path,
        epochs=50,                  # Số vòng lặp train (có thể tăng lên 80-100 nếu nhận diện ký tự phức tạp hơn)
        imgsz=320,                  # Kích thước ảnh (với ảnh crop từng ký tự hoặc biển số nhỏ, 320 hoặc 640 tùy thuộc vào dataset)
        batch=64,                   # Batch size phù hợp với VRAM 8GB của RTX 4060
        device=0,                   # Sử dụng GPU 0 (RTX 4060)
        workers=4,                  # Số luồng tải dữ liệu song song
        box=7.5,                    # Hệ số loss cho bounding box (giúp khung bao ký tự chuẩn xác hơn)
        cls=1.0,                    # Hệ số loss cho phân loại ký tự
        name="ocr_character_run"    # Thư mục lưu kết quả trong runs/detect/
    )

    print("Quá trình fine-tune mô hình OCR ký tự đã hoàn tất!")

if __name__ == '__main__':
    main()