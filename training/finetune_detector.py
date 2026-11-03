
from pathlib import Path
import os
import yaml
from ultralytics import YOLO


def main():
    # Xác định thư mục gốc của repository
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Ưu tiên biến môi trường, mặc định dùng dataset local
    DATASET_ROOT = Path(
        os.environ.get(
            "DETECTOR_DATA_DIR",
            str(PROJECT_ROOT / "dataset_detector")
        )
    ).expanduser().resolve()

    # File cấu hình gốc của dataset
    DATA_YAML = DATASET_ROOT / "data.yaml"

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy data.yaml tại: {DATA_YAML}"
        )

    # Đọc cấu hình và cập nhật đường dẫn theo môi trường
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["path"] = str(DATASET_ROOT)
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"

    # Lưu cấu hình runtime vào thư mục có quyền ghi
    OUTPUT_ROOT = PROJECT_ROOT / "runs"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    runtime_yaml = OUTPUT_ROOT / "detector_runtime.yaml"

    with open(runtime_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    print(f"Project: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Config: {runtime_yaml}")

    # Load YOLO26 nano pretrained
    model = YOLO("yolo26n.pt")

    # Fine-tuning
    model.train(
        data=str(runtime_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=str(OUTPUT_ROOT),
        name="plate_detector_run"
    )

    print("Fine-tuning completed!")


if __name__ == "__main__":
    main()
