
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    # 1. Xác định thư mục gốc dự án
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # 2. Đường dẫn checkpoint đã fine-tune
    WEIGHTS_PATH = (
        PROJECT_ROOT
        / "runs"
        / "ocr_character_run"
        / "weights"
        / "best.pt"
    )

    # 3. Đường dẫn dataset OCR
    DATASET_ROOT = PROJECT_ROOT / "dataset_ocr"
    DATA_YAML = DATASET_ROOT / "data.yaml"

    # 4. Thư mục chứa mô hình đã export
    OUTPUT_DIR = PROJECT_ROOT / "models"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH = OUTPUT_DIR / "recognizer.onnx"

    # 5. Kiểm tra file đầu vào
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {WEIGHTS_PATH}"
        )

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy dataset YAML: {DATA_YAML}"
        )

    # 6. Chuẩn hóa YAML phục vụ INT8 calibration
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["path"] = str(DATASET_ROOT.resolve())
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"

    if config["nc"] != 30 or len(config["names"]) != 30:
        raise ValueError("OCR dataset phải có 30 classes")

    # Calibration mặc định sử dụng split validation
    if not (DATASET_ROOT / "valid" / "images").is_dir():
        raise FileNotFoundError(
            "Không tìm thấy valid/images cho INT8 calibration"
        )

    runtime_yaml = OUTPUT_DIR / "ocr_runtime.yaml"

    with open(runtime_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_PATH}")

    # 7. Load checkpoint
    model = YOLO(str(WEIGHTS_PATH))

    # 8. Export ONNX INT8
    exported_path = model.export(
        format="onnx",
        imgsz=320,
        quantize=8,
        data=str(runtime_yaml),
        batch=1,
        dynamic=False,
        simplify=True,
    )

    # 9. Sao chép và đổi tên file
    exported_path = Path(exported_path).resolve()

    if not exported_path.is_file():
        raise RuntimeError(
            f"Export thất bại: {exported_path}"
        )

    if exported_path != OUTPUT_PATH.resolve():
        shutil.copy2(exported_path, OUTPUT_PATH)

    print("\n=== EXPORT COMPLETED ===")
    print(f"Format: ONNX INT8")
    print(f"Input size: 320x320")
    print(f"Batch size: 1")
    print(f"Saved to: {OUTPUT_PATH}")
    print(
        f"File size: "
        f"{OUTPUT_PATH.stat().st_size / 1024**2:.2f} MB"
    )


if __name__ == "__main__":
    main()
