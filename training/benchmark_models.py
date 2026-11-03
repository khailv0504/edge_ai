
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "benchmark"

MODELS = {
    "detector": {
        "dataset_env": "DETECTOR_DATA_DIR",
        "weights_env": "DETECTOR_WEIGHTS",
        "dataset": "dataset_detector",
        "run": "plate_detector_run",
        "imgsz": 640,
        "nc": 2,
    },
    "ocr": {
        "dataset_env": "OCR_DATA_DIR",
        "weights_env": "OCR_WEIGHTS",
        "dataset": "dataset_ocr",
        "run": "ocr_character_run",
        "imgsz": 320,
        "nc": 30,
    },
}


def get_paths(config):
    dataset = Path(
        os.environ.get(
            config["dataset_env"],
            str(PROJECT_ROOT / config["dataset"])
        )
    ).expanduser().resolve()

    default_weights = (
        PROJECT_ROOT / "runs" /
        config["run"] / "weights" / "best.pt"
    )

    weights = Path(
        os.environ.get(
            config["weights_env"],
            str(default_weights)
        )
    ).expanduser().resolve()

    if not (dataset / "data.yaml").is_file():
        raise FileNotFoundError(
            f"Missing dataset YAML: {dataset / 'data.yaml'}"
        )

    if not weights.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {weights}"
        )

    return dataset, weights


def prepare_yaml(name, config, dataset):
    with open(dataset / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["nc"] == config["nc"]
    assert len(data["names"]) == config["nc"]

    # Sửa lại đường dẫn theo cấu trúc Roboflow
    data["path"] = str(dataset)
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"

    for folder in ["test/images", "test/labels"]:
        if not (dataset / folder).is_dir():
            raise FileNotFoundError(
                f"Missing directory: {dataset / folder}"
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    runtime_yaml = OUTPUT_ROOT / f"{name}_runtime.yaml"

    with open(runtime_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data, f, sort_keys=False, allow_unicode=True
        )

    return runtime_yaml, data["names"]


def measure_latency(model, dataset, imgsz, device):
    # Load ảnh một lần, loại bỏ thời gian đọc ổ đĩa
    image_dir = dataset / "test" / "images"
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    images = []

    for path in sorted(image_dir.rglob("*")):
        if path.suffix.lower() not in extensions:
            continue

        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)

        if len(images) >= 10:
            break

    if not images:
        raise RuntimeError("No valid test images")

    use_cuda = (
        str(device).lower() not in {"cpu", "mps"}
        and torch.cuda.is_available()
    )

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize()

    warmup = 10
    repeats = 50
    times = []

    for i in range(warmup + repeats):
        img = images[i % len(images)]

        synchronize()
        start = time.perf_counter()

        model.predict(
            source=img,
            imgsz=imgsz,
            device=device,
            conf=0.25,
            iou=0.6,
            verbose=False,
        )

        synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        if i >= warmup:
            times.append(elapsed)

    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "fps_batch1": float(1000 / np.mean(times)),
    }


def evaluate(name, device):
    config = MODELS[name]
    dataset, weights = get_paths(config)

    runtime_yaml, class_names = prepare_yaml(
        name, config, dataset
    )

    print(f"\n{'=' * 60}")
    print(f"MODEL: {name.upper()}")
    print(f"WEIGHTS: {weights}")
    print(f"DATASET: {dataset}")
    print(f"IMAGE SIZE: {config['imgsz']}")
    print(f"{'=' * 60}\n")

    model = YOLO(str(weights))

    # 1. Accuracy evaluation trên tập test
    metrics = model.val(
        data=str(runtime_yaml),
        split="test",
        imgsz=config["imgsz"],
        batch=16,
        device=device,
        workers=4,
        conf=0.001,
        iou=0.6,
        plots=True,
        save_json=False,
        project=str(OUTPUT_ROOT),
        name=f"{name}_test",
        exist_ok=True,
    )

    # 2. Benchmark latency, batch size = 1
    latency = measure_latency(
        model,
        dataset,
        config["imgsz"],
        device
    )

    # 3. Metrics theo từng class có mặt trong tập test
    per_class = []

    for cls_id in metrics.box.ap_class_index:
        idx = int(cls_id)

        per_class.append({
            "class_id": idx,
            "name": str(class_names[idx]),
            "map50_95": float(metrics.box.maps[idx]),
        })

    # 4. Tổng hợp kết quả
    report = {
        "model": name,
        "weights": str(weights),
        "dataset": str(dataset),
        "imgsz": config["imgsz"],
        "device": str(device),
        "accuracy": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map75": float(metrics.box.map75),
            "map50_95": float(metrics.box.map),
        },
        "validation_speed_ms": metrics.speed,
        "latency_batch1": latency,
        "per_class": per_class,
    }

    output_file = OUTPUT_ROOT / f"{name}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n===== {name.upper()} RESULTS =====")
    for metric, value in report["accuracy"].items():
        print(f"{metric:12s}: {value:.4f}")

    print(f"P50 latency: {latency['p50_ms']:.2f} ms")
    print(f"P95 latency: {latency['p95_ms']:.2f} ms")
    print(f"FPS batch=1: {latency['fps_batch1']:.2f}")
    print(f"Saved: {output_file}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["detector", "ocr", "both"],
        default="both",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="GPU index (0) or cpu",
    )

    args = parser.parse_args()

    targets = (
        ["detector", "ocr"]
        if args.target == "both"
        else [args.target]
    )

    for target in targets:
        evaluate(target, args.device)


if __name__ == "__main__":
    main()
