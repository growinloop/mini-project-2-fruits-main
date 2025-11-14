from pathlib import Path

# Paths
BASE_DIR = Path.cwd()
IMG_DIR = BASE_DIR / "data/raw/images"
JSON_DIR = BASE_DIR / "data/raw/json_labels"
DATASET_YOLO = BASE_DIR / "processed/preprocessed_data/yolov5"
DATASET_EFFDET = BASE_DIR / "processed/preprocessed_data/efficientdet"
RESULT_DIR = BASE_DIR / "processed/results_comparison"
YOLO_WEIGHTS_FILE = RESULT_DIR / "yolov5su.pt"

print(f" BASE_DIR: {BASE_DIR}")
print(f" IMG_DIR: {IMG_DIR}")
print(f" JSON_DIR: {JSON_DIR}")
print(f" DATASET_YOLO: {DATASET_YOLO}")
print(f" DATASET_EFFDET: {DATASET_EFFDET}")
print(f" RESULT_DIR: {RESULT_DIR}")

# Training Parameters
TRAIN_CONFIG = {
    'epochs': 100,
    'yolo_batch_size': 16,
    'yolo_img_size': 640,
    'effdet_batch_size': 4,
    'effdet_img_size': 512,
    'effdet_lr': 0.01,
    'effdet_weight_decay': 0.0001,
    'patience': 30,
    'random_seed': 42,
    'train_split': 0.8,
    'val_split': 0.5,
}

# Model Parameters
MODEL_CONFIG = {
    'yolo_model': 'yolov5su',
    'effdet_model': 'tf_efficientdet_d0',
    'num_workers': 0,
    'device': 'cuda',
}

#  Create Directories
def create_directories():
    for d in [DATASET_YOLO, DATASET_EFFDET, RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # YOLO 하위 디렉토리
    for split in ['train', 'val', 'test']:
        (DATASET_YOLO / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_YOLO / 'labels' / split).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print(" 디렉토리 생성 완료")
    print(f"  - YOLO: {DATASET_YOLO}")
    print(f"  - EfficientDet: {DATASET_EFFDET}")
    print(f"  - Results: {RESULT_DIR}")
