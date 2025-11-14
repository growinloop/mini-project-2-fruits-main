from common_imports import *
from config import *


def train_yolo(epochs=None):
    if epochs is None:
        epochs = TRAIN_CONFIG['epochs']

    print("\\n YOLOv5 학습 시작")
    model = YOLO(str(YOLO_WEIGHTS_FILE))

    device = '0' if torch.cuda.is_available() and MODEL_CONFIG['device'] == 'cuda' else 'cpu'

    results = model.train(
        data=str(DATASET_YOLO / 'data.yaml'),
        epochs=epochs,
        imgsz=TRAIN_CONFIG['yolo_img_size'],
        batch=TRAIN_CONFIG['yolo_batch_size'],
        name='yolov5_freshness',
        device=device,
        patience=TRAIN_CONFIG['patience'],
        workers=MODEL_CONFIG['num_workers'],
        project=str(RESULT_DIR)
    )

    print("YOLOv5 학습 완료")
    return model


def evaluate_yolo(model=None, split='test'):
    print(f"\\n YOLOv5 {split} 평가 시작")

    if model is None:
        best_weights = RESULT_DIR / 'yolov5_freshness' / 'weights' / 'best.pt'
        if best_weights.exists():
            model = YOLO(str(best_weights))
        else:
            print("✗ 학습된 모델을 찾을 수 없습니다!")
            return {}

    metrics = model.val(
        data=str(DATASET_YOLO / 'data.yaml'),
        split=split,
        project=str(RESULT_DIR),
        name=f'yolov5_{split}'
    )

    yolo_metrics = {
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr)
    }

    print(f"  YOLOv5 {split} 평가 완료")
    print(f"  - mAP@0.5: {yolo_metrics['mAP50']:.3f}")
    print(f"  - mAP@0.5:0.95: {yolo_metrics['mAP50_95']:.3f}")
    print(f"  - Precision: {yolo_metrics['precision']:.3f}")
    print(f"  - Recall: {yolo_metrics['recall']:.3f}")

    # 결과 저장
    metrics_path = RESULT_DIR / f'yolo_{split}_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': yolo_metrics}, f, indent=4, ensure_ascii=False)
    print(f" Metrics 저장: {metrics_path}")

    return yolo_metrics


if __name__ == "__main__":
    model = train_yolo(epochs=10) # 테스트

    evaluate_yolo(model, split='val')
    evaluate_yolo(model, split='test')