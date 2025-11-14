from common_imports import *
from config import *
from dataset import EffDetDataset, collate_fn


def train_efficientdet(splits, classes, epochs=None):
    if epochs is None:
        epochs = TRAIN_CONFIG['epochs']

    print("\n✓ EfficientDet 학습 시작")

    device = torch.device('cuda' if torch.cuda.is_available() and MODEL_CONFIG['device'] == 'cuda' else 'cpu')
    num_classes = len(classes)

    train_loader = DataLoader(
        EffDetDataset(splits['train'], img_size=TRAIN_CONFIG['effdet_img_size']),
        batch_size=TRAIN_CONFIG['effdet_batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=MODEL_CONFIG['num_workers']
    )
    val_loader = DataLoader(
        EffDetDataset(splits['val'], img_size=TRAIN_CONFIG['effdet_img_size']),
        batch_size=TRAIN_CONFIG['effdet_batch_size'],
        collate_fn=collate_fn,
        num_workers=MODEL_CONFIG['num_workers']
    )

    config = get_efficientdet_config(MODEL_CONFIG['effdet_model'])
    config.num_classes = num_classes
    config.image_size = (TRAIN_CONFIG['effdet_img_size'], TRAIN_CONFIG['effdet_img_size'])

    model = DetBenchTrain(EfficientDet(config, pretrained_backbone=True), config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['effdet_lr'],
        weight_decay=TRAIN_CONFIG['effdet_weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            output = model(images, targets)
            loss = output['loss'] if isinstance(output, dict) else output

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                output = model(images, targets)
                loss = output['loss'] if isinstance(output, dict) else output
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        if epoch == 0 or val_loss < best_loss:
            if val_loss < best_loss:
                best_loss = val_loss

            patience_counter = 0
            torch.save({
                'model_state_dict': model.model.state_dict(),
                'config': config
            }, RESULT_DIR / 'efficientdet_best.pth')
            print(f"✓ Saved (Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    print(f"✓ EfficientDet 학습 완료 (Best Loss: {best_loss:.4f})")

    # Loss curve 저장
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("EfficientDet Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "efficientdet_loss_curve.png")
    plt.close()
    print(f"✓ Loss curve 저장: {RESULT_DIR / 'efficientdet_loss_curve.png'}")

    return config


def evaluate_efficientdet_coco(config, splits, classes):
    if not COCO_AVAILABLE:
        print("✗ pycocotools 없음 - Simple 평가로 대체")
        return evaluate_efficientdet_simple(config, splits, classes)

    print("\n✓ EfficientDet COCO 평가 시작")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    checkpoint_path = RESULT_DIR / 'efficientdet_best.pth'
    if not checkpoint_path.exists():
        print("✗ 모델 가중치 파일이 없습니다!")
        return {}

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    net = EfficientDet(config, pretrained_backbone=False)
    net.load_state_dict(checkpoint['model_state_dict'])

    bench = DetBenchPredict(net)
    bench.eval()
    bench.to(device)

    # GT annotations 로드
    gt_anno_file = DATASET_EFFDET / 'coco_test.json'
    coco_gt = COCO(str(gt_anno_file))

    # 예측 수행
    results = []
    test_data = splits['test']

    for img_id, item in enumerate(tqdm(test_data, desc="Predicting")):
        img = cv2.imread(item['image'])
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        img_resized = cv2.resize(img_rgb, (512, 512))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            detections = bench(img_tensor)

        if detections is None or len(detections.shape) != 3:
            continue

        det = detections[0].cpu().numpy()
        for i in range(det.shape[0]):
            if det.shape[1] < 6:
                continue

            score = float(det[i, 4])
            class_id = int(det[i, 5])

            if score < 0.001 or class_id < 0 or class_id >= len(classes):
                continue

            x1, y1, x2, y2 = det[i, :4]
            x1 = float(x1 * w / 512)
            y1 = float(y1 * h / 512)
            x2 = float(x2 * w / 512)
            y2 = float(y2 * h / 512)

            if x2 <= x1 or y2 <= y1:
                continue

            results.append({
                'image_id': int(img_id),
                'category_id': int(class_id),
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(score)
            })

    if not results:
        print("✗ 예측 결과 없음")
        return {}

    # COCO 평가
    pred_file = RESULT_DIR / 'coco_test_predictions.json'
    with open(pred_file, 'w') as f:
        json.dump(results, f)

    try:
        coco_dt = coco_gt.loadRes(str(pred_file))
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            'mAP50_95': float(coco_eval.stats[0]),
            'mAP50': float(coco_eval.stats[1]),
            'precision': float(coco_eval.stats[0]),
            'recall': float(coco_eval.stats[8])
        }
    except Exception as e:
        print(f"✗ COCO 평가 중 오류: {e}")
        return {}

    # 결과 저장
    metrics_path = RESULT_DIR / 'efficientdet_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': metrics}, f, indent=4, ensure_ascii=False)
    print(f"  ✓ Metrics 저장: {metrics_path}")

    return metrics


def evaluate_efficientdet_simple(config, splits, classes):
    print("\n✓ EfficientDet Simple 평가 중...")
    # 간단한 평가 로직 구현
    return {
        'mAP50': 0.0,
        'mAP50_95': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from data_preprocessing import load_and_split_data

    splits, classes = load_and_split_data()
    if classes:
        config = train_efficientdet(splits, classes, epochs=10)
        evaluate_efficientdet_coco(config, splits, classes)