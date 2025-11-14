from common_imports import *
from config import *

def load_and_split_data():
    jsons = list(JSON_DIR.glob("*.json"))
    if not jsons:
        print("✗ JSON 파일을 찾을 수 없습니다!")
        return {}, []

    train, temp = train_test_split(jsons, train_size=TRAIN_CONFIG['train_split'],
                                   random_state=TRAIN_CONFIG['random_seed'])
    val, test = train_test_split(temp, train_size=TRAIN_CONFIG['val_split'],
                                 random_state=TRAIN_CONFIG['random_seed'])

    splits = {'train': [], 'val': [], 'test': []}
    classes, class_to_idx = [], {}

    for split, files in zip(['train', 'val', 'test'], [train, val, test]):
        for j in tqdm(files, desc=f"Loading {split}"):
            with open(j, 'r', encoding='utf-8') as f:
                d = json.load(f)

            img_path = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                p = IMG_DIR / f"{j.stem}{ext}"
                if p.exists():
                    img_path = str(p)
                    break
            if not img_path:
                continue

            name = f"{d['cate1']}_{d['cate3']}"
            if name not in classes:
                class_to_idx[name] = len(classes)
                classes.append(name)

            bbox = d['bndbox']
            splits[split].append({
                'image': img_path,
                'bbox': [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
                'label': class_to_idx[name],
                'json_stem': j.stem
            })

    print(f"\n✓ 데이터 로딩 완료:")
    print(f"  - Train: {len(splits['train'])}개")
    print(f"  - Val: {len(splits['val'])}개")
    print(f"  - Test: {len(splits['test'])}개")
    print(f"  - Classes: {len(classes)}개")

    return splits, classes


def prepare_yolo_format(splits, classes):
    print("\n✓ YOLO 데이터셋 생성 중...")

    successful_count = {'train': 0, 'val': 0, 'test': 0}
    failed_items = []

    for split, items in splits.items():
        for item in tqdm(items, desc=f"YOLO {split}"):
            try:
                # 이미지 읽기
                img = cv2.imread(item['image'])
                if img is None:
                    failed_items.append((item['json_stem'], "이미지 로드 실패"))
                    continue
                
                h, w = img.shape[:2]
                
                # 이미지가 너무 작은지 확인
                if h < 10 or w < 10:
                    failed_items.append((item['json_stem'], f"이미지 크기 너무 작음: {w}x{h}"))
                    continue

                # 이미지 저장
                img_save = DATASET_YOLO / 'images' / split / f"{item['json_stem']}.jpg"
                success = cv2.imwrite(str(img_save), img)
                if not success:
                    failed_items.append((item['json_stem'], "이미지 저장 실패"))
                    continue

                bbox = item['bbox']
                
                # BBox 유효성 검사
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    failed_items.append((item['json_stem'], f"잘못된 bbox: {bbox}"))
                    continue
                
                # YOLO 형식으로 변환 (정규화된 좌표)
                x_center = ((bbox[0] + bbox[2]) / 2) / w
                y_center = ((bbox[1] + bbox[3]) / 2) / h
                width = (bbox[2] - bbox[0]) / w
                height = (bbox[3] - bbox[1]) / h
                
                # 값 검증 및 클리핑 (0~1 범위)
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                
                # 너무 작은 bbox 필터링
                if width < 0.01 or height < 0.01:
                    failed_items.append((item['json_stem'], f"bbox 너무 작음: {width:.4f}x{height:.4f}"))
                    continue
                
                # 클래스 ID 검증
                if item['label'] < 0 or item['label'] >= len(classes):
                    failed_items.append((item['json_stem'], f"잘못된 클래스 ID: {item['label']}"))
                    continue

                # 라벨 파일 저장 (YOLO 형식: class_id x_center y_center width height)
                label_save = DATASET_YOLO / 'labels' / split / f"{item['json_stem']}.txt"
                with open(label_save, 'w') as f:
                    f.write(f"{item['label']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                successful_count[split] += 1
                    
            except Exception as e:
                failed_items.append((item['json_stem'], f"예외 발생: {str(e)}"))
                continue

    # 결과 출력
    print(f"\n✓ YOLO 데이터셋 생성 완료: {DATASET_YOLO}")
    print(f"\n성공:")
    for split in ['train', 'val', 'test']:
        img_count = len(list((DATASET_YOLO / 'images' / split).glob('*.jpg')))
        label_count = len(list((DATASET_YOLO / 'labels' / split).glob('*.txt')))
        print(f"  - {split}: 이미지 {img_count}개, 라벨 {label_count}개 (처리 성공: {successful_count[split]})")
    
    if failed_items:
        print(f"\n실패 ({len(failed_items)}개):")
        for stem, reason in failed_items[:10]:
            print(f"  - {stem}: {reason}")
        if len(failed_items) > 10:
            print(f"  ... 외 {len(failed_items)-10}개")

    # data.yaml 생성
    data_yaml = {
        'path': str(DATASET_YOLO.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }

    with open(DATASET_YOLO / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    
    print(f"\n✓ data.yaml 생성 완료")


def create_coco_annotations(splits, classes):
    print("\n✓ COCO annotations 생성 중...")

    coco_categories = [
        {"id": idx, "name": name, "supercategory": "freshness"}
        for idx, name in enumerate(classes)
    ]

    for split in ['train', 'val', 'test']:
        coco_data = {
            'info': {
                "description": f"Custom Dataset - {split} Set",
                "version": "1.0",
                "year": 2025,
            },
            'licenses': [{"id": 0, "name": "Unknown", "url": ""}],
            'categories': coco_categories,
            'images': [],
            'annotations': []
        }

        ann_id = 0

        for img_id, item in enumerate(splits[split]):
            try:
                img = cv2.imread(item['image'])
                if img is None:
                    continue
                h, w = img.shape[:2]
            except Exception as e:
                print(f"✗ 이미지 로드 실패 {item['image']}: {e}")
                continue

            coco_data['images'].append({
                'id': img_id,
                'file_name': str(item['image']),
                'width': w,
                'height': h
            })

            bbox = item['bbox']
            coco_data['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': item['label'],
                'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'iscrowd': 0
            })
            ann_id += 1

        anno_path = DATASET_EFFDET / f'coco_{split}.json'
        with open(anno_path, 'w') as f:
            json.dump(coco_data, f, indent=4)

        print(f"  - {split}: {anno_path} ({len(coco_data['images'])}개 이미지)")

    return DATASET_EFFDET / 'coco_test.json'


def validate_yolo_labels(dataset_path):
    """YOLO 라벨 파일 검증"""
    print("\n✓ YOLO 라벨 파일 검증 중...")
    
    issues = []
    total_labels = 0
    
    for split in ['train', 'val', 'test']:
        label_dir = dataset_path / 'labels' / split
        image_dir = dataset_path / 'images' / split
        
        if not label_dir.exists():
            continue
        
        label_files = list(label_dir.glob('*.txt'))
        image_files = list(image_dir.glob('*.jpg'))
        
        # 이미지-라벨 매칭 확인
        label_stems = {f.stem for f in label_files}
        image_stems = {f.stem for f in image_files}
        
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems
        
        if missing_labels:
            issues.append(f"{split}: {len(missing_labels)}개 이미지에 라벨 없음")
        if missing_images:
            issues.append(f"{split}: {len(missing_images)}개 라벨에 이미지 없음")
            
        for label_file in label_files:
            total_labels += 1
            try:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    
                    # 빈 파일 체크
                    if not content:
                        issues.append(f"빈 파일: {label_file.name}")
                        continue
                    
                    # 각 라인 검증
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if not line.strip():
                            continue
                            
                        parts = line.strip().split()
                        if len(parts) != 5:
                            issues.append(f"{label_file.name} (line {line_num}): 5개 값 필요, {len(parts)}개 발견")
                            continue
                        
                        # 숫자 변환 테스트
                        try:
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # 범위 체크 (0~1)
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                   0 < width <= 1 and 0 < height <= 1):
                                issues.append(f"{label_file.name} (line {line_num}): 좌표 범위 오류 [{x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f}]")
                            
                            if cls_id < 0:
                                issues.append(f"{label_file.name} (line {line_num}): 음수 클래스 ID")
                                
                        except ValueError as e:
                            issues.append(f"{label_file.name} (line {line_num}): 숫자 변환 실패 - {e}")
                            
            except Exception as e:
                issues.append(f"{label_file.name}: 파일 읽기 실패 - {e}")
    
    print(f"총 {total_labels}개 라벨 파일 검증 완료")
    
    if issues:
        print(f"\n✗ 발견된 문제: {len(issues)}개")
        for issue in issues[:20]:  # 처음 20개만 출력
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... 외 {len(issues)-20}개")
        return False
    else:
        print("✓ 모든 라벨 파일이 정상입니다!")
        return True


def clean_cache_files():
    """YOLO 캐시 파일 삭제"""
    print("\n✓ 캐시 파일 정리 중...")
    cache_files = list(DATASET_YOLO.rglob("*.cache"))
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            print(f"  - 삭제: {cache_file}")
        except Exception as e:
            print(f"  - 삭제 실패: {cache_file} ({e})")


if __name__ == "__main__":
    from config import create_directories
    create_directories()
    
    # 캐시 파일 먼저 정리
    clean_cache_files()

    splits, classes = load_and_split_data()
    if classes:
        prepare_yolo_format(splits, classes)
        
        # 라벨 검증
        if validate_yolo_labels(DATASET_YOLO):
            create_coco_annotations(splits, classes)
            print("\n✓ 모든 데이터 전처리 완료!")
        else:
            print("\n✗ 라벨 파일에 문제가 있습니다. 위 메시지를 확인하세요.")
    else:
        print("✗ 데이터를 로드할 수 없습니다!")