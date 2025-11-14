import json
from config import *
from utils import setup_korean_font, print_stage
from data_preprocessing import load_and_split_data, prepare_yolo_format, create_coco_annotations, clean_cache_files
from yolo_trainer import train_yolo, evaluate_yolo
from efficientdet_trainer import train_efficientdet, evaluate_efficientdet_coco
from visualization import visualize_comparison, print_results_table


def main():
    print("=" * 70)
    print("YOLOv5 vs EfficientDet 성능 비교 시스템")
    print("=" * 70)

    # 초기 설정
    create_directories()
    setup_korean_font()

    # 1단계: 데이터 전처리
    print_stage(1, "데이터 전처리")
    splits, classes = load_and_split_data()

    if not classes:
        print("✗ 데이터셋을 찾을 수 없습니다!")
        return

    if len(splits['test']) == 0:
        print("✗ 테스트 데이터가 없습니다!")
        return

    print(f"\n✓ 클래스 목록 ({len(classes)}개):")
    for i, cls in enumerate(classes):
        print(f"  {i}: {cls}")

    # YOLO 및 COCO 형식 변환
    prepare_yolo_format(splits, classes)
    create_coco_annotations(splits, classes)

    # 2단계: YOLOv5 학습
    print_stage(2, "YOLOv5 학습 및 평가")
    yolo_model = train_yolo()
    yolo_metrics = evaluate_yolo(yolo_model, split='test')

    # 3단계: EfficientDet 학습
    print_stage(3, "EfficientDet 학습 및 평가")
    effdet_config = train_efficientdet(splits, classes)
    effdet_metrics = evaluate_efficientdet_coco(effdet_config, splits, classes)

    # 4단계: 결과 비교
    print_stage(4, "결과 비교 및 시각화")
    winner = print_results_table(yolo_metrics, effdet_metrics)
    visualize_comparison(yolo_metrics, effdet_metrics, " Performance Comparison")

    # 5단계: 최종 결과 저장
    print_stage(5, "최종 결과 저장")
    results_summary = {
        'dataset_info': {
            'num_train': len(splits['train']),
            'num_val': len(splits['val']),
            'num_test': len(splits['test']),
            'num_classes': len(classes),
            'classes': classes
        },
        'yolo_metrics': yolo_metrics,
        'efficientdet_metrics': effdet_metrics,
        'winner': winner
    }

    summary_path = RESULT_DIR / 'final_results.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 최종 결과 저장: {summary_path}")

    # 결과 요약
    print("\n" + "="*70)
    print("✓ 실행 완료!")
    print("="*70)
    print(f"\n📁 결과 위치: {RESULT_DIR.resolve()}")
    print(f"\n📊 생성된 파일:")
    print(f"  - YOLO 결과: yolov5_test/")
    print(f"  - EfficientDet 모델: efficientdet_best.pth")
    print(f"  - 비교 그래프: comparison_Performance_Comparison.png")
    print(f"  - Loss 곡선: efficientdet_loss_curve.png")
    print(f"  - 최종 결과: final_results.json")
    print(f"\n🏆 최종 승자: {winner}")
    print("="*70)


if __name__ == "__main__":
    main()