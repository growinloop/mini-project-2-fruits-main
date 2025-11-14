# 🍎 과일 품질 등급 분류 시스템 - YOLOv5 vs EfficientDet 비교 분석

---
## 🧑‍🤝‍🧑 팀명: **Fruits**


### 👑 팀장  
- 한대성  

### 🌱 팀원  
- 김소영  
- 이주형
  
### 🗓 프로젝트 기간  
**2025년 11월 3일 ~ 2025년 11월 14일**

---

## 📋 프로젝트 개요

### 주제
**과일 품질 등급 분류를 통한 소비자 의사결정 시스템**

본 프로젝트는 **객체 탐지 기반의 과일 등급 자동 분류 시스템**을 구축하고, 산업에서 가장 인기 있는 두 모델(**YOLOv5** vs **EfficientDet**)의 성능을 직접 비교 분석합니다.

### 📚 학술적 배경
- **논문 기반**: 객체 탐지 최신 논문들을 조사한 결과, **YOLOv5**와 **EfficientDet**이 가장 광범위하게 인용되는 모델
- **One-shot 분류**: 단순히 과일의 "형태", "색상", "질감"을 **분리하지 않고** 통합적으로 인식하여 **"신선한 사과_특상"** 등 하나의 카테고리로 직접 분류
- **논문 검증**: 기존 논문의 성능 비교 결과를 실제 데이터로 재현하고 검증

### 🎯 핵심 목표
1. **모델 성능 비교**: YOLOv5 vs EfficientDet 정확도, 속도, 효율성 분석
2. **과일 품질 등급 자동화**: 특상/상/중 3단계 등급의 신뢰성 있는 분류
3. **실무 적용 가능성**: 소매 및 수입검사 시스템 구축을 위한 기초 연구
4. **고민 과정 공유**: 프로젝트 탐색과정에서의 문제해결 능력 시연

---

## 📁 프로젝트 구조

```
mini-project-2-fruits/
├── data/
│   ├── raw/                          # 원본 데이터
│   │   ├── images/                   # 과일 이미지
│   │   └── json_labels/              # 바운딩박스 레이블 (JSON 형식)
│   └── test_data/                    # 별도 테스트 데이터셋
│       ├── images/
│       └── json_labels/
├── document/                         # 프로젝트 자료
├── processed/                        # 로컬 전용(.gitignore 등록됨)
│   ├── preprocessed_data/
│   │   ├── yolov5/                   # YOLOv5 포맷 변환 데이터
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   │   └── data.yaml
│   │   └── efficientdet/             # EfficientDet 포맷 데이터
│   │       └── coco_*.json
│   └── results_comparison/           # 학습 결과 및 평가 지표
│       ├── yolov5su.pt               # YOLOv5 사전학습 모델
│       ├── efficientdet_best.pth     # EfficientDet 최고 성능 체크포인트
│       └── *.json, *.png             # 메트릭 및 시각화
└── src/
    ├── study/                        # 학습용 기타 파일들
    ├── config.py                     # 설정 및 경로
    ├── utils.py                      # 유틸리티 함수
    ├── data_preprocessing.py         # 데이터 전처리
    ├── dataset.py                    # 데이터셋 클래스
    ├── yolo_trainer.py               # YOLO 학습/평가
    ├── efficientdet_trainer.py       # EfficientDet 학습/평가
    ├── visualization.py              # 시각화
    ├── main.py                       # 메인 실행 파일 (Python)
    └── yolov5_efficientdet_comb.ipynb  # 메인 실행 파일 (Jupyter)
```

---

## 📊 데이터셋 정보

### 클래스 구성
- **사과 (Apple Fuji)**: 특상, 상, 중 - 상품 등급
- **배 (Pear Chuhwang)**: 특상, 상, 중 - 상품 등급
- **감 (persimmon booyu)**: 특상, 상, 중 - 상품 등급
- **총 클래스**: 약 9개

### 데이터 분할
- **학습 데이터**: 원본 데이터의 80%
- **검증 데이터**: 원본 데이터의 20% (80%의 50%)
- **테스트 데이터**: 별도의 독립적인 테스트 데이터셋

각 샘플은 다음 정보를 포함합니다:
- 이미지 파일 (JPG, PNG, JPEG)
- 바운딩박스 좌표: `[xmin, ymin, xmax, ymax]`
- 카테고리 정보: `cate1` (과일 종류), `cate3` (크기 등급)

---

## 🚀 시작하기

### 필수 라이브러리 설치

```bash
# 핵심 딥러닝 프레임워크
pip install torch torchvision

# 객체 탐지 모델
pip install ultralytics      # YOLOv5
pip install effdet timm       # EfficientDet

# 데이터 처리 및 평가
pip install pycocotools
pip install opencv-python
pip install scikit-learn

# 시각화
pip install matplotlib seaborn

# 기타 유틸리티
pip install numpy pandas tqdm pyyaml
```

---

## 💻 실행 방법

### 방법 1️⃣: Python 스크립트 실행 (권장)

모듈화된 Python 파일을 사용하여 전체 파이프라인을 실행합니다.

#### 1-1. 전체 파이프라인 자동 실행
```bash
cd src
python main.py
```

이 명령은 다음 작업을 자동으로 수행합니다:
1. 데이터 전처리 및 분할
2. YOLOv5 학습 및 평가
3. EfficientDet 학습 및 평가
4. 성능 비교 시각화

#### 1-2. 개별 모듈 실행

특정 단계만 실행하고 싶을 때:

```bash
# 1. 데이터 전처리만 실행
python data_preprocessing.py

# 2. YOLOv5만 학습/평가
python yolo_trainer.py

# 3. EfficientDet만 학습/평가
python efficientdet_trainer.py

# 4. 시각화만 실행
python visualization.py
```

#### 1-3. 설정 변경

`config.py` 파일에서 하이퍼파라미터 조정:

```python
TRAIN_CONFIG = {
    'epochs': 100,              # 학습 에포크
    'yolo_batch_size': 16,      # YOLO 배치 크기
    'effdet_batch_size': 4,     # EfficientDet 배치 크기
    'patience': 30,             # Early stopping patience
    'learning_rate': 0.001,
    'image_size_yolo': 640,
    'image_size_effdet': 512,
}
```

#### 1-4. 코드에서 직접 호출

```python
from yolo_trainer import train_yolo, evaluate_yolo
from efficientdet_trainer import train_efficientdet, evaluate_efficientdet

# YOLOv5만 재학습
model = train_yolo(epochs=50)
metrics = evaluate_yolo(model)

# EfficientDet만 재학습
effdet_model = train_efficientdet(epochs=50)
effdet_metrics = evaluate_efficientdet(effdet_model)
```

### 방법 2️⃣: Jupyter Notebook 실행 (대화형)

단계별로 결과를 확인하며 실행하고 싶을 때 사용합니다.

#### 2-1. 노트북 열기
```bash
cd src
jupyter notebook yolov5_efficientdet_comb.ipynb
```

#### 2-2. 실행 방식

**전체 실행:**
```python
# 노트북 마지막 셀에서
if __name__ == "__main__":
    main()
```

**단계별 실행:**
- **셀 1-3**: 라이브러리 임포트 및 경로 설정
- **셀 4-5**: 데이터 전처리
- **셀 6-7**: YOLOv5 학습 및 평가
- **셀 8-11**: EfficientDet 학습 및 평가
- **셀 12-13**: 성능 비교 시각화

각 셀을 개별적으로 실행하여 중간 결과를 확인할 수 있습니다.

---

## 🔧 주요 기능

### 1. 데이터 전처리
- JSON 레이블 파일 자동 파싱
- 이미지-레이블 매칭
- Train/Val/Test 자동 분할 (8:1:1)
- 바운딩박스 정규화
- 멀티 포맷 지원: YOLO 및 COCO 형식 자동 변환

### 2. YOLOv5 모델
- **입력**: YOLO 형식 데이터셋 (정규화된 바운딩박스)
- **학습 설정**:
  - 배치 크기: 16
  - 이미지 크기: 640×640
  - Epochs: 기본 100 (조정 가능)
  - Early Stopping: patience=30
- **평가 지표**: mAP@0.5, mAP@0.5:0.95, Precision, Recall

### 3. EfficientDet 모델
- **아키텍처**: EfficientDet-D0 (사전학습 백본)
- **입력**: COCO 형식 어노테이션 + 이미지
- **학습 설정**:
  - 배치 크기: 4
  - 이미지 크기: 512×512
  - Epochs: 기본 100 (조정 가능)
  - Early Stopping: patience=30
  - Optimizer: AdamW (lr=0.01)
  - Scheduler: CosineAnnealingLR
- **평가 방법**:
  - COCO 평가 (가능 시)
  - 단순 IoU 기반 평가 (COCO 없을 시)
  - 혼동 행렬 분석

### 4. 실시간 모니터링
- 학습 진행 상황 실시간 표시
- 손실 곡선 자동 생성
- 다양한 지표로 자동 평가

### 5. 자동 시각화
- 혼동 행렬 (정규화 + 개수)
- 클래스별 정확도 막대 차트
- 성능 비교 그래프
- Classification Report

---

## 📈 결과 확인

### 생성되는 파일 위치
결과는 `processed/results_comparison/` 폴더에 저장됩니다.

```
processed/results_comparison/
├── final_results.json                   # 📊 최종 성능 비교 결과
│
├── yolo_metrics.json                    # YOLOv5 성능 지표
├── efficientdet_metrics.json            # EfficientDet 성능 지표
├── final_test_results.json              # 최종 종합 결과
│
├── performance_comparison_test.png      # 📈 성능 비교 그래프
├── final_comparison_graph.png           # 최종 비교 그래프
├── test_summary_graph.png               # 요약 그래프
│
├── efficientdet_confusion_matrix_normalized.png    # 정규화 혼동 행렬
├── efficientdet_confusion_matrix_count.png         # 개수 혼동 행렬
├── efficientdet_confusion_matrix.json              # 혼동 행렬 데이터
├── efficientdet_classification_report.txt          # 분류 리포트
├── efficientdet_per_class_accuracy.png             # 클래스별 정확도
├── efficientdet_loss_curve.png                     # 학습 손실 곡선
│
├── efficientdet_best.pth                # EfficientDet 체크포인트
└── yolov5su.pt                          # YOLOv5 사전학습 가중치
```

### 성능 지표 형식

```json
{
  "summary": {
    "mAP50": 0.85,           // 50% IoU 기준 평균 정확도
    "mAP50_95": 0.65,        // 50-95% IoU 범위 평균 정확도
    "precision": 0.88,       // 정밀도
    "recall": 0.82           // 재현율
  },
  "overall_accuracy": 0.90,  // 전체 정확도 (EfficientDet)
  "class_accuracies": {      // 클래스별 정확도
    "apple_fuji_L": 0.92,
    "apple_fuji_M": 0.89,
    ...
  }
}
```

---

## 🔄 파이프라인 흐름

```
1. 데이터 로드 (JSON + 이미지)
    ↓
2. Train/Val/Test 분할 (8:1:1)
    ↓
3. 데이터 형식 변환
   ├─→ YOLO 형식 (.txt)
   └─→ COCO 형식 (.json)
    ↓
┌─→ YOLOv5 학습 (100 epochs) ──→ YOLOv5 테스트
│                                    ↓
│                              mAP, Precision, Recall
│
├→ EfficientDet 학습 (100 epochs) → EfficientDet 테스트
│                                    ↓
│                              혼동 행렬, 클래스별 정확도
│
└─── 성능 비교 시각화 ──→ 최종 결과 저장 (JSON + PNG)
```

---

## ⚙️ 커스터마이징

### Python 방식
각 모듈은 독립적으로 실행 가능하므로 필요한 부분만 수정하여 사용할 수 있습니다.

```python
# config.py에서 설정 변경
TRAIN_CONFIG['epochs'] = 50
TRAIN_CONFIG['yolo_batch_size'] = 8

# 또는 함수 호출 시 직접 지정
from yolo_trainer import train_yolo

model = train_yolo(
    data_yaml='path/to/data.yaml',
    epochs=50,
    batch_size=8,
    img_size=640
)
```

### Jupyter Notebook 방식
노트북에서 셀 단위로 파라미터를 수정하여 실행:

```python
# YOLOv5 학습 설정
epochs = 50  # ← 변경
batch_size = 8

# EfficientDet 학습 설정
effdet_epochs = 100
effdet_batch_size = 4
```

### 이미지 크기 설정
- **YOLOv5**: 640×640 (권장값, config.py에서 수정 가능)
- **EfficientDet**: 512×512 (dataset.py의 EffDetDataset에서 수정)

### 신뢰도 임계값
```python
# efficientdet_trainer.py 내부
confidence_threshold = 0.3  # 탐지 신뢰도 기준
iou_threshold = 0.5         # IoU 매칭 기준
```

---

## 🐛 트러블슈팅

### 1. `pycocotools` 없음 경고
```
⚠️ Warning: pycocotools 없음
```
**해결책**: 
```bash
pip install pycocotools
```
- 설치 실패 시 COCO 평가는 건너뛰고 단순 IoU 기반 평가로 진행됩니다.

### 2. CUDA 메모리 부족
**증상**: `RuntimeError: CUDA out of memory`

**해결책**: config.py에서 배치 크기 감소
```python
TRAIN_CONFIG = {
    'yolo_batch_size': 8,      # 기본값 16 → 8
    'effdet_batch_size': 2,    # 기본값 4 → 2
}
```

### 3. 이미지 파일을 찾을 수 없음
**원인**: JSON 파일의 `stem`과 실제 이미지 파일명 불일치

**확인 사항**:
- JSON 레이블의 파일명과 이미지 파일명이 일치하는지
- 지원되는 형식: `.jpg`, `.png`, `.jpeg` (대소문자 구분 없음)

### 4. 한글 폰트 설정 실패
**해결책**:
- **Windows**: `C:/Windows/Fonts/malgun.ttf` 존재 확인
- **Mac**: `AppleGothic` 자동 사용
- **Linux**: 
  ```bash
  sudo apt-get install fonts-nanum
  ```

### 5. 모듈을 찾을 수 없음 (ModuleNotFoundError)
**원인**: Python 경로 문제

**해결책**:
```bash
# src 폴더에서 실행
cd src
python main.py

# 또는 PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

---

## 📝 코드 구조 설명 (Python 버전)

### 주요 파일 및 역할

| 파일명 | 역할 | 주요 함수 |
|--------|------|-----------|
| `config.py` | 설정 및 경로 관리 | 경로 상수, 하이퍼파라미터 |
| `utils.py` | 공통 유틸리티 | 파일 처리, 로깅 |
| `data_preprocessing.py` | 데이터 전처리 | `preprocess_data()` |
| `dataset.py` | 데이터셋 클래스 | `EffDetDataset` |
| `yolo_trainer.py` | YOLO 학습/평가 | `train_yolo()`, `test_yolo()` |
| `efficientdet_trainer.py` | EfficientDet 학습/평가 | `train_efficientdet()`, `test_efficientdet()` |
| `visualization.py` | 시각화 | `visualize_comparison()` |
| `main.py` | 메인 실행 | `main()` |

### 주요 함수

| 함수명 | 목적 | 입력 | 출력 |
|--------|------|------|------|
| `preprocess_data()` | 데이터 로드 및 분할 | JSON_DIR | `splits`, `classes` |
| `prepare_yolo_dataset()` | YOLO 형식 변환 | `splits`, `classes` | YOLO 디렉토리 구조 |
| `EffDetDataset` | PyTorch 데이터셋 | 이미지 경로, 바운딩박스 | 텐서 포맷 데이터 |
| `train_yolo()` | YOLOv5 학습 | YAML 설정, epochs | 학습된 모델 |
| `train_efficientdet()` | EfficientDet 학습 | `splits`, `classes`, epochs | 모델, config |
| `test_yolo()` | YOLOv5 평가 | 모델, YAML 설정 | 성능 지표 dict |
| `test_efficientdet()` | EfficientDet 평가 | config, `splits`, `classes` | 성능 지표 dict |
| `evaluate_efficientdet_with_confusion_matrix()` | 혼동 행렬 분석 | config, `splits`, `classes`, device | 혼동 행렬, 클래스별 정확도 |
| `visualize_comparison()` | 성능 비교 시각화 | 두 모델의 지표 | PNG 그래프 |

---

## 📚 데이터 형식

### JSON 레이블 형식
```json
{
  "cate1": "apple",           // 과일 종류
  "cate3": "fuji_L",         // 품종 및 크기
  "bndbox": {
    "xmin": 100,
    "ymin": 150,
    "xmax": 300,
    "ymax": 350
  }
}
```

### YOLO 레이블 형식 (.txt)
```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
0 0.5 0.5 0.3 0.3
```

### COCO 어노테이션 형식
```json
{
  "images": [
    {
      "id": 0,
      "file_name": "image_001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [100, 150, 200, 200]  // [x, y, width, height]
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "apple_fuji_L"
    }
  ]
}
```

---

## 📄 추가 문서 자료

프로젝트의 분석 설계, 분석 결과 등은 [`document/`](./document/) 폴더에 정리되어 있습니다.
마크다운 형태가 깨지는 현상이 있으니 저장된 문서를 다운해서 보시면 됩니다.

- 주요 문서:
    - [일일 보고서](./document/01_DailyReport.md)
    - [분석 계획 보고서](./document/02_AnalysisPlanReport.md)
    - [코딩 컨벤션](./document/03_CodingConvention.md)
    - [데이터 명세서](./document/04_DataStatement.md)
    - [분석 결과 보고서](./document/05_AnalysisResultsReport.md)

---

## 📚 참고 자료

- **YOLOv5**: https://github.com/ultralytics/yolov5
- **EfficientDet**: https://github.com/rwightman/efficientdet-pytorch
- **COCO 평가**: https://github.com/cocodataset/cocoapi
- **Ultralytics 문서**: https://docs.ultralytics.com/

---

## 👨‍💻 개발자 노트

### 알려진 한계사항
1. **EfficientDet 데이터 로더**의 `collate_fn` - 가변 크기 박스 처리 시 패딩 사용
2. **바운딩박스 IoU 기반 매칭** - 단일 클래스 예측 로직만 구현 (다중 객체 미지원)
3. **COCO 평가** - `pycocotools` 미설치 시 대체 평가 방식 사용

### 향후 개선 방향
- [ ] Multi-box detection 지원 (한 이미지 내 여러 객체 동시 탐지)
- [ ] 앙상블 모델 추가 (YOLOv5 + EfficientDet)
- [ ] 실시간 추론 최적화 (TensorRT, ONNX)
- [ ] 모바일 환경 배포 (TensorFlow Lite, PyTorch Mobile)
- [ ] 웹 기반 데모 인터페이스 (Gradio, Streamlit)

### Python vs Jupyter Notebook
- **Python**: 자동화된 배치 실행, 재현성, 프로덕션 환경에 적합
- **Jupyter Notebook**: 탐색적 분석, 교육용, 단계별 디버깅에 적합

---

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

---

**작성일**: 2025년 11월 12일  
**마지막 수정**: 2025년 11월 13일  
**작성자**: 한대성

![image](https://github.com/user-attachments/assets/a5ce47b5-7171-4695-800a-c411db03fcf5)
