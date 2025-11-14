import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from effdet import create_model, get_efficientdet_config

# --------------------------------------------------------------------------
## 1. "데이터 준비 클래스" (가상 데이터셋)
# --------------------------------------------------------------------------
# 실제 학습을 위해서는 모델에 맞는 형식으로 이미지와 타겟(바운딩 박스, 클래스)을
# 전달해야 합니다. torch.utils.data.Dataset을 상속받아 만듭니다.
# --------------------------------------------------------------------------
class DummyDataset(Dataset):
    """
    EfficientDet 학습을 위한 가상의 데이터셋 클래스입니다.
    실제로는 이 부분에 이미지와 어노테이션을 로드하는 코드가 들어갑니다.
    """
    def __init__(self, num_samples, image_size):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        # 전체 데이터셋의 샘플 개수를 반환합니다.
        return self.num_samples

    def __getitem__(self, idx):
        """
        하나의 샘플(이미지 + 타겟)을 반환합니다.
        effdet의 'train' 모드는 'target' 딕셔너리를 필요로 합니다.
        """
        
        # 1. 가상 이미지 생성 (C, H, W)
        image = torch.rand(3, self.image_size, self.image_size)
        
        # 2. 가상 타겟(정답) 생성
        #    'bbox': [N, 4] (ymin, xmin, ymax, xmax) 포맷 (0~1 정규화된 좌표)
        #    'cls': [N] (클래스 인덱스)
        
        # 예시: 2개의 객체가 있다고 가정
        bboxes = torch.tensor([
            [0.1, 0.1, 0.3, 0.3],
            [0.5, 0.5, 0.7, 0.8]
        ], dtype=torch.float32)
        
        # 클래스 ID (여기서는 0번, 1번 클래스라고 가정)
        labels = torch.tensor([0, 1], dtype=torch.int64)
        
        # effdet이 요구하는 딕셔너리 포맷으로 타겟을 구성합니다.
        target = {
            'bbox': bboxes,
            'cls': labels
        }
        
        return image, target

# --------------------------------------------------------------------------
## 2. "모델 생성 및 실행" (메인 스크립트)
# --------------------------------------------------------------------------
def main():
    
    print("--- EfficientDet 데모 시작 ---")
    
    # --- 하이퍼파라미터 설정 ---
    MODEL_NAME = 'efficientdet_d0' # D0 모델 (EfficientNet-B0 백본)
    IMAGE_SIZE = 512                # D0의 기본 해상도
    NUM_CLASSES = 20                # 학습시킬 커스텀 클래스 개수 (COCO는 90)
    BATCH_SIZE = 4

    # -------------------------------------------------
    # A. "훈련(Train) 모드" 시연
    # -------------------------------------------------
    print(f"\n[A. 훈련 모드 시연 (bench_task='train')]")
    
    # 1. 모델 생성 (Train)
    #    bench_task='train': 학습 모드로 모델 생성. 
    #                        'forward'가 'loss'를 반환하도록 설정됨.
    #    pretrained=True: EfficientNet 백본은 ImageNet 가중치,
    #                     BiFPN/Head는 COCO 가중치를 가져옴.
    #                     (단, num_classes가 다르면 Head는 자동 리셋됨)
    print(f"'{MODEL_NAME}' 훈련용 모델 생성 중...")
    model_train = create_model(
        MODEL_NAME,
        bench_task='train',
        num_classes=NUM_CLASSES,
        pretrained=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE) # 입력 이미지 크기 명시
    )
    model_train.train() # 모델을 훈련 모드로 설정
    print("훈련용 모델 생성 완료.")

    # 2. 데이터로더 준비 (Train)
    #    'collate_fn'은 배치 내의 타겟들을 리스트로 묶어줍니다.
    dataset = DummyDataset(num_samples=10, image_size=IMAGE_SIZE)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)) # (img, target) 쌍을 분리
    )

    # 3. 훈련 1-Step 실행
    #    데이터로더에서 1배치 가져오기
    #    images: (B, C, H, W) 텐서
    #    targets: (B,) 크기의 튜플, 각 요소는 {'bbox': ..., 'cls': ...} 딕셔너리
    try:
        images, targets = next(iter(data_loader))
        
        # 이미지와 타겟을 모델에 함께 전달
        # bench_task='train' 모드에서는 모델이 (images, targets)를 인자로 받음
        outputs = model_train(images, targets)
        
        # 출력 확인: 'loss' 키에 전체 손실 값이 계산되어 나옴
        print(f"훈련 1-Step 실행 완료.")
        print(f"  - 입력 이미지 배치: {len(images)} x {images[0].shape}")
        print(f"  - 입력 타겟 배치: {len(targets)} (타겟 딕셔너리 튜플)")
        print(f"  - 모델 출력 (Loss 딕셔너리):")
        print(f"    - total loss: {outputs['loss']:.4f}")
        print(f"    - class loss: {outputs['class_loss']:.4f}")
        print(f"    - box loss: {outputs['box_loss']:.4f}")

    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")

    # -------------------------------------------------
    # B. "추론(Predict) 모드" 시연
    # -------------------------------------------------
    print(f"\n[B. 추론 모드 시연 (bench_task='predict')]")

    # 1. 모델 생성 (Predict)
    #    bench_task='predict': 추론 모드로 모델 생성.
    #                        'forward'가 NMS가 적용된 최종 박스들을 반환함.
    #    pretrained=True: COCO 사전 학습 가중치 전체 로드 (num_classes 변경 안 함)
    print(f"'{MODEL_NAME}' 추론용 모델 생성 중...")
    model_predict = create_model(
        MODEL_NAME,
        bench_task='predict',
        pretrained=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    model_predict.eval() # 모델을 추론 모드로 설정 (필수)
    print("추론용 모델 생성 완료.")

    # 2. 추론용 가상 이미지 생성
    #    (Batch, Channel, Height, Width)
    dummy_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    print(f"추론용 가상 입력 생성: {dummy_input.shape}")

    # 3. 추론 실행
    with torch.no_grad(): # 추론 시에는 gradient 계산 방지
        # bench_task='predict' 모드에서는 모델이 'images'만 인자로 받음
        detections = model_predict(dummy_input)

    # 4. 출력 확인
    #    detections: [Batch, Max_Detections, 6] 크기의 텐서
    #    각 Detection: [ymin, xmin, ymax, xmax, score, class_id]
    print(f"추론 1-Step 실행 완료.")
    print(f"  - 모델 출력 (Detections) Shape: {detections.shape}")
    
    # 0번 배치의 첫 번째 탐지 결과 (예시)
    first_detection = detections[0, 0]
    print(f"  - 0번 이미지의 첫 번째 탐지 결과:")
    print(f"    - Box: [ymin:{first_detection[0]:.2f}, xmin:{first_detection[1]:.2f}, ymax:{first_detection[2]:.2f}, xmax:{first_detection[3]:.2f}]")
    print(f"    - Score: {first_detection[4]:.4f}")
    print(f"    - Class ID: {first_detection[5].int()}")


if __name__ == "__main__":
    main()