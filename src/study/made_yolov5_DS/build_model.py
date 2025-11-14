# 파일명: build_model.py

import torch
import torch.nn as nn

# 1. 다른 파일에서 모듈 임포트
from backbone_neck import Placeholder_Backbone, Placeholder_Neck
from head import Detect

# --- 모델 파라미터 정의 (YOLOv5s 기준) ---
NUM_CLASSES = 80
ANCHORS = [
    [10,13, 16,30, 33,23],      # P3/8
    [30,61, 62,45, 59,119],     # P4/16
    [116,90, 156,198, 373,326]  # P5/32
]

# (Placeholder 모듈에 맞게 채널 단순화)
# 실제 YOLOv5s: 
#   Backbone 출력 채널: [256, 512, 1024]
#   Neck 출력 채널 (헤드 입력): [128, 256, 512]

# (이 예제 Placeholder 모듈 기준 채널)
BB_OUT_CH = [64, 128, 256]    # 백본 출력 채널
HEAD_IN_CH = [128, 256, 512] # 넥 출력 채널 (헤드 입력 채널)
# -----------------------------------


class CompleteYOLOv5Model(nn.Module):
    """
    Backbone, Neck, Head를 조립하는 최종 모델
    (실제 yolov5/models/yolo.py 의 Model 클래스와 유사한 구조)
    """
    def __init__(self, nc=80, anchors=()):
        super().__init__()
        
        # 1. 백본 모듈 생성 (임포트한 클래스 사용)
        self.backbone = Placeholder_Backbone(c_out1=BB_OUT_CH[0], 
                                            c_out2=BB_OUT_CH[1], 
                                            c_out3=BB_OUT_CH[2])
        
        # 2. 넥 모듈 생성 (임포트한 클래스 사용)
        self.neck = Placeholder_Neck(c_in1=BB_OUT_CH[0], 
                                     c_in2=BB_OUT_CH[1], 
                                     c_in3=BB_OUT_CH[2],
                                     c_out_head=HEAD_IN_CH)
        
        # 3. 헤드 모듈 생성 (임포트한 클래스 사용)
        self.head = Detect(nc=nc, anchors=anchors, ch=HEAD_IN_CH)

    def forward(self, x):
        """데이터의 흐름을 정의합니다: x -> Backbone -> Neck -> Head"""
        
        # 1. 입력 x가 백본을 통과
        # (출력: 3개 피처맵 리스트)
        backbone_features = self.backbone(x)
        
        # 2. 백본의 출력이 넥을 통과
        # (출력: 헤드용 3개 피처맵 리스트)
        neck_features = self.neck(backbone_features)
        
        # 3. 넥의 출력이 헤드를 통과
        # (출력: 최종 예측 텐서 리스트)
        predictions = self.head(neck_features)
        
        return predictions


# --- ---------------------- ---
# ---     최종 모델 테스트     ---
# --- ---------------------- ---

# 1. 최종 모델 인스턴스 생성
model = CompleteYOLOv5Model(nc=NUM_CLASSES, anchors=ANCHORS)
model.train() # 학습 모드

# 2. 가짜 입력 이미지 (Batch=4, Channel=3, H=640, W=640)
#    (Placeholder 백본이 640->80, 40, 20 으로 잘 줄이도록 입력 크기 조정)
mock_image = torch.randn(4, 3, 160, 160) 
# (입력 160 -> layer1(80) -> layer2(40) -> layer3(20))
# (넥 통과 후 -> 헤드 입력 (80, 40, 20) - 실제와 크기가 맞지 않지만 구조 테스트용)

# 3. 모델 실행 (forward)
final_predictions = model(mock_image)

print("--- 🚀 최종 조립 모델 실행 완료 ---")
print(f"헤드 출력 레이어 개수: {len(final_predictions)}")
print(f"P3 예측 형상: {final_predictions[0].shape}") # (4, 3, 20, 20, 85)
print(f"P4 예측 형상: {final_predictions[1].shape}") # (4, 3, 10, 10, 85)
print(f"P5 예측 형상: {final_predictions[2].shape}") # (4, 3, 5, 5, 85)
# (참고: 입력 크기와 Placeholder 모듈 정의에 따라 크기가 결정됨)