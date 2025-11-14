import torch
import torch.nn as nn

# 1. 우리가 만든 모듈 임포트
from backbone_neck import BackboneNeck
from custom_head import CustomHead

class MyEfficientDet(nn.Module):
    """
    Backbone/Neck 모듈과 Custom Head 모듈을 조립한
    최종 EfficientDet 모델
    """
    def __init__(self, model_name='efficientdet_d0', num_classes=80):
        super().__init__()
        
        # 1. 백본 + 넥 생성
        self.backbone_neck = BackboneNeck(model_name=model_name)
        
        # 2. 헤드 생성
        #    BackboneNeck에서 FPN 채널 정보를 가져와 헤드에 전달
        fpn_channels = self.backbone_neck.fpn_channels
        self.head = CustomHead(
            in_channels=fpn_channels,
            num_classes=num_classes
        )
        
    def forward(self, x):
        # 1. 백본 + 넥 통과 -> [P3, P4, P5, P6, P7] 피처 맵 리스트 반환
        fpn_features = self.backbone_neck(x)
        
        # 2. 커스텀 헤드 통과 -> 예측 결과 반환
        cls_outputs, box_outputs = self.head(fpn_features)
        
        return cls_outputs, box_outputs

# --------------------------
# 메인 실행 블록
# --------------------------
if __name__ == "__main__":
    
    # --- 설정 ---
    IMG_SIZE = 512      # D0 모델의 기본 입력 크기
    NUM_CLASSES = 20    # 학습시킬 커스텀 클래스 개수
    BATCH_SIZE = 2
    
    print("--- 1. 커스텀 EfficientDet 모델 생성 ---")
    model = MyEfficientDet(
        model_name='efficientdet_d0', 
        num_classes=NUM_CLASSES
    )
    model.eval() # 추론 모드 (BatchNorm 등에 영향)
    
    # --- 2. 가상 입력 데이터 생성 ---
    # (B, C, H, W)
    dummy_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    print(f"\n--- 2. 입력 데이터 Shape ---")
    print(f"Input: {dummy_input.shape}")
    
    # --- 3. 모델 추론 실행 ---
    with torch.no_grad():
        cls_preds, box_preds = model(dummy_input)
        
    print(f"\n--- 3. 모델 출력 결과 확인 ---")
    print(f"Class 예측 개수 (P3~P7): {len(cls_preds)}")
    print(f"Box 예측 개수 (P3~P7): {len(box_preds)}")
    
    print("\n각 레벨별 예측 Shape (Batch, Channels, Height, Width):")
    for i in range(len(cls_preds)):
        print(f"  [P{i+3}]")
        print(f"    Class: {cls_preds[i].shape}") # [B, 9*NUM_CLASSES, H, W]
        print(f"    Box:   {box_preds[i].shape}") # [B, 9*4, H, W]