#
# 파일명: backbone_neck.py
#
import torch
import torch.nn as nn
from effdet import create_model

class BackboneNeck(nn.Module):
    """
    Ross Wightman의 effdet 라이브러리를 사용하여
    EfficientNet 백본과 BiFPN 넥을 담당하는 모듈입니다.
    """
    def __init__(self, model_name='efficientdet_d0', pretrained=True):
        super().__init__()
        
        print(f"'{model_name}'의 백본(EfficientNet)과 넥(BiFPN) 로드 중...")
        
        # 1. effdet 모델 전체를 생성합니다. (bench_task='train'은 NMS 없는 원시 출력을 의미)
        #    이 모델 인스턴스에서 필요한 'backbone'과 'fpn' 모듈만 가져와 사용할 것입니다.
        #    'pretrained=True'는 백본과 BiFPN 가중치를 모두 로드합니다.
        full_model_wrapper = create_model(
            model_name,
            bench_task='train', # 'train' 모드는 헤드 교체 및 커스텀에 용이
            pretrained=True
        )
        
        # 래퍼(full_model_wrapper)가 아닌,
        # 내부의 실제 모델(.model)에 접근해야 합니다.
        core_model = full_model_wrapper.model
        
        # 2. 필요한 모듈만 클래스 멤버로 저장
        # (1) 백본 (예: EfficientNet-B0)
        self.backbone = core_model.backbone
        
        # (2) 넥 (BiFPN)
        self.fpn = core_model.fpn
        
        # 3. BiFPN이 출력하는 피처 맵의 채널 수를 저장합니다. (헤드 구현 시 필요)
        #    D0 모델의 경우 기본 64입니다.
        self.fpn_channels = core_model.config.fpn_channels
        
        print(f"로드 완료. BiFPN 출력 채널: {self.fpn_channels}")

    def forward(self, x):
        """
        입력 이미지를 받아 백본과 BiFPN을 순차적으로 통과시킨
        피처 맵 리스트(P3, P4, P5, P6, P7)를 반환합니다.
        """
        
        # 1. 백본 통과 (P1 ~ P5 피처 맵 생성)
        features = self.backbone(x)
        
        # 2. BiFPN 통과 (P3 ~ P7 피처 맵 생성)
        #    fpn 모듈은 P3~P5 피처맵(features[2:])을 입력으로 받아
        #    내부적으로 P6, P7을 만들고 융합(fusion)을 수행합니다.
        fpn_features = self.fpn(features) # P3, P4, P5 입력
        
        # fpn_features는 P3, P4, P5, P6, P7에 해당하는 5개의 텐서 리스트입니다.
        return fpn_features