import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    """
    EfficientDet 헤드에서 사용하는 Depthwise Separable Convolution 구현
    (연산량 감소를 위해 사용됨)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class CustomHead(nn.Module):
    """
    EfficientDet의 Class/Box 예측 헤드를 직접 구현합니다.
    BiFPN에서 나온 모든 레벨의 피처 맵에 대해 파라미터를 공유합니다.
    """
    def __init__(self, in_channels, num_classes, head_depth=3, num_anchors=9):
        """
        in_channels: BiFPN 피처 맵의 채널 수 (예: D0=64)
        num_classes: 최종 분류할 클래스 개수 (예: COCO=80)
        head_depth: 헤드 네트워크의 반복 횟수 (예: D0=3)
        num_anchors: 각 위치에서 예측할 앵커 개수 (기본 9)
        """
        super(CustomHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 1. Class 예측 헤드 (분류용)
        cls_layers = []
        for _ in range(head_depth):
            cls_layers.append(SeparableConv2d(in_channels, in_channels))
            cls_layers.append(nn.BatchNorm2d(in_channels))
            cls_layers.append(nn.SiLU()) # SiLU (Swish) 활성화 함수
        
        # 최종 분류 예측 레이어 (클래스 개수 * 앵커 개수 만큼 출력)
        cls_layers.append(
            SeparableConv2d(in_channels, num_classes * num_anchors, 3, 1, 1)
        )
        self.cls_head = nn.Sequential(*cls_layers)

        # 2. Box 예측 헤드 (회귀용)
        box_layers = []
        for _ in range(head_depth):
            box_layers.append(SeparableConv2d(in_channels, in_channels))
            box_layers.append(nn.BatchNorm2d(in_channels))
            box_layers.append(nn.SiLU())
            
        # 최종 박스 예측 레이어 (좌표 4개 * 앵커 개수 만큼 출력)
        box_layers.append(
            SeparableConv2d(in_channels, 4 * num_anchors, 3, 1, 1)
        )
        self.box_head = nn.Sequential(*box_layers)

    def forward(self, fpn_features):
        """
        fpn_features: [P3, P4, P5, P6, P7] 피처 맵 리스트
        """
        cls_outputs = []
        box_outputs = []
        
        # 모든 FPN 레벨(P3~P7)에 대해 동일한 헤드(가중치 공유)를 적용
        for feat in fpn_features:
            cls_pred = self.cls_head(feat) # [B, C_cls, H, W]
            box_pred = self.box_head(feat) # [B, C_box, H, W]
            
            # (나중에 Loss 계산을 위해 [B, H, W, C] 형태로 변형하기도 함)
            cls_outputs.append(cls_pred)
            box_outputs.append(box_pred)
            
        # [P3_cls, P4_cls, ...] , [P3_box, P4_box, ...]
        return cls_outputs, box_outputs