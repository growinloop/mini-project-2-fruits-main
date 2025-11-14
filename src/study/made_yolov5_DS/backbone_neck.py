# 파일명: my_backbone_and_neck.py
# (원래는 yolov5/models/common.py 에 있는 복잡한 모듈들입니다)

import torch
import torch.nn as nn

class Placeholder_Backbone(nn.Module):
    """
    Backbone을 단순하게 흉내 낸 모듈입니다.
    YOLOv5 백본은 3개의 다른 크기 피처맵을 넥(Neck)으로 전달합니다.
    (예: C3, C4, C5 레이어의 출력)
    """
    def __init__(self, c1=3, c_out1=64, c_out2=128, c_out3=256):
        super().__init__()
        # 실제로는 CSPDarknet 구조이지만, 여기서는 단순한 Conv로 대체합니다.
        # 설명을 위해 입력 채널을 3개(c_out1, c_out2, c_out3)로 가정합니다.
        self.layer1 = nn.Conv2d(c1, c_out1, 3, 2, 1)  # Stride 2 (크기 1/2)
        self.layer2 = nn.Conv2d(c_out1, c_out2, 3, 2, 1) # Stride 4 (크기 1/4)
        self.layer3 = nn.Conv2d(c_out2, c_out3, 3, 2, 1) # Stride 8 (크기 1/8)
        
    def forward(self, x):
        # 3개의 피처맵을 리스트로 반환 (실제 PANet의 입력 방식)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return [out1, out2, out3] # (설명을 위해 단순화. 실제론 C3,C4,C5)

class Placeholder_Neck(nn.Module):
    """
    Neck(PANet)을 단순하게 흉내 낸 모듈입니다.
    백본의 피처맵 3개를 입력받아, 헤드(Head)로 3개의 피처맵을 전달합니다.
    """
    def __init__(self, c_in1=64, c_in2=128, c_in3=256, c_out_head=[128, 256, 512]):
        super().__init__()
        # 실제로는 복잡한 Upsample, Concat, CSP 레이어가 있지만,
        # 여기서는 헤드로 나갈 채널(c_out_head)로 변경하는 1x1 Conv로 대체합니다.
        self.p3_out = nn.Conv2d(c_in1, c_out_head[0], 1)
        self.p4_out = nn.Conv2d(c_in2, c_out_head[1], 1)
        self.p5_out = nn.Conv2d(c_in3, c_out_head[2], 1)

    def forward(self, x):
        # x는 백본의 출력 [out1, out2, out3] 입니다.
        p3 = self.p3_out(x[0]) # (bs, 128, 80, 80)
        p4 = self.p4_out(x[1]) # (bs, 256, 40, 40)
        p5 = self.p5_out(x[2]) # (bs, 512, 20, 20)
        
        # 헤드로 3개의 피처맵 리스트를 전달
        return [p3, p4, p5]