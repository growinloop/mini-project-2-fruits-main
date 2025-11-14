import torch
import torch.nn as nn
import math

# --- Mock Implementations (As requested) ---
# Backbone과 Neck은 이미 구현되어 있고, 그 '출력'만 있다고 가정합니다.
# 우리는 이 '출력'을 가짜 텐서(mock tensor)로 만들어 사용할 것입니다.
# ------------------------------------------

class Detect(nn.Module):
    """YOLOv5 감지 헤드 (Detection Head)"""
    
    # stride는 디코딩에 필요하지만, 모듈 정의 자체에서는 필수는 아님
    # (예: [8., 16., 32.])
    
    def __init__(self, nc=80, anchors=(), ch=()):
        """
        YOLOv5 감지 헤드를 초기화합니다.
        :param nc: int, 클래스 개수 (예: COCO는 80)
        :param anchors: list of lists, 각 감지 레이어의 앵커 박스
                       예: [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], ...]
        :param ch: list of int, Neck에서 들어오는 각 피처맵의 입력 채널
                   예: [128, 256, 512] (YOLOv5s 기준)
        """
        super().__init__()
        self.nc = nc  # 클래스 개수
        self.no = nc + 5  # 앵커당 출력 개수 (xywh + obj + classes)
        self.nl = len(anchors)  # 감지 레이어 개수 (보통 3개)
        self.na = len(anchors[0]) // 2  # 레이어당 앵커 개수 (보통 3개)
        
        # 'anchors'를 파라미터가 아닌 버퍼(buffer)로 등록합니다.
        # (학습되진 않지만 모델 state_dict에 저장됨)
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))

        # --- 헤드의 핵심 ---
        # 1x1 Conv 레이어를 담을 ModuleList
        # Neck에서 오는 3개의 피처맵 각각에 대해 1x1 Conv를 적용
        self.m = nn.ModuleList()
        for i in range(self.nl):
            # 입력 채널: ch[i] (Neck의 출력 채널)
            # 출력 채널: self.no * self.na
            # (예: 3 anchors * (80 classes + 5 outputs) = 3 * 85 = 255 채널)
            self.m.append(nn.Conv2d(ch[i], self.no * self.na, kernel_size=1))
            
        # 학습 안정성을 위한 편향(bias) 초기화
        self._initialize_biases()

    def _initialize_biases(self):
        # YOLOv5 리포지토리의 표준 편향 초기화 방법
        # obj와 cls 손실의 균형을 맞추기 위함
        for m in self.m:
            b = m.bias.view(self.na, -1)
            b.data[:, 4] += math.log(8 / (640 / 640) ** 2)  # obj bias
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.999))  # cls bias
            m.bias.data = b.view(-1)

    def forward(self, x):
        """
        헤드의 포워드 패스
        :param x: list of Tensors, Neck에서 온 피처맵 리스트
                  예: [ (bs, 128, 80, 80), (bs, 256, 40, 40), (bs, 512, 20, 20) ]
        :return: list of Tensors (학습 시)
        """
        outputs = []
        for i in range(self.nl):
            # 1. 1x1 컨볼루션을 적용
            #    입력: (bs, ch[i], grid_h, grid_w)
            #    출력: (bs, na * no, grid_h, grid_w)
            conv_out = self.m[i](x[i])
            
            # 2. 손실 계산 및 후처리를 위해 모양(shape) 변경
            bs, _, ny, nx = conv_out.shape
            # (bs, na * no, ny, nx) -> (bs, na, no, ny, nx) -> (bs, na, ny, nx, no)
            # 이 permute는 출력을 (배치, 앵커, 그리드y, 그리드x, 출력값) 순서로 정렬합니다.
            pred = conv_out.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            outputs.append(pred)

        # 학습(training) 시에는 이 'outputs' 리스트가 손실 함수로 전달됩니다.
        # 추론(inference) 시에는 이 'outputs'를 디코딩하고 NMS를 적용합니다.
        return outputs

# --- End of Head Implementation ---
"""

# --- ---------------------- ---
# ---     예제 사용법 (Test)     ---
# --- ---------------------- ---

# 1. 모델 파라미터 정의 (YOLOv5s 모델을 가정)
BATCH_SIZE = 4
NUM_CLASSES = 80  # COCO 클래스 개수

# Neck의 출력 채널 (Backbone + Neck을 거친 후)
NECK_CHANNELS = [128, 256, 512] 

# COCO 데이터셋 기준 앵커 (P3, P4, P5 앵커)
ANCHORS = [
    [10,13, 16,30, 33,23],      # P3/8 (작은 객체 감지용)
    [30,61, 62,45, 59,119],     # P4/16 (중간 객체 감지용)
    [116,90, 156,198, 373,326]  # P5/32 (큰 객체 감지용)
]

# 2. 가짜 입력 데이터 생성 (Mock Neck Output)
# 헤드(Detect) 모듈은 3개의 피처맵을 리스트로 입력받습니다.

# P3 출력 (stride 8), 80x80
mock_p3 = torch.randn(BATCH_SIZE, NECK_CHANNELS[0], 80, 80) # (4, 128, 80, 80)

# P4 출력 (stride 16), 40x40
mock_p4 = torch.randn(BATCH_SIZE, NECK_CHANNELS[1], 40, 40) # (4, 256, 40, 40)

# P5 출력 (stride 32), 20x20
mock_p5 = torch.randn(BATCH_SIZE, NECK_CHANNELS[2], 20, 20) # (4, 512, 20, 20)

# Neck의 최종 출력 (헤드의 입력)
mock_neck_output = [mock_p3, mock_p4, mock_p5]

print(f"--- 💡 헤드(Head)로 들어갈 입력 (Mock Neck Output) ---")
print(f"P3 피처맵 형상: {mock_neck_output[0].shape}")
print(f"P4 피처맵 형상: {mock_neck_output[1].shape}")
print(f"P5 피처맵 형상: {mock_neck_output[2].shape}")
print("-" * 50)


# 3. 헤드 모듈 생성 및 실행
# 위에서 정의한 파라미터로 Detect 헤드 모듈을 생성합니다.
yolo_head = Detect(nc=NUM_CLASSES, anchors=ANCHORS, ch=NECK_CHANNELS)

# 모델을 학습 모드(train)로 설정 (forward 결과가 리스트로 나옴)
yolo_head.train() 

# 가짜 Neck 출력을 헤드에 통과시킵니다.
predictions = yolo_head(mock_neck_output)


# 4. 헤드 출력 분석
# 'predictions'는 3개 텐서(P3, P4, P5 예측)를 담은 리스트입니다.
print(f"--- 🚀 헤드(Head)의 최종 출력 (Raw Predictions) ---")
print(f"총 출력 레이어 개수: {len(predictions)}")

# P3 예측 결과 형상: (bs, na, ny, nx, no)
# (4, 3, 80, 80, 85)
# 4  = 배치 크기 (BATCH_SIZE)
# 3  = 앵커 개수 (self.na)
# 80 = 그리드 높이 (ny)
# 80 = 그리드 너비 (nx)
# 85 = 출력 개수 (self.no) = 4(xywh) + 1(obj) + 80(classes)
print(f"P3 출력 형상: {predictions[0].shape}")

# P4 예측 결과 형상: (4, 3, 40, 40, 85)
print(f"P4 출력 형상: {predictions[1].shape}")

# P5 예측 결과 형상: (4, 3, 20, 20, 85)
print(f"P5 출력 형상: {predictions[2].shape}")

print("\n이 'predictions' 텐서가 학습 시 손실 함수(Loss Function)로 전달됩니다.")
"""