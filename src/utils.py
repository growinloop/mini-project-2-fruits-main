from common_imports import *


def setup_korean_font():
    try:
        if os.name == 'nt':
            font_path = 'C:/Windows/Fonts/malgun.ttf'
            if os.path.exists(font_path):
                font_name = fm.FontProperties(fname=font_path).get_name()
                plt.rc('font', family=font_name)
            else:
                plt.rc('font', family='DejaVu Sans')
        elif os.name == 'posix':
            plt.rc('font', family='AppleGothic')

        plt.rcParams['axes.unicode_minus'] = False
        print("한글 폰트 설정 완료")
    except Exception as e:
        print(f"✗ 폰트 설정 실패: {e}")
        plt.rc('font', family='DejaVu Sans')


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def print_stage(stage_num, stage_name):
    print(f"【{stage_num}단계】 {stage_name}")


if __name__ == "__main__":
    setup_korean_font()
