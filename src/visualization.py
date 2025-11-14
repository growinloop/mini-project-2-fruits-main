from common_imports import *
from config import *


def visualize_comparison(yolo_metrics, effdet_metrics, title_suffix=""):
    metrics_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    yolo_values = [
        yolo_metrics['mAP50'],
        yolo_metrics['mAP50_95'],
        yolo_metrics['precision'],
        yolo_metrics['recall']
    ]
    effdet_values = [
        effdet_metrics['mAP50'],
        effdet_metrics['mAP50_95'],
        effdet_metrics['precision'],
        effdet_metrics['recall']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, yolo_values, width, label='YOLOv5', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, effdet_values, width, label='EfficientDet', color='#4ECDC4')

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'YOLOv5 vs EfficientDet{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    filename = f'comparison{title_suffix.replace(" ", "_")}.png'
    save_path = RESULT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"비교 그래프 저장: {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, normalized=True):
    extended_classes = classes + ['No Detection']
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(extended_classes))))

    plt.figure(figsize=(max(12, len(extended_classes) * 0.8),
                        max(10, len(extended_classes) * 0.7)))

    if normalized:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = '.2f'
        title = 'Confusion Matrix (Normalized)'
        filename = 'confusion_matrix_normalized.png'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix (Count)'
        filename = 'confusion_matrix_count.png'

    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=extended_classes, yticklabels=extended_classes)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = RESULT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion Matrix 저장: {save_path}")

    return cm


def plot_per_class_accuracy(class_accuracies):
    if not class_accuracies:
        return

    plt.figure(figsize=(12, 6))
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    class_names = [x[0] for x in sorted_classes]
    accuracies = [x[1] for x in sorted_classes]

    bars = plt.bar(range(len(class_names)), accuracies, color='skyblue', edgecolor='navy')
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = RESULT_DIR / 'per_class_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" 클래스별 정확도 저장: {save_path}")


def print_results_table(yolo_metrics, effdet_metrics):
    print(f"{'최종 성능 비교':^70}")
    print(f"\\n{'Metric':<20} {'YOLOv5':<15} {'EfficientDet':<15} {'Difference':<15}")

    for metric, key in [('mAP@0.5', 'mAP50'),
                        ('mAP@0.5:0.95', 'mAP50_95'),
                        ('Precision', 'precision'),
                        ('Recall', 'recall')]:
        y_val = yolo_metrics[key]
        e_val = effdet_metrics[key]
        diff = y_val - e_val
        print(f"{metric:<20} {y_val:<15.3f} {e_val:<15.3f} {diff:+.3f}")

    if yolo_metrics['mAP50'] > effdet_metrics['mAP50']:
        winner = "YOLOv5"
        diff = yolo_metrics['mAP50'] - effdet_metrics['mAP50']
    elif effdet_metrics['mAP50'] > yolo_metrics['mAP50']:
        winner = "EfficientDet"
        diff = effdet_metrics['mAP50'] - yolo_metrics['mAP50']
    else:
        winner = "동점"
        diff = 0

    if winner != "동점":
        print(f" {winner}가 {diff:.3f}만큼 더 높은 mAP@0.5를 달성!")
    else:
        print(f" 두 모델이 동일한 성능!")

    return winner


if __name__ == "__main__":
    yolo_metrics = {'mAP50': 0.95, 'mAP50_95': 0.90, 'precision': 0.92, 'recall': 0.93}
    effdet_metrics = {'mAP50': 0.88, 'mAP50_95': 0.85, 'precision': 0.87, 'recall': 0.89}

    visualize_comparison(yolo_metrics, effdet_metrics, " Performance")
    print_results_table(yolo_metrics, effdet_metrics)