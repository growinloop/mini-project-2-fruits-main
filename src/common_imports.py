# Standard Library
import os
import json
from pathlib import Path

# Data Processing
import cv2
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#  Visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print(" YOLOv5 not installed")
    YOLO_AVAILABLE = False

#  EfficientDet
try:
    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
    EFFDET_AVAILABLE = True
except ImportError:
    print(" EfficientDet not installed")
    EFFDET_AVAILABLE = False

# COCO API
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_AVAILABLE = True
except ImportError:
    print(" pycocotools not installed")
    COCO_AVAILABLE = False

#  Export All
__all__ = [
    # Standard
    'os', 'json', 'Path',
    # Data
    'cv2', 'np', 'yaml', 'tqdm', 'train_test_split',
    'confusion_matrix', 'classification_report',
    # PyTorch
    'torch', 'nn', 'Dataset', 'DataLoader',
    # Visualization
    'plt', 'fm', 'sns',
    # Models
    'YOLO', 'get_efficientdet_config', 'EfficientDet',
    'DetBenchTrain', 'DetBenchPredict',
    'COCO', 'COCOeval',
    # Flags
    'YOLO_AVAILABLE', 'EFFDET_AVAILABLE', 'COCO_AVAILABLE'
]