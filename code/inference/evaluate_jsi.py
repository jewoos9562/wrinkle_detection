import numpy as np
import cv2
import os

def calculate_jsi(pred_mask_path, gt_mask_path):
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    # 이진화 (0과 1)
    _, pred_binary = cv2.threshold(pred_mask, 127, 1, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    jsi = intersection / union
    return jsi

def evaluate_jsi(pred_dir, gt_dir):
    jsi_scores = []
    
    for filename in os.listdir(gt_dir):
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        if not os.path.exists(pred_path):
            print(f"Prediction missing for {filename}")
            continue

        jsi = calculate_jsi(pred_path, gt_path)
        jsi_scores.append(jsi)

    mean_jsi = np.mean(jsi_scores) if jsi_scores else 0
    print(f"Mean JSI: {mean_jsi:.4f}")
    return mean_jsi

pred_mask_dir = "/root/skin/wrinkle/dataset/test/results/output_masks_2_640"  # 예측 마스크 저장 경로
gt_mask_dir = "/root/skin/wrinkle/dataset/test/GT"      # 정답 마스크 경로

evaluate_jsi(pred_mask_dir, gt_mask_dir)

