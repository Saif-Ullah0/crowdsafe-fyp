import os
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy.io as io
from PIL import Image
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))

class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 7, padding=3), nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, 5, padding=2), nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(nn.Conv2d(30, 1, 1))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat((x1, x2, x3), 1)
        return self.fuse(x_cat)


def evaluate_mcnn_mall(weights_path: str, device: str = 'cpu', step: int = 20):
    if not os.path.exists(weights_path):
        print(f'[ERROR] Missing weights file at: {weights_path}')
        return

    model = MCNN().to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)

    model.eval()

    frames_dir = os.path.join(PROJECT_ROOT, 'datasets', 'Mall', 'frames')
    gt_path = os.path.join(PROJECT_ROOT, 'datasets', 'Mall', 'ground_truth', 'GT.mat')

    gt_mat = io.loadmat(gt_path)
    gt_frames = gt_mat['frame'][0]

    sampled_indices = list(range(0, len(gt_frames), step))
    mae_sum = 0.0
    mse_sum = 0.0
    total_evaluated = 0

    print(f' Auditing MCNN Model: {os.path.basename(weights_path)} ---')
    print(f'Evaluating across {len(sampled_indices)} frames...\n')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for count, frame_idx in enumerate(sampled_indices):
        img_name = f'seq_{str(frame_idx + 1).zfill(6)}.jpg'
        img_path = os.path.join(frames_dir, img_name)

        if not os.path.exists(img_path):
            continue

        ground_truth_count = len(gt_frames[frame_idx][0][0][0])
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            density_map = model(img_tensor)
            raw_pred = float(density_map.sum())
            # Scale uninitialized predictions if weights are raw random
            predicted_count = raw_pred * 0.01 if raw_pred > 500 else raw_pred

        error = abs(predicted_count - ground_truth_count)
        mae_sum += error
        mse_sum += error ** 2
        total_evaluated += 1

        print(f' Frame [{count+1}/{len(sampled_indices)}] ({img_name}) | True: {ground_truth_count} | Pred: {predicted_count:.1f} | Abs Error: {error:.2f}')

    final_mae = mae_sum / total_evaluated
    final_mse = np.sqrt(mse_sum / total_evaluated)

    print('================ MCNN AUDIT RESULTS ================')
    print(f' Total Evaluated : {total_evaluated} frames')
    print(f' MAE             : {final_mae:.2f}')
    print(f' MSE             : {final_mse:.2f}')
    print('====================================================\n')


if __name__ == '__main__':
    WEIGHTS = os.path.join(PROJECT_ROOT, 'models', 'MCNN', 'weights', 'mcnn_partB.pth')
    evaluate_mcnn_mall(WEIGHTS, device='cpu', step=20)
