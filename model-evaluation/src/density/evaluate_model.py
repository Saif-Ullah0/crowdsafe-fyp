import os
import sys
import glob
import torch
import numpy as np
import scipy.io as io
from PIL import Image

# Dynamic import path for model
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "CSRNet"))

from model import CSRNet  # Swap for DMCount when evaluating DM-Count weights


def evaluate_mall_dataset(weights_path: str, device: str = "cpu", step: int = 20):
    """
    Evaluates MAE and MSE of a model checkpoint on the Mall Dataset.
    step=20 evaluates 100 frames out of 2000 for fast local CPU auditing.
    """
    # 1. Initialize model
    model = CSRNet()
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    # 2. Paths to frames and ground-truth file
    frames_dir = os.path.join(PROJECT_ROOT, "datasets", "Mall", "frames")
    gt_path = os.path.join(PROJECT_ROOT, "datasets", "Mall", "ground_truth", "GT.mat")

    if not os.path.exists(gt_path):
        print(f"[Error] GT.mat not found at {gt_path}")
        return

    # Load Mall ground truth annotations
    gt_mat = io.loadmat(gt_path)
    gt_frames = gt_mat["frame"][0]

    # Select sub-sampled index sequence
    sampled_indices = list(range(0, len(gt_frames), step))

    mae_sum = 0.0
    mse_sum = 0.0
    total_evaluated = 0

    print(f"\n--- Auditing Model: {os.path.basename(weights_path)} ---")
    print(f"Sub-sampling step={step}: Evaluating {len(sampled_indices)} frames out of {len(gt_frames)} total...\n")

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for count, frame_idx in enumerate(sampled_indices):
        frame_data = gt_frames[frame_idx]
        img_name = f"seq_{str(frame_idx + 1).zfill(6)}.jpg"
        img_path = os.path.join(frames_dir, img_name)

        if not os.path.exists(img_path):
            continue

        # Get true ground truth headcount
        true_points = frame_data[0][0][0]
        ground_truth_count = len(true_points)

        # Run inference
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            density_map = model(img_tensor).squeeze().cpu().numpy()
            predicted_count = float(density_map.sum())

        # Accumulate errors
        error = abs(predicted_count - ground_truth_count)
        mae_sum += error
        mse_sum += error ** 2
        total_evaluated += 1

        print(f" Frame [{count+1}/{len(sampled_indices)}] ({img_name}) | True: {ground_truth_count} | Pred: {predicted_count:.1f} | Abs Error: {error:.2f}")

    final_mae = mae_sum / total_evaluated
    final_mse = np.sqrt(mse_sum / total_evaluated)

    print("\n================ FINAL MODEL AUDIT RESULTS ================")
    print(f" Model Evaluated : {os.path.basename(weights_path)}")
    print(f" Total Evaluated : {total_evaluated} frames")
    print(f" Mean Absolute Error (MAE) : {final_mae:.2f}")
    print(f" Root Mean Squared Error (MSE): {final_mse:.2f}")
    print("===========================================================\n")


if __name__ == "__main__":
    WEIGHTS = os.path.join(PROJECT_ROOT, "models", "CSRNet", "weights", "partBmodel_best.pth.tar")
    evaluate_mall_dataset(WEIGHTS, device="cpu", step=20)