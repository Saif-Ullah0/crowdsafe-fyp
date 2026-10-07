import os
import sys
import torch
import numpy as np
import scipy.io as io
from PIL import Image
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

# Add DMCount to path
DMCOUNT_DIR = os.path.join(PROJECT_ROOT, "models", "DMCount")
if DMCOUNT_DIR not in sys.path:
    sys.path.insert(0, DMCOUNT_DIR)

from vgg import VGG19_DMCount as DMCountModel


def evaluate_dmcount_mall(weights_path: str, device: str = "cpu", step: int = 20):
    if not os.path.exists(weights_path):
        print(f"[ERROR] Missing weights file at: {weights_path}")
        return

    model = DMCountModel(pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    frames_dir = os.path.join(PROJECT_ROOT, "datasets", "Mall", "frames")
    gt_path = os.path.join(PROJECT_ROOT, "datasets", "Mall", "ground_truth", "GT.mat")

    gt_mat = io.loadmat(gt_path)
    gt_frames = gt_mat["frame"][0]

    sampled_indices = list(range(0, len(gt_frames), step))
    mae_sum = 0.0
    mse_sum = 0.0
    total_evaluated = 0

    print(f"\n--- Auditing DM-Count Model: {os.path.basename(weights_path)} ---")
    print(f"Evaluating across {len(sampled_indices)} frames...\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for count, frame_idx in enumerate(sampled_indices):
        img_name = f"seq_{str(frame_idx + 1).zfill(6)}.jpg"
        img_path = os.path.join(frames_dir, img_name)

        if not os.path.exists(img_path):
            continue

        ground_truth_count = len(gt_frames[frame_idx][0][0][0])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out, _ = model(img_tensor)
            predicted_count = float(out.sum())

        error = abs(predicted_count - ground_truth_count)
        mae_sum += error
        mse_sum += error ** 2
        total_evaluated += 1

        print(f" Frame [{count+1}/{len(sampled_indices)}] ({img_name}) | True: {ground_truth_count} | Pred: {predicted_count:.1f} | Abs Error: {error:.2f}")

    final_mae = mae_sum / total_evaluated
    final_mse = np.sqrt(mse_sum / total_evaluated)

    print("\n================ DM-COUNT AUDIT RESULTS ================")
    print(f" Total Evaluated : {total_evaluated} frames")
    print(f" MAE             : {final_mae:.2f}")
    print(f" MSE             : {final_mse:.2f}")
    print("========================================================\n")


if __name__ == "__main__":
    WEIGHTS = os.path.join(PROJECT_ROOT, "models", "DMCount", "weights", "model_qnrf.pth")
    evaluate_dmcount_mall(WEIGHTS, device="cpu", step=20)