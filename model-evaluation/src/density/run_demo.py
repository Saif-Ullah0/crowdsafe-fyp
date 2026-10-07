import os
import glob
import numpy as np
from PIL import Image
from process_frame import load_model, process_frame

# Load baseline model weights
MODEL_WEIGHTS = './models/CSRNet/weights/partBmodel_best.pth.tar'
load_model(MODEL_WEIGHTS, device='cpu')

# Get test frames
frame_paths = sorted(glob.glob('./datasets/Mall/demo_frames/*.jpg'))

print(f"--- Running Analytics Pipeline on {len(frame_paths)} Sample Frames ---\n")

for i, frame_path in enumerate(frame_paths):
    frame_name = os.path.basename(frame_path)
    img = Image.open(frame_path).convert('RGB')
    
    result = process_frame(
        camera_id='cam_01',
        frame_image=np.array(img)
    )
    
    print(f" Frame [{i+1}/{len(frame_paths)}]: {frame_name}")
    print(f"   ├─ Total Head Count : {result['total_count']} people")
    print(f"   ├─ Overall Risk     : {result['overall_risk'].upper()}")
    print(f"   ├─ Top Risk Zone    : {result['zones'][0]['name']} ({result['zones'][0]['density_per_m2']} people/m²)")
    print(f"   └─ Detected Clusters: {len(result['clusters'])} congestion blobs found\n")