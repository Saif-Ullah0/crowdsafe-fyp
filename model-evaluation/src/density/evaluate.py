import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import sys
import csv

# Add CSRNet module to path
sys.path.insert(0, './models/CSRNet')
from model import CSRNet

def evaluate(weights_path, test_img_dir, test_gt_dir, dataset_name='SHA-B'):
    model = CSRNet()
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f'Weights successfully loaded from: {weights_path}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_dir = Path(test_img_dir)
    gt_dir  = Path(test_gt_dir)

    mae_list = []
    mse_list = []

    img_paths = sorted(img_dir.glob('*.jpg'))

    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
        predicted = output.sum().item()

        dm_path = gt_dir.parent / 'density_maps' / f'{img_path.stem}.npy'
        if not dm_path.exists():
            continue
            
        gt_density = np.load(str(dm_path))
        actual = gt_density.sum()

        error = abs(predicted - actual)
        mae_list.append(error)
        mse_list.append(error ** 2)

        print(f'{img_path.stem} | Actual: {actual:.0f} | Predicted: {predicted:.1f} | Error: {error:.1f}')

    if not mae_list:
        print("No matching test density map files found! Check dataset paths.")
        return 0, 0

    mae = np.mean(mae_list)
    mse = np.sqrt(np.mean(mse_list))
    
    print("\n" + "=" * 45)
    print(f'Dataset Benchmark : {dataset_name}')
    print(f'Mean Absolute Error (MAE) : {mae:.2f}')
    print(f'Mean Squared Error (MSE)  : {mse:.2f}')
    print("=" * 45)
    return mae, mse

if __name__ == '__main__':
    Path('./experiments').mkdir(parents=True, exist_ok=True)
    
    mae, mse = evaluate(
        weights_path='./models/CSRNet/weights/partBmodel_best.pth.tar',
        test_img_dir='./datasets/ShanghaiTech/part_B_final/test_data/images',
        test_gt_dir ='./datasets/ShanghaiTech/part_B_final/test_data/ground_truth',
        dataset_name='SHA-B Pretrained Baseline'
    )