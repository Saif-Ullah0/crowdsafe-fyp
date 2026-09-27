import numpy as np
import scipy.io as io
import scipy.ndimage
from PIL import Image
from pathlib import Path

def make_density_map(image_path, points, sigma=15):
    img = Image.open(image_path)
    w, h = img.size
    density = np.zeros((h, w), dtype=np.float32)
    for pt in points:
        x = min(int(pt[0]), w - 1)
        y = min(int(pt[1]), h - 1)
        density[y, x] += 1
    density = scipy.ndimage.gaussian_filter(density, sigma=sigma)
    return density

def process_shanghaitech(root, part='B', limit=5):
    root = Path(root)
    # Uses your existing folder naming with _final
    part_folder = f'part_{part}_final'
    
    for split in ['train_data', 'test_data']:
        img_dir = root / part_folder / split / 'images'
        gt_dir  = root / part_folder / split / 'ground_truth'
        dm_dir  = root / part_folder / split / 'density_maps'
        
        dm_dir.mkdir(parents=True, exist_ok=True)
        
        if not img_dir.exists():
            print(f"Skipping: {img_dir} does not exist.")
            continue

        images = sorted(img_dir.glob('*.jpg'))[:limit]
        print(f"\n---> Processing {len(images)} sample images for ShanghaiTech {part_folder}/{split}...")

        for img_path in images:
            gt_path = gt_dir / f'GT_{img_path.stem}.mat'
            if not gt_path.exists():
                continue
            mat = io.loadmat(str(gt_path))
            points = mat['image_info'][0][0][0][0][0]
            density = make_density_map(img_path, points)
            np.save(dm_dir / f'{img_path.stem}.npy', density)
            print(f'   [OK] {img_path.stem} | ground truth count: {len(points)} | map sum: {density.sum():.1f}')

def process_ucfqnrf(root, limit=5):
    root = Path(root)
    for split in ['Train', 'Test']:
        dm_dir = root / split / 'density_maps'
        dm_dir.mkdir(parents=True, exist_ok=True)
        
        if not (root / split).exists():
            print(f"Skipping: {root / split} does not exist.")
            continue

        ann_files = sorted((root / split).glob('*_ann.mat'))[:limit]
        print(f"\n---> Processing {len(ann_files)} sample images for UCF-QNRF/{split}...")

        for mat_path in ann_files:
            stem     = mat_path.stem.replace('_ann', '')
            img_path = root / split / f'{stem}.jpg'
            if not img_path.exists():
                continue
            mat     = io.loadmat(str(mat_path))
            points  = mat['annPoints']
            density = make_density_map(img_path, points)
            np.save(dm_dir / f'{stem}.npy', density)
            print(f'   [OK] {stem} | ground truth count: {len(points)} | map sum: {density.sum():.1f}')

if __name__ == '__main__':
    SHA = './datasets/ShanghaiTech'
    QNRF = './datasets/UCF-QNRF'
    
    # Set sample limit (e.g., 5 images per split)
    SAMPLE_LIMIT = 5 
    
    print('=== ShanghaiTech Part A (Quick Test) ===')
    process_shanghaitech(SHA, 'A', limit=SAMPLE_LIMIT)
    
    print('\n=== ShanghaiTech Part B (Quick Test) ===')
    process_shanghaitech(SHA, 'B', limit=None)
    
    print('\n=== UCF-QNRF (Quick Test) ===')
    process_ucfqnrf(QNRF, limit=SAMPLE_LIMIT)