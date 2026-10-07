import numpy as np
import matplotlib.pyplot as plt

# 1. Load one generated density map
path = 'datasets/ShanghaiTech/part_A_final/train_data/density_maps/IMG_1.npy'
density_map = np.load(path)

# 2. Print verification stats in terminal
print("=" * 40)
print("Density Map Shape:", density_map.shape)
print(f"Estimated Person Count (Sum): {np.sum(density_map):.2f}")
print("=" * 40)

# 3. Save visualization image instead of plt.show()
plt.figure(figsize=(10, 6))
plt.imshow(density_map, cmap='jet')
plt.title(f"Density Map - Count: {np.sum(density_map):.1f}")
plt.colorbar()

output_image_path = 'density_map_preview.png'
plt.savefig(output_image_path, bbox_inches='tight')
print(f"Saved visualization image to: {output_image_path}")