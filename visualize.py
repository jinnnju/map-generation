import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os

def visualize_map(npy_path):
    data = np.load(npy_path)
    filename = os.path.basename(npy_path).lower()

    plt.figure(figsize=(5, 4))

    if 'elevation' in filename:
        im = plt.imshow(data, cmap='viridis', vmin=0, vmax=2000)
        cbar = plt.colorbar(im, label='Elevation')
        cbar.locator = ticker.MultipleLocator(500)
        cbar.update_ticks()
        plt.title('Elevation')
    elif 'water' in filename:
        im = plt.imshow(data, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im)
        plt.title('Water Map')
    
    else:
        im = plt.imshow(data, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im)
        plt.title('Vegetation Index')

    # 마지막 부분 교체
    plt.tight_layout()
    save_path = os.path.splitext(npy_path)[0] + "_viz.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize .npy map file (elevation, vegetation, water)")
    parser.add_argument("npy_path", type=str, help="Path to .npy map file")
    args = parser.parse_args()

    visualize_map(args.npy_path)

