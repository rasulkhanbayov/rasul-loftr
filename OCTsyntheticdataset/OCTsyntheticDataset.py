# -*- coding: utf-8 -*-
import os
import random
import time
import kornia
import imgaug.augmenters as iaa
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import kornia.geometry.transform as KTF
import matplotlib.pyplot as plt
import torch.nn.functional as F
from warnings import filterwarnings


filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


class OCTsyntheticDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir

        base_names = []
        for f in os.listdir(img_dir):
            if f.endswith('_enface.png'):
                base_name = f[:-len('_enface.png')]
                if (os.path.exists(os.path.join(img_dir, base_name + '_DataCapture.png')) and
                    os.path.exists(os.path.join(img_dir, base_name + '_homography.txt'))):
                    base_names.append(base_name)
        self.data = sorted(base_names)
        self.coarse_scale = 0.125

    def __len__(self):
        return len(self.data)

    def load_homography(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        H = []
        for line in lines:
            row = [float(x) for x in line.strip().split(',')]
            H.append(row)
        return np.array(H, dtype=np.float32)

    def __getitem__(self, idx):
        base_name = self.data[idx]

        img0_path = os.path.join(self.img_dir, base_name + '_gan_enface.png')
        img1_path = os.path.join(self.img_dir, base_name + '_DataCapture.png')

        homo_path = os.path.join(self.img_dir, base_name + '_homography.txt')

        img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
        # img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

        # TO EXTRACT GREEN CHANNEL
        img_color = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img1 = img_color[:, :, 1]  # Green channel only

        # img0 = cv2.flip(img0, 0)  # Flip vertically

        H = self.load_homography(homo_path)

        crop_size, new_img1, H2 = create_random_sample(img1, H)

        new_img1 = cv2.resize(new_img1, (512, 512), interpolation=cv2.INTER_LINEAR)
        scale_factor = (512.0 / crop_size) # Or 0.25 to reduce 2048 -> 1024 or 512
        # img0 = cv2.resize(img0, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        # img1 = cv2.resize(img1, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        S = np.array([[scale_factor, 0, 0],
                    [0, scale_factor, 0],
                    [0, 0, 1]], dtype=np.float32)
        # S_inv = np.linalg.inv(S)
        H2 = S @ H2
        
        img0_tensor = torch.from_numpy(img0.astype(np.float32) / 255.0).unsqueeze(0)
        img1_tensor = torch.from_numpy(new_img1.astype(np.float32) / 255.0).unsqueeze(0)
        H_tensor = torch.from_numpy(H2).float()

        mask0 = torch.from_numpy((img0 > 0).astype(np.float32))
        mask1 = torch.from_numpy(np.ones_like(new_img1).astype(np.float32))

        if mask0 is not None:
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            mask0 = ts_mask_0
            mask1 = ts_mask_1

        return {
            'image0': img0_tensor,
            'image1': img1_tensor,
            'T_0to1': H_tensor,
            'T_1to0': torch.linalg.inv(H_tensor),
            'mask0': mask0,
            'mask1': mask1,
            'dataset_name': 'OCTsynthetic',
            'pair_id': idx,
            'pair_names': (base_name + '_enface', base_name + '_DataCapture'),
        }
    
def visualize_sample(sample, alpha=0.5):
    H = sample['T_0to1'].numpy()

    # Load the images
    src_img = sample['image0'].squeeze(0).numpy()
    dst_img = sample['image1'].squeeze(0).numpy()

    # src_img = cv2.resize(src_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    # dst_img = cv2.resize(dst_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Warp the source image using the homography matrix
    warped_src = cv2.warpPerspective(src_img, H, (dst_img.shape[1], dst_img.shape[0]))

    # Create a transparent overlay
    overlay = cv2.addWeighted(warped_src, alpha, dst_img, 1 - alpha, 0)

    # Show results side-by-side
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Source Image")
    plt.imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title("Warped Source")
    plt.imshow(cv2.cvtColor(warped_src, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.IMREAD_COLOR))

    plt.tight_layout()
    plt.show()
    # plt.savefig("/home/guests/rasul_khanbayov/New/loFTR-master/OCTsyntheticdataset/visualized_sample_33.png")

    return overlay

def main():
    dataset_path = "C:/Users/rasul/Desktop/LoFTR-master/data/OCTsynthetic/data_new/"
    dataset = OCTsyntheticDataset(dataset_path)
    
    sample = dataset[39]

    visualize_sample(sample, 0.5)

def create_random_sample(fundus_image, H1, max_retries=10):
    import random
    import numpy as np
    import cv2

    for attempt in range(max_retries):
        try:
            # Pick random crop size
            crop_size = random.randint(1000, 2048)

            fundus_h, fundus_w = (2048, 2048)
            enface_h, enface_w = (512, 512)

            # Step 1: Get projected corners of enface image in fundus space
            enface_corners = np.array([
                [0, 0, 1],
                [enface_w, 0, 1],
                [enface_w, enface_h, 1],
                [0, enface_h, 1]
            ]).T  # shape (3, 4)

            projected = H1 @ enface_corners
            projected /= projected[2]  # normalize
            projected_xy = projected[:2].T  # shape (4, 2)

            # Get bounding box around projected enface
            min_x, min_y = projected_xy.min(axis=0)
            max_x, max_y = projected_xy.max(axis=0)

            # Make sure enface projection fits inside the crop and within image
            margin_x = crop_size - (max_x - min_x)
            margin_y = crop_size - (max_y - min_y)
            if margin_x < 0 or margin_y < 0:
                raise ValueError("Projected enface image too large to fit in crop")

            # Choose crop top-left corner such that projected region is inside
            min_crop_x = max(0, int(max_x - crop_size))
            max_crop_x = min(int(min_x), fundus_w - crop_size)
            min_crop_y = max(0, int(max_y - crop_size))
            max_crop_y = min(int(min_y), fundus_h - crop_size)

            # 🧩 NEW: Add padding if needed
            pad_left = pad_top = 0

            if max_crop_x < min_crop_x:
                pad_left = int(abs(min_crop_x - max_crop_x))
            if max_crop_y < min_crop_y:
                pad_top = int(abs(min_crop_y - max_crop_y))

            if pad_left > 0 or pad_top > 0:
                fundus_image = cv2.copyMakeBorder(
                    fundus_image,
                    pad_top, 0, pad_left, 0,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
                crop_x_offset = pad_left
                crop_y_offset = pad_top
                crop_x = max_crop_x + crop_x_offset
                crop_y = max_crop_y + crop_y_offset

                # Adjust homography to account for padding
                pad_matrix = np.array([
                    [1, 0, pad_left],
                    [0, 1, pad_top],
                    [0, 0, 1]
                ])
                H1 = pad_matrix @ H1
            else:
                crop_x = random.randint(min_crop_x, max_crop_x)
                crop_y = random.randint(min_crop_y, max_crop_y)

            # Step 3: Create translation matrix
            T = np.array([
                [1, 0, -crop_x],
                [0, 1, -crop_y],
                [0, 0, 1]
            ])

            # Step 4: Compute new homography
            H2 = T @ H1  # enface → cropped fundus
            fundus_crop = fundus_image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

            return crop_size, fundus_crop, H2

        except ValueError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to find valid crop after {max_retries} attempts") from e
            continue  # try again


if __name__ == '__main__':
    main()