import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from skimage.restoration import estimate_sigma
from scipy.ndimage import median_filter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None, device='cuda'):
        self.image_paths = image_paths
        self.patch_size = 512
        self.Img_list = self.get_path(self.image_paths)
        print(f'Total imgs in current set {self.image_paths} is {len(self.Img_list[0])}.')
        self.device = device
        self.transform = transform
        self.noise_levels = [self.get_noise_level(idx) for idx in range(len(self))]
        print("cal noise finish")

    def get_path(self, image_paths):
        img_files = []
        gt_files = []
        input_path = image_paths
        f_list = sorted(os.listdir(input_path))
        for f_name in f_list:
            f_path = os.path.join(input_path, f_name)
            if os.path.isfile(f_path) and f_path.endswith(('.png', '.jpg', '.jpeg')):
                img_files.append(f_path)
                file_name, file_ext = os.path.splitext(f_path)
                gt_name = file_name.replace('images','labels')
                gt_name = gt_name[:-3]+'mask'
                gt_name = (gt_name +'.png')
                gt_files.append(gt_name)
        return img_files, gt_files

    def __len__(self):
        assert len(self.Img_list[0]) == len(self.Img_list[1])
        return len(self.Img_list[0])

    def __getitem__(self, idx):
        print(f"Acessing item {idx}")
        logger.info("Accessing item %d", idx)
        img_path = self.Img_list[0][idx]
        gt_path = self.Img_list[1][idx]

        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')  # convert label images to grayscale

        if self.transform:
            img = self.transform(img)
            target = self.transform(target)
            target = target.squeeze(0)
        print(f"target shape:{target.shape}")
        sample = {'image': img, 'gt': target, 'noise_level': self.noise_levels[idx]}
        return sample

    def get_noise_level(self, idx):
        print(f"Cal noise item: {idx}")
        logger.info("Accessing item %d", idx)
        img_path = self.Img_list[0][idx]
        img = Image.open(img_path).convert('RGB')
        np_img = np.array(img)
        sigma = estimate_sigma(np_img,channel_axis=-1)
        return sigma

    def top_n_noisy_images(self, n=5):
        print("Selecting the top 5 noisy images...")
        noisy_indices = np.argsort(self.noise_levels)[-n:]

        # Convert numpy integer to native Python int
        noisy_indices = [int(i) for i in noisy_indices.flatten()]

        samples = [self[i] for i in noisy_indices]
        print("done")
        return samples

def save_images(images, output_dir, title):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image_dict in enumerate(images):
        img, gt = image_dict['image'], image_dict['gt']
        img_np = img.permute(1, 2, 0).numpy()
        gt_np = gt.numpy()

        img_filename = os.path.join(output_dir, f"{title}_image_{i+1}.png")
        #gt_filename = os.path.join(output_dir, f"{title}_gt_{i+1}.png")

        plt.imsave(img_filename, img_np)
        #plt.imsave(gt_filename, gt_np, cmap='gray')


# Example usage

# Sample data (replace with actual data)
image_paths = "/home/jhpark/road/dataset/Train/images"
output_directory = "/home/jhpark/road/dataset/imagetest/"

transform = transforms.Compose([transforms.Resize(512),transforms.ToTensor()])
dataset = CustomImageDataset(image_paths, transform=transform)

top_noisy_images = dataset.top_n_noisy_images(n=5)
#save_images(top_noisy_images, output_directory, title="Top Noisy Images")
