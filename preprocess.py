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
    def __init__(self, image_paths, device='cuda'):
        self.image_paths = image_paths
        self.patch_size = 512
        self.Img_list = self.get_path(self.image_paths)
        #print(f'Total imgs in current set {self.image_paths} is {len(self.Img_list[0])}.')
        self.device = device
        self.img_transform = transforms.Compose([
            transforms.Resize(self.patch_size),
            transforms.ToTensor()
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize(self.patch_size),
            transforms.ToTensor()])  # range [0, 255] -> [0.0,1.0]
        #self.noise_levels = [self.get_noise_level(idx) for idx in range(len(self))]

    def get_path(self, input_path):
        img_files = []
        gt_files = []
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
        img_path = self.Img_list[0][idx]
        gt_path = self.Img_list[1][idx]

        img = self.preprocess_image(img_path)
        #img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')  # convert label images to grayscale
        img = self.img_transform(Image.fromarray(img))
        target = self.label_transform(target)
        #target = target.squeeze(0)
        sample = {'image': img, 'gt': target}
        return sample

    def preprocess_image(self,image_path):
        # Load the image
        image = cv2.imread(image_path)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise ValueError(f"Could not open or find the image: {image_path}")
        
        if len(image.shape) == 2:  # Grayscale image
            image = self.preprocess_channel(image)
        else:  # Color image
            # Split the image into R, G, B channels and preprocess each channel
            channels = cv2.split(image)
            processed_channels = [self.preprocess_channel(channel) for channel in channels]
            image = cv2.merge(processed_channels)

        return image

    def preprocess_channel(self,channel):
        
        channel = cv2.GaussianBlur(channel, (5, 5), 0)
        # Apply median filter to handle salt and pepper noise
        channel = median_filter(channel, size=3)
        # Apply horizontal median filter to handle stripe noise
        channel = median_filter(channel, size=(1, 5))

        return channel

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

dataset = CustomImageDataset(image_paths)


