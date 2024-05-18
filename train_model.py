import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import logging
from uModel import Unet
from preprocess import CustomImageDataset
from sklearn.model_selection import train_test_split

BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCH = 10
image_paths = "/home/jhpark/road/dataset/Train/images"
train_dataset = CustomImageDataset(image_paths=image_paths)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(in_channels=3,out_channels=1,init_features=32).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []  # 훈련 손실 저장 리스트
valid_losses = []  # 검증 손실 저장 리스트
for epoch in range(NUM_EPOCH):
    model.train()
    running_loss = 0.0
    itemnum = 0
    for data in train_loader:
        image = data['image']
        target = data['gt']
        image,target = image.to(device),target.to(device)
        optimizer.zero_grad()
        print(f"image shape:{image.shape}") #[4, 3, 512, 512]
        outputs = model(image)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        running_loss +=loss.item()
        print(f"epoch {epoch+1}, batch {itemnum+1}/{len(train_loader)} finished,Train Loss: {loss.item()}")
        itemnum +=1
    train_losses.append(running_loss / len(train_loader))
print("train finish. save model...")
PATH = f"/home/jhpark/road/saved_model/model_e{NUM_EPOCH}_b{BATCH_SIZE}_l{LEARNING_RATE}"
torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)