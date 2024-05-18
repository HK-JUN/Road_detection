import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=False)
        self.relu =  nn.ReLU()
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class Unet(nn.Module):
    def __init__(self,in_channels,out_channels=1, init_features=32):
        super(Unet,self).__init__()
        self.enc1 = ConvBlock(in_channels,init_features)
        self.enc2 = ConvBlock(init_features,init_features*2)
        self.enc3 = ConvBlock(init_features*2,init_features*4)
        self.enc4 = ConvBlock(init_features*4,init_features*8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bridge = ConvBlock(init_features*8,init_features*16)

        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(init_features * 16, init_features * 8, kernel_size=3, padding=1)
        )
        self.dec4 = ConvBlock((init_features * 8) * 2, init_features * 8)
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(init_features * 8, init_features * 4, kernel_size=3, padding=1)
        )
        self.dec3 = ConvBlock((init_features * 4) * 2, init_features * 4)

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(init_features * 4, init_features * 2, kernel_size=3, padding=1)
        )
        self.dec2 = ConvBlock((init_features * 2) * 2, init_features * 2)
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(init_features * 2, init_features, kernel_size=3, padding=1)
        )
        self.dec1 = ConvBlock(init_features * 2, init_features)

        self.conv = nn.Conv2d(in_channels=init_features, out_channels=out_channels, kernel_size=1)
    
    def forward(self,x):
        # Encoding path
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        bridge = self.bridge(self.pool(x4))

        aug_x4 = torch.cat([self.upconv4(bridge),x4],dim=1)
        x4d = self.dec4(aug_x4)
        aug_x3 = torch.cat([self.upconv3(x4d),x3],dim=1)
        x3d = self.dec3(aug_x3)
        aug_x2 = torch.cat([self.upconv2(x3d),x2],dim=1)
        x2d = self.dec2(aug_x2)
        aug_x1 = torch.cat([self.upconv1(x2d),x1],dim=1)
        x1d = self.dec1(aug_x1)
        out = self.conv(x1d)
        return out