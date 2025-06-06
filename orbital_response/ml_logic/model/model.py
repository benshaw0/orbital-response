import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Convolution block for decoder
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )


# U-Net model with ResNet34 encoder
class UNetModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()

        # Load pretrained resnet and modify input channels
        self.encoder = models.resnet34(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Decoder with transpose convs
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = conv_block(512, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = conv_block(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        e2 = self.encoder.layer1(self.encoder.maxpool(e1))
        e3 = self.encoder.layer2(e2)
        e4 = self.encoder.layer3(e3)
        e5 = self.encoder.layer4(e4)

        d5 = self.dec5(torch.cat([self.up5(e5), e4], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))

        out = self.final_conv(d2)
        return F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)


# Function to return an instance of the model
def get_model(n_classes=5):
    return UNetModel(n_classes=n_classes)
