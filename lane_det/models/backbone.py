import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, pretrained=True, replace_stride_with_dilation=None):
        super(ResNet18, self).__init__()
        # Load standard ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # We only need layers up to layer4
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Delete unused layers to save memory
        del resnet.avgpool
        del resnet.fc
        
        # Channels for each layer
        self.out_channels = [64, 128, 256, 512]

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # Stride 4

        # Layers
        c2 = self.layer1(x) # Stride 4, 64
        c3 = self.layer2(c2) # Stride 8, 128
        c4 = self.layer3(c3) # Stride 16, 256
        c5 = self.layer4(c4) # Stride 32, 512
        
        return [c2, c3, c4, c5]
