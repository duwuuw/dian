import torch
import torch.nn as nn
import torch.nn.functional as F

class AlenNet(nn.Module):
    def __init__(self,input_channels = 3,num_classes = 10,dropout = 0.05):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 96,kernel_size = 11,stride = 4),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5,alpha = 1e-4,beta = 0.75,k = 2),
            nn.MaxPool2d(kernel_size = 3,stride = 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 96,out_channels = 256,kernel_size = 5,padding = 2),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5,alpha = 1e-4,beta = 0.75,k = 2),
            nn.MaxPool2d(kernel_size = 3,stride = 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 256,out_channels = 384,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 384,out_channels = 384,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 384,out_channels = 256,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3,stride = 2),
            nn.AdaptiveAvgPool2d((6,6))
        )
        self.block4 = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(in_features = 256*6*6,out_features = 4096),
            nn.ReLU(inplace = True),

            nn.Dropout(p = dropout),
            nn.Linear(in_features = 4096,out_features = 4096),
            nn.ReLU(inplace = True),

            nn.Dropout(p = dropout),
            nn.Linear(in_features = 4096,out_features = num_classes)
        )
    
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x,1)
        x = self.block4(x)
        return x
"""
很显然这里只是我排除错误的地方，，嗯对，现在被废弃，了
    def check(self,x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            return x
device = ('cuda')
model = AlenNet().to(device)
x = torch.randn(1,3,224,224).to(device)
print(model.check(x).shape)
"""


class BasicBlock(nn.Module):
    """ResNet18/34 使用的 BasicBlock，对应 torchvision.models.resnet.BasicBlock"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # conv1: 3x3, stride=stride, padding=1, bias=False (因为后面跟了BN)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # conv2: 3x3, stride=1, padding=1, bias=False
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # shortcut 投影层（stride!=1 或 channel 不匹配时使用）
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """官方 torchvision 风格的 ResNet18 实现"""

    def __init__(self, num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.in_channels = 64

        # Stem: 7x7 conv + BN + ReLU + 3x3 maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 stage，每个 stage 2 个 BasicBlock
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # 权重初始化（和官方一致）
        self._init_weights(zero_init_residual)

    def _make_layer(self, out_channels, num_blocks, stride=1):
        """构建一个 stage：第一个 block 负责下采样，后续 block stride 恒为 1"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual):
        # Kaiming 初始化卷积层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 将最后一个 BN 的 weight 初始化为 0（官方的 residual 优化技巧）
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

