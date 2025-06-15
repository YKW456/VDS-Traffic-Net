import math

import torch
from torch import nn
from torchsummary import summary


# 定义Residual块
class Residual(nn.Module):
    # input_channels:输入通道数；num_channels:输出通道数；use_1conv:是否使用1*1的卷积；strides:默认步幅为1
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        # 若该卷积层使用1*1的卷积，则会将该层的stride和1*1的卷积的步幅设置为2
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1,
                               stride=strides)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)  # 批量规范化
        self.bn2 = nn.BatchNorm2d(num_channels)
        # 判断在残差块的右边是否使用1*1的卷积，若使用时将其步幅设置为2
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y = self.relu(y + x)
        return y


class ResNet18(nn.Module):
    def __init__(self, Residual):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(Residual(64, 64, use_1conv=False, strides=1),
                                Residual(64, 64, use_1conv=False, strides=1))

        self.b3 = nn.Sequential(Residual(64, 128, use_1conv=True, strides=2),
                                Residual(128, 128, use_1conv=False, strides=1))

        self.b4 = nn.Sequential(Residual(128, 256, use_1conv=True, strides=2),
                                Residual(256, 256, use_1conv=False, strides=1))

        self.b5 = nn.Sequential(Residual(256, 256, use_1conv=True, strides=2),
                                Residual(256, 256, use_1conv=False, strides=1))

        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(256, 8))

        # 参数太大，不好训练，真实参数为 ↓
        # self.b5 = nn.Sequential(Residual(256, 512, use_1conv=True, strides=2),
        #                         Residual(512, 512, use_1conv=False, strides=1))
        #
        # self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                         nn.Flatten(),
        #                         nn.Linear(512, 10))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),  # 输入: (3, 224, 224), 输出: (16, 224, 224)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 输出: (16, 112, 112)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # 输入: (16, 112, 112), 输出: (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 输出: (32, 56, 56)
        )
        self.fc = nn.Linear(32 * 56 * 56, 8)  # 输入: 32 * 56 * 56, 输出: 8

    def forward(self, x):
        x = self.layer1(x)  # 输出: (batch_size, 16, 112, 112)
        x = self.layer2(x)  # 输出: (batch_size, 32, 56, 56)
        x = x.view(x.size(0), -1)  # 展平: (batch_size, 32 * 56 * 56)
        x = self.fc(x)  # 输出: (batch_size, 8)
        return x


# 定义基础残差块
class BasicBlock(nn.Module):
    expansion = 1  # 扩展系数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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

# 定义简化后的 ResNet2
class ResNet2(nn.Module):
    def __init__(self, block, layers, num_classes=8):
        self.inplanes = 16  # 减少初始通道数
        super(ResNet2, self).__init__()
        # 1. conv1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False)  # 减少通道数
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # 2. conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])  # 减少通道数
        # 3. conv3_x
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)  # 减少通道数
        # 4. conv4_x
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)  # 减少通道数
        # 5. 移除 conv5_x（layer4）

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化，适用于任意输入尺寸
        self.fc = nn.Linear(64 * block.expansion, num_classes)  # 减少全连接层输入维度

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # 更新 self.inplanes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x

# 创建简化后的 ResNet2 模型
def resnet2_simple(num_classes=8):
    return ResNet2(BasicBlock, [1, 1, 1], num_classes)  # 减少每个阶段的残差块数量

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(Residual).to(device)
    print(summary(model, (1, 224, 224)))