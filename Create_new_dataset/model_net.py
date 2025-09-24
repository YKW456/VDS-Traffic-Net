import math
import torch
from torch import nn

# Optional import for torchsummary
try:
    from torchsummary import summary
    HAS_TORCHSUMMARY = True
except ImportError:
    HAS_TORCHSUMMARY = False
    print("Warning: torchsummary not available. Model summary will be skipped.")


# Define Residual block
class Residual(nn.Module):
    # input_channels: input channels; num_channels: output channels; use_1conv: whether to use 1x1 convolution; strides: default stride is 1
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        # If using 1x1 convolution, set stride of this layer and 1x1 convolution to 2
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1,
                               stride=strides)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)  # Batch normalization
        self.bn2 = nn.BatchNorm2d(num_channels)
        # Determine whether to use 1x1 convolution on the right side of residual block, set stride to 2 if used
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

        # Original parameters were too large for training, real parameters are ↓
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
            nn.Conv2d(3, 16, kernel_size=5, padding=2),  # Input: (3, 224, 224), Output: (16, 224, 224)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: (16, 112, 112)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # Input: (16, 112, 112), Output: (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: (32, 56, 56)
        )
        self.fc = nn.Linear(32 * 56 * 56, 8)  # Input: 32 * 56 * 56, Output: 8

    def forward(self, x):
        x = self.layer1(x)  # Output: (batch_size, 16, 112, 112)
        x = self.layer2(x)  # Output: (batch_size, 32, 56, 56)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 32 * 56 * 56)
        x = self.fc(x)  # Output: (batch_size, 8)
        return x


# Define basic residual block
class BasicBlock(nn.Module):
    expansion = 1  # Expansion factor

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


# Define simplified ResNet2
class ResNet2(nn.Module):
    def __init__(self, block, layers, num_classes=8):
        self.inplanes = 16  # Reduce initial channels
        super(ResNet2, self).__init__()
        # 1. conv1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False)  # Reduce channels
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # 2. conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])  # Reduce channels
        # 3. conv3_x
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)  # Reduce channels
        # 4. conv4_x
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)  # Reduce channels
        # 5. Remove conv5_x (layer4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling for any input size
        self.fc = nn.Linear(64 * block.expansion, num_classes)  # Reduce FC layer input dimension

        # Initialize weights
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
        self.inplanes = planes * block.expansion  # Update self.inplanes
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
        x = x.view(x.size(0), -1)  # Flatten output
        x = self.fc(x)

        return x


# Create simplified ResNet2 model
def resnet2_simple(num_classes=8):
    return ResNet2(BasicBlock, [1, 1, 1], num_classes)  # Reduce number of residual blocks per stage

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(Residual).to(device)
    if HAS_TORCHSUMMARY:
        print(summary(model, (1, 224, 224)))
    else:
        print("Model created successfully, but torchsummary not available for detailed summary.")