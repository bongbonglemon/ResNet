import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3]):
        super(ResNet, self).__init__()

        self.inplanes = 64

        # 3*224*224
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batchnorm = nn.BatchNorm2d(64)
        # 64*112*112
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        # 64*56*56
        self.conv2 = self._make_layer(64, layers[0])
        self.conv3 = self._make_layer(128, layers[1])
        self.conv4 = self._make_layer(256, layers[2])
        self.conv5 = self._make_layer(512, layers[3])
        # 512*7*7
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 512*1*1
        self.linear = nn.Linear(512, 1000)
        self.projection = None

    def forward(self, x):
        x = self.conv1(x)
        # KIV
        x = self.batchnorm(x)
        x = F.relu(x)
        ####
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        ####
        x = self.avg_pool(x)
        x = x.view(1, -1)
        x = self.linear(x)

        return x

    def _make_layer(self, planes, num_blocks):
        layers = []
        layers.append(resblock(self.inplanes, planes))
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(resblock(self.inplanes, planes))
        return nn.Sequential(*layers)


class resblock(nn.Module):
    def __init__(self, m, n):
        super(resblock, self).__init__()
        # 64*56*56
        self.conv1 = nn.Conv2d(m, n, kernel_size=3,
                               padding=1, stride=2 if n > m else 1)  # TODO: ugly tho
        # 128*28*28
        self.conv2 = nn.Conv2d(n, n, kernel_size=3, padding=1)
        # projection shortcut
        self.projection = nn.Conv2d(m, n, kernel_size=1, stride=2)
        self.batchnorm = nn.BatchNorm2d(n)
        self.to_project = n > m

    def forward(self, x):
        y = self.conv1(x)
        y = self.batchnorm(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.batchnorm(y)
        if(self.to_project):
            x = self.projection(x)  # TODO: should i turn off auto_grad
            x = self.batchnorm(x)
        x = F.relu(x+y)
        return x


resnet18 = ResNet([2, 2, 2, 2])
test = torch.rand(1, 3, 224, 224)
out = resnet18.forward(test)
print(out.shape)
