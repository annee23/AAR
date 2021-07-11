import torch
import torch.nn as nn
from torch.nn import functional as F
from GaborNet import GaborConv2d
from torch.autograd import Variable
import cv2
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_filename = "./2.ppm"
image = cv2.imread(input_filename)
image = np.moveaxis(image, -1, 0)[None, ...]
image = torch.from_numpy(image).cuda().float()

class GaborNN(nn.Module):
    def __init__(self):
        super(GaborNN, self).__init__()
        self.g0 = GaborConv2d(in_channels=3, out_channels=96, kernel_size=(11, 11))
        self.c1 = nn.Conv2d(96, 384, (3, 3))
        self.fcs = nn.Sequential(
            nn.Linear(384 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 96),
        )

    def forward(self, x):
        x = F.leaky_relu(self.g0(x))
        x = nn.MaxPool2d(kernel_size=3)(x)
        x = F.leaky_relu(self.c1(x))
        x = nn.MaxPool2d(kernel_size=11)(x)
        x = x.view(-1, 384 * 500)
        x = self.fcs(x)
        return x

net = GaborNN().to(device)
print(net(image).shape)
print(net(image))

