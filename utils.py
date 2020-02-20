import random
import numpy as np
from scipy import interpolate
from torch import distributions, nn
import torch.nn.functional as f
from torch.utils.data import Dataset

from config import *


class ImageSet(Dataset):
    def __init__(self, input, result, boundaries):
        self.input = input
        self.result = result
        self.boundaries = boundaries

    def __getitem__(self, item):
        return self.input[item], self.result[item], self.boundaries[item]

    def __len__(self):
        return len(self.input)


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(30)
        self.conv2 = nn.Conv2d(30, 60, kernel_size=5)
        self.bn_conv2 = nn.BatchNorm2d(60)
        self.linear1 = nn.Linear(320 * 3, 1256)
        self.linear2 = nn.Linear(1256, k + k * n + n * l * k + k * n)

    def forward(self, x: torch.Tensor):
        x = f.relu(f.max_pool2d(self.bn_conv1(self.conv1(x)), kernel_size=2))
        x = f.relu(f.max_pool2d(self.bn_conv2(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320 * 3)
        x = f.relu(self.linear1(x))
        x = self.linear2(x)
        x = torch.cat([torch.softmax(x[:, :k], dim=1), x[:, k:]], 1)
        return x


def prepare_input(image):
    image_mod = image.copy()
    x = np.zeros((image_x * image_y,))
    y = np.zeros((image_x * image_y,))
    z = np.zeros((image_x * image_y,))
    counter = 0
    for i, row in enumerate(image_mod):
        for j, a in enumerate(row):
            x[counter] = j
            y[counter] = i
            z[counter] = a
            counter += 1
    img_center_x = image_x / 2
    img_center_y = image_y / 2
    dx = random.randint(-max_random, max_random)
    dy = random.randint(-max_random, max_random)
    hole_beg_x = img_center_x - hole_size_x / 2 + dx
    hole_end_x = img_center_x + hole_size_x / 2 + dx
    hole_beg_y = img_center_y - hole_size_y / 2 + dy
    hole_end_y = img_center_y + hole_size_y / 2 + dy
    mask = []

    # remove center rectangle
    for a in range(image_x * image_y):
        if not hole_beg_x < x[a] < hole_end_x or not hole_beg_y < y[a] < hole_end_y:
            mask.append(a)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    # move points to fill hole
    x_old = np.copy(x)
    y_old = np.copy(y)
    b1 = hole_end_y - hole_end_x
    b2 = hole_end_y + hole_beg_x
    for a in range(len(x)):
        if hole_beg_x <= x[a] <= hole_end_x or hole_beg_y <= y[a] <= hole_end_y:
            if x[a] + b1 > y[a] and -x[a] + b2 > y[a]:
                d = (hole_end_y + hole_beg_y) / (2 * hole_beg_y)
                c = (2 - 2 * d) / hole_size_x
                y[a] *= c * abs(x[a] - (hole_beg_x + hole_end_x) / 2) + d
            elif x[a] + b1 < y[a] and -x[a] + b2 < y[a]:
                d = (hole_end_x + hole_beg_x) / (2 * hole_beg_x)
                c = (2 - 2 * d) / hole_size_y
                x[a] *= c * abs(y[a] - (hole_beg_y + hole_end_y) / 2) + d
            elif x[a] + b1 > y[a] > -x[a] + b2:
                d = (hole_end_y + hole_beg_y) / (2 * hole_beg_y)
                c = (2 - 2 * d) / hole_size_x
                y[a] = image_y - (image_y - y[a]) * c * abs(x[a] - (hole_beg_x + hole_end_x) / 2) + d

            elif x[a] + b1 < y[a] < -x[a] + b2:
                d = (hole_end_x + hole_beg_x) / (2 * hole_beg_x)
                c = (2 - 2 * d) / hole_size_y
                x[a] = image_x - (image_x - x[a]) * c * abs(y[a] - (hole_beg_y + hole_end_y) / 2) + d

    x_2 = np.arange(0, 28, 1)
    y_2 = np.arange(0, 28, 1)
    x_2, y_2 = np.meshgrid(x_2, y_2)
    z_new = interpolate.griddata((x, y), z, (x_2, y_2), method='linear')
    x_new = interpolate.griddata((x, y), x_old, (x_2, y_2), method='linear')
    y_new = interpolate.griddata((x, y), y_old, (x_2, y_2), method='linear')
    return np.stack([z_new, x_new, y_new]), ((hole_beg_x, hole_end_x), (hole_beg_y, hole_end_y))


def loss_function(x: torch.Tensor, orig, boundaries, k, l, n):
    hole_beg_x = boundaries[0][0].int()
    hole_end_x = boundaries[0][1].int()
    hole_beg_y = boundaries[1][0].int()
    hole_end_y = boundaries[1][1].int()
    x = x.view((len(x), -1))
    sum = torch.tensor(0).double().to(device)
    p: torch.Tensor = x[:, :k].reshape(-1, k)
    m: torch.Tensor = x[:, k:k + k * n].reshape(-1, k, n)
    A: torch.Tensor = x[:, k + k * n:k + k * n + n * l * k].reshape(-1, k, n, l)
    d: torch.Tensor = x[:, k + k * n + n * l * k:].reshape(-1, k, n)
    dist = distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(m, A, torch.abs(d))
    layers = torch.stack([orig[i, hole_beg_x[i]:hole_end_x[i], hole_beg_y[i]:hole_end_y[i]] for i in range(len(x))])
    sum = sum - (p.log() + dist.log_prob(layers.reshape(len(layers), 1, -1))).sum()
    return sum
