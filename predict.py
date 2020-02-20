import copy

from PIL import Image
from torchvision.transforms.functional import to_tensor
import numpy as np
from config import *
from utils import prepare_input, Network
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import sys

if __name__ == '__main__':
    assert len(sys.argv) > 1

    model = Network().double().to(device)
    data: dict = torch.load(model_filename)
    model.load_state_dict(data['model_state_dict'])
    mean = data['mean']
    std = data['std']
    img = np.array(Image.open(sys.argv[1]).convert('L'))
    assert img.shape == (28, 28)
    image_orig = copy.deepcopy(img)
    img, boundaries = prepare_input((img - mean)/std)

    with torch.no_grad():
        model.train(False)
        res = model(to_tensor(img).to(device).view(1, *img.shape))
        a = (res[:, k:k + k * n].reshape(k, n) * std + mean).cpu().data.numpy()
        p = res[:, :k].reshape(-1).cpu().data.numpy()

        h = sum(p[i] * a[i] for i in range(k))
        h[h < 0] = 0
        h[h > 255] = 255

    ax = plt.gca()
    (hole_beg_x, hole_end_x), (hole_beg_y, hole_end_y) = boundaries
    rect = patches.Rectangle((hole_beg_y - 0.5, hole_beg_x - 0.5), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    new_image = image_orig.copy()
    new_image[int(hole_beg_x):int(hole_end_x), int(hole_beg_y):int(hole_end_y)] = h.reshape(10, 10)

    plt.imshow(new_image, cmap='gray')
    plt.show()
