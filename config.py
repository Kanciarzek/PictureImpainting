import torch

image_x: int = 28
image_y: int = image_x
hole_size_x: int = 10
hole_size_y: int = hole_size_x
max_random: int = 4
n_of_samples: int = 5000

batch_size: int = 50
epoch: int = 50
lr: float = 0.001
val_split: float = 0.2

random_seed: int = 10
shuffle_dataset: bool = True

# network parameters
k: int = 3
l: int = 1
n: int = hole_size_x * hole_size_y

model_filename = 'model667.pth'

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
