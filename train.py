import copy
import numpy as np
import torchvision
from torch.optim import Adam
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm

from config import *
from utils import prepare_input, loss_function, ImageSet, Network

data = torchvision.datasets.MNIST(".", download=True, train=True)
new_data = torchvision.datasets.MNIST(".", download=True, train=False, transform=None)
mean = (data.data.type(torch.float32) / 255).mean().item()
std = (data.data.type(torch.float32) / 255).std().item()
data.transform = Compose([Lambda(lambda x: (np.array(x).reshape((28, 28)) - mean) / std)])

images = []
images_orig = []
boundaries = []

for i, (image, label) in tqdm(enumerate(data), desc='Preparing dataset', total=n_of_samples, position=0, leave=True):
    if i == n_of_samples:
        break
    images_orig.append(image)
    torch_image, bound = prepare_input(image)
    images.append(torch.from_numpy(torch_image))
    boundaries.append(bound)

dataset_size: int = len(images)
indices: list = list(range(dataset_size))
split: int = int(np.floor(val_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler: SubsetRandomSampler = SubsetRandomSampler(train_indices)
test_sampler: SubsetRandomSampler = SubsetRandomSampler(val_indices)

my_set = ImageSet(images, images_orig, boundaries)
train_loader: DataLoader = DataLoader(my_set, batch_size=batch_size, sampler=train_sampler)
val_loader: DataLoader = DataLoader(my_set, batch_size=batch_size, sampler=test_sampler)

model: Network = Network().double().to(device)
optimizer = Adam(model.parameters(), lr)

min_loss = float('inf')
best_model = copy.deepcopy(model)
for e in range(epoch):
    train_loss = 0
    val_loss = 0
    model.train(True)
    for x, orig, bound in tqdm(train_loader, desc='Training: ', position=0, leave=True):
        x = x.to(device)
        orig = orig.to(device)
        optimizer.zero_grad()
        result = model(x.double())
        loss = loss_function(result, orig, bound, k, l, n)
        train_loss += loss
        loss.backward()
        optimizer.step()
    model.train(False)
    with torch.no_grad():
        for x, orig, bound in tqdm(val_loader, desc='Validation: ', position=0, leave=True):
            x = x.to(device)
            orig = orig.to(device)
            result = model(x.double())
            loss = loss_function(result, orig, bound, k, l, n)
            val_loss += loss
    if min_loss > val_loss:
        min_loss = val_loss
        best_model = copy.deepcopy(model)
    print("Epoch {} completed, train loss: {}".format(e, train_loss))
    print('Validation loss: {}'.format(val_loss))

torch.save({'model_state_dict': best_model.state_dict(), 'mean': mean, 'std': std}, model_filename)
