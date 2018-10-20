import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import constants
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
    	self.fns = list()
    	self.labels = list()
    	for fn, label in data:
    		self.fns.append(fn)
    		self.labels.append(label)
    	self.transform = transform

    def __len__(self):
    	return len(self.fns)

    def __getitem__(self, idx):
    	image = Image.open(self.fns[idx])
    	tmp = image.getpixel((0,0))
    	if isinstance(tmp, int) or len(tmp) != 3:
    		image = image.convert('RGB')
    	if self.transform:
    		image = self.transform(image)
        return image, self.labels[idx]
    	# return image, self.labels[idx], self.fns[idx]
