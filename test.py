import os 
import sys 
import constants
import glob
from random import shuffle
import os
from torch.autograd import Variable
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import cv2
import mahotas as mt
import numpy as np
import threading
import logging
import glcm


SUBDATASET_DIR = os.path.join(constants.DATA_DIR, 'subdataset')
TEST_IMAGE = '/home/t3min4l/workspace/texture-classification/data/subdataset/canvas1/canvas1-a-p001.png'


folders = [x[0] for x in os.walk(SUBDATASET_DIR)]

# print(folders)
del folders[0]

def get_labels(folders):
    substr1 = 'subdataset/'
    interval = len(substr1)
    labels = list()
    for folder in folders:
        idx = folder.find(substr1)
        label = folder[idx+interval:len(folder)]
        labels.append(label)
    return labels

labels = get_labels(folders)
# print(labels)

def build_images_path_label(labels):
    dataset = list()
    for idx, label in enumerate(labels):
        label_dir = os.path.join(SUBDATASET_DIR, label)
        fns = glob.glob(label_dir+'/*.png')
        print(label)
        print(len(fns))
        for fn in fns:
            dataset.append([fn, idx])
    return dataset

def extract_features(image):
    textures = mt.features.haralick(image)
    print(textures)
    ht_mean = textures.mean(axis=0)

    return ht_mean

dataset = build_images_path_label(labels)
shuffle(dataset)
# print(dataset)

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x


image = Image.open(TEST_IMAGE)
tmp = image.getpixel((0,0))
if isinstance(tmp, int) or len(tmp) != 3:
    image = image.convert('RGB')

image = cv2.imread(TEST_IMAGE)
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
cv2.imshow('origin image',image)
cv2.waitKey(5000)
cv2.destroyAllWindows()
image_transformer = transforms.Compose([
        # transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

image_input = image_transformer(image)
image_input = image_input.unsqueeze(0)
print(image_input.shape)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('glcm',glcm)
# cv2.waitKey(5000)
# cv2.destroyWindow()


# model = AlexNet()
# model.to('cpu')
# print(image.shape)
# print(image_input.shape)
# # image_input = image_input.permute(0,3,1,2)
# print(image_input.shape)
# output = model(image_input)
# print(output)
# print(list(model.features.children()))
output = glcm.glcm(image, [1], [2], mode='raw')
print(output.shape)