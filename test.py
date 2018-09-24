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
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import greycomatrix

SUBDATASET_DIR = os.path.join(constants.DATA_DIR, 'subdataset')
TEST_IMAGE = '/home/t3min4l/workspace/texture-classification/data/subdataset/canvas1/canvas1-a-p001.png'


folders = [x[0] for x in os.walk(SUBDATASET_DIR)]

image = cv2.imread(TEST_IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(type(image))

contrast_glcm = np.zeros(image.shape)
dissimilarity_glcm = np.zeros(image.shape)
energy_glcm = np.zeros(image.shape)
correlation_glcm = np.zeros(image.shape)
ASM_glcm = np.zeros(image.shape)

for i in range(image.shape[0]):
    for j in range(image.shape[1])