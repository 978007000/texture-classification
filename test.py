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
from skimage.feature import greycomatrix, greycoprops

SUBDATASET_DIR = os.path.join(constants.DATA_DIR, 'subdataset')
TEST_IMAGE = '/home/t3min4l/workspace/texture-classification/data/subdataset/linsseeds1/linseeds1-a-p001.png'


folders = [x[0] for x in os.walk(SUBDATASET_DIR)]

image = cv2.imread(TEST_IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(type(image))

fig = plt.figure()
fig.suptitle('GLCM textures')

ax = plt.subplot(241)
plt.axis('off')
ax.set_title('Original Image')
plt.imshow(image, cmap='gray')

contrast_glcm = np.zeros(image.shape)
dissimilarity_glcm = np.zeros(image.shape)
homogeneity_glcm = np.zeros(image.shape)
energy_glcm = np.zeros(image.shape)
correlation_glcm = np.zeros(image.shape)
ASM_glcm = np.zeros(image.shape)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if i < 2 or j < 2:
            continue

        if i > (contrast_glcm.shape[0] -3) or (j > contrast_glcm.shape[0] - 3):
            continue

        glcm_window = image[i-2: i+3, j-2:j+3]

        glcm = greycomatrix(glcm_window, [1], [0], symmetric=True, normed=True)

        contrast_glcm[j, j] = greycoprops(glcm, 'contrast')
        dissimilarity_glcm[i, j] = greycoprops(glcm, 'dissimilarity')
        homogeneity_glcm[i, j] = greycoprops(glcm, 'homogeneity')
        energy_glcm[i, j] = greycoprops(glcm, 'energy')
        correlation_glcm[i, j] = greycoprops(glcm, 'correlation')
        ASM_glcm[i, j] = greycoprops(glcm, 'ASM')

        glcm = None
        glcm_window = None

texturelist = {1: 'contrast', 2: 'dissimilarity', 3: 'homogeneity', 4: 'energy', 5: 'correlation', 6: 'ASM'}
for key in texturelist:
    ax = plt.subplot(2,3,key)
    plt.axis('off')
    ax.set_title(texturelist[key])
    plt.imshow(eval(texturelist[key] + '_glcm'), cmap='gray')

plt.show()