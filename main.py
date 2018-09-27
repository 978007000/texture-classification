import os
import sys
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.nn as nn 
import torch.optim as optim 
import torch.utils.data as utilsData
import torchvision.models as models
import torchvision.transforms as transforms
import argpase
import csv
import pickle
import time

import constants

from model import CustomResnet
from utils import *
from dataset import MyDataset

project_description = "Texture classification - Using transfer learning with ResNet and with stacked model from https://www.sciencedirect.com/science/article/pii/S0893608017302320"

parser = argpase.ArgumentParser(description= project_description)
parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
parser.add_argument('--optim', default='adam', choices=['adam', 'sgd'], type=str, help='Optimizers for project')
parser.add_argument('--interval', default=256, type=int, help='Number of epochs')
parser.add_argument('--weight-decay', default=5e-6, type=float, help='Weight decay')
parser.add_argument('--frozen', default=8, type=int, help='Freeze the model until --frozen block')
parser.add_argument('--train', '-t', action='store_true', help='Training the model')
parser.add_argument('--resume', '-r', action='store_true', help='Resume the model from checkpoints')
parser.add_argument('--predict', '-p', action='store_true', help='Predict the data')
parser.add_argument('--inspect', 'i', action='store_true', help='Inspect the saved model')
args = parser.parser_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start_epoch = 0
best_acc = 0
print('Model using device:{}'.format(device))
input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms = {
	'train' : transforms.Compose([
			transforms.RandomSizedCrop(input_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()
			transforms.Normalize(mean, std)
		])
	'val' : transforms.Compose([
			transforms.Scale(input_size),
			transforms.CenterCrop(input_size),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
		])
}

folders = [x[0] for x in os.walk(constants.DATASET_PATH)]
labels = get_labels(folders)
dataset = build_fns_labels(labels)



