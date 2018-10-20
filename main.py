import os
import sys
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import csv
import pickle
import time

import constants

from model import CustomResnet
from utils import *
from dataset import MyDataset

project_description = "Texture classification - Using transfer learning with ResNet and with stacked model from https://www.sciencedirect.com/science/article/pii/S0893608017302320"

parser = argparse.ArgumentParser(description= project_description)
parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
parser.add_argument('--optim', default='adam', choices=['adam', 'sgd'], type=str, help='Optimizers for model')
parser.add_argument('--interval', default=300, type=int, help='Number of epochs')
parser.add_argument('--weight_decay', default=5e-6, type=float, help='Weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')

parser.add_argument('--train', '-t', action='store_true', help='Training the model')
parser.add_argument('--option', default=0, type=int, choices=[0, 1], help='Option 0: train the model with pretrained ResNet -- Option 1: train the model with stackd model')
parser.add_argument('--depth', default=34, type=int, choices=[18, 34, 50, 101, 152], help='Depth of ResNets to use')
parser.add_argument('--frozen', default=8, type=int, help='Freeze the model until --frozen block')
parser.add_argument('--resume', '-r', action='store_true', help='Resume the model from checkpoints')
parser.add_argument('--predict', '-p', action='store_true', help='Predict the data')
parser.add_argument('--inspect', '-i', action='store_true', help='Inspect the saved model')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start_epoch = 0
best_acc = 0
print('Model using device:{}'.format(device))
ResNets_input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms = {
	'train' : transforms.Compose([
			transforms.RandomSizedCrop(ResNets_input_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
		]),
	'val' : transforms.Compose([
			transforms.Scale(ResNets_input_size),
			transforms.CenterCrop(ResNets_input_size),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
		])
}

folders = [x[0] for x in os.walk(constants.DATASET_PATH)]
labels = get_labels(folders)
train_paths, val_paths, test_paths = build_fns_labels(labels)
print('{}.{}.{}.{}'.format(type(train_paths), len(train_paths), len(val_paths), len(test_paths)))
print(sum([x[1]==6 for x in train_paths]))
train_set = MyDataset(train_paths, transform=transforms['train'])
val_set = MyDataset(val_paths, transform=transform['val'])
test_set = MyDataset(test_set, transform=transform['val'])

def save_convergence_model(save_loss, epoch_loss, epoch):
	save_loss = epoch_loss
	print('Saving convergence model at epoch {}-th with loss {}'.format(epoch, save_loss))
	state = {
		'model' : model.state_dict(),
		'loss' 	: save_loss,
		'epoch'	: epoch
	}

	if not os.path.isdir(constants.CHECKPOINT_DIR):
		os.mkdir(constants.CHECKPOINT_DIR)
	torch.save(state, os.path.join(constants.CHECKPOINT_DIR, 'convergence.t7'))

def save_best_acc_model(save_acc, epoch_acc, epoch):
	save_acc = epoch_acc
	print('Saving best acc model at epoch {}-th with acc {:.3}%'.format(epoch, save_acc))
	state = {
		'model' : model.state_dict(),
		'acc'	: save_acc,
		'epoch'	: epoch
	}

	if not os.path.isdir(constants.CHECKPOINT_DIR):
		os.mkdir(constants.CHECKPOINT_DIR)
	torch.save(state, os.path.join(constants.CHECKPOINT_DIR, 'best_acc_model.t7'))

def train_val(epoch):
