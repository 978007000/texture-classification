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

from model import CustomResnet, net_frozen
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

train_loader = torch.utils.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, batch_sampler=None)
val_loader = torch.utils.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True, batch_sampler=None)
test_loader = torch.utils.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True, batch_sampler=None)

def save_convergence_model(save_loss, epoch):
	print('Saving convergence model at epoch {}-th with loss {}'.format(epoch, save_loss))
	state = {
		'model' : model.state_dict(),
		'loss' 	: save_loss,
		'epoch'	: epoch
	}

	if not os.path.isdir(constants.CHECKPOINT_DIR):
		os.mkdir(constants.CHECKPOINT_DIR)
	torch.save(state, os.path.join(constants.CHECKPOINT_DIR, 'convergence.t7'))

def save_best_acc_model(save_acc, epoch):
	print('Saving best acc model at epoch {}-th with acc {:.3}%'.format(epoch, save_acc))
	state = {
		'model' : model.state_dict(),
		'acc'	: save_acc,
		'epoch'	: epoch
	}

	if not os.path.isdir(constants.CHECKPOINT_DIR):
		os.mkdir(constants.CHECKPOINT_DIR)
	torch.save(state, os.path.join(constants.CHECKPOINT_DIR, 'best_acc_model.t7'))

save_loss = 0
save_acc = 0


model = CustomResnet(args.depth, num_classes)
criterion = nn.CrossEntropyLoss()

model, optimizer = net_frozen(args, model)

model.to(device)



def train_val(epoch):
	print('======================================\n')
	print('=>   Training at epoch {}-th'.format(epoch))

	global save_loss
	global save_acc

	train_loss = 0 
	train_correct = 0
	total = 0

	for batch_id, (images, labels) in enumerate(train_loader):
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = net(images)
		loss = criterion(outputs, labels)
		loss.criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += labels.size(0)
		train_correct += predicted.eq(labels).sum().item

	epoch_train_correct = train_correct/total
	epoch_train_loss = train_loss/(batch_id+1)
	print('Loss: {:.3} || Error: {:.3} %'.format(epoch_loss, error*100))

	if epoch == 0:
		save_loss = epoch_loss
		print('==> Saving convergence model ...')
		save_convergence_model(save_loss, model)
	elif save_loss > train_loss:
		save_loss = epoch_loss
		print('==> Saving ...')
		save_convergence_model(save_loss, model)

	print('========================================\n')
	print('==> Validating ...')
	val_loss = 0
	val_correct = 0 
	total = 0
	model.eval()

	for batch_id, (images, labels) in enumerate(val_loader):
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		loss = criterion(outputs, labels)

		_, predicted = outputs.max(1)
		val_loss += loss
		total += labels.size(0)
		val_correct += predicted.eq(labels).sum().item()

	epoch_val_correct = val_correct/total
	print('Acc: {.3} %'.format(epoch_train_correct*100))
	if epoch_val_correct > save_acc:
		save_acc = epoch_val_correct
		print('==> Saving best-acc-on-val model ...')
		save_best_acc_model(save_acc, model)

def predict(conv=args.conv):
	assert os.path.isdir(constants.CHECKPOINT_DIR), 'Error: Model is not availabel'
	if conv:
		checkpoint = torch.load(os.path.join(constants.CHECKPOINT_DIR, 'convergence.t7'))
		model.load_state_dict(checkpoint['model'])
	else:
		checkpoint = torch.load(os.path.join(constants.CHECKPOINT_DIR, 'best_acc_model.t7'))
		model.load_state_dict(checkpoint['model'])

	torch.set_grad_enabled(False)
	net.eval()
	test_correct = 0
	for batch_id, (images, labels) in enumerate(test_loader):
		images, labels = images.to(device), labels.to(device)
		outputs = models(images)

		_, predicted = outputs.max(1)
		test_correct += predicted.eq(labels).sum().item()

	print('Accuracy on test data: {}%'.format((test_correct/total)*100))

