import os
from torch.autograd import Variable
import torchvision.models as models
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import constants


class SDAE(nn.Module):
	"""Stack Denoising Autoencoder"""
	def __init__(self, img_size):
		super(SDAE, self).__init__()
		self.img_size = img_size
		

# class MyAlexNet_SDAe(nn.Module):
# 	def __init__(self, args):

class MyAlexNet(nn.Module):
	def __init__(self, pretrained=False, num_classes = constants.NUM_LABELS):
		super(MyAlexNet, self).__init__()
		self.pretrained_model = models.AlexNet(pretrained=pretrained)
		self.features = self.pretrained_model.features
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
    	x = x.view(x.size(0), 256 * 7 * 7)
    	return self.classifier(x)
    
    def frozen_until(self, to_layer=1):
    	print('Frozen pretrained model to the first sequential layer')
    	for child in self.features.children():
    		for param in child.parameters():
    			param.require_grad = False

class CustomResnet(nn.Module):
	def __init__(self, depth, num_classes):
		super(CustomResnet, self).__init__()
		if depth == 18:
			self.pretrained_model = models.resnet18(pretrained=True)
		elif depth == 34:
			self.pretrained_model = models.resnet34(pretrained=True)
		elif depth == 50:
			self.pretrained_model = models.resnet50(pretrained=True)
		elif depth == 152:
			self.pretrained_model = models.resnet152(pretrained=True)
		self.num_features = models.fc.in_features

		self.shared = nn.Sequential(*list(self.pretrained_model.children())[:-1])
		self.target = nn.Sequential(nn.Linear(self.num_features, num_classes))
	def forward(self, x):
		x = self.shared(x)
		x = torch.squeeze(x)
		return self.target(x)

	def frozen_until(self, frozen_layer):
		print('Frozen shared part of pretrained model to {}-th layer.'.format(frozen_layer))

		child_counter = 0
		for child in self.shared.children():
			if child_counter <= frozen_layer:
				print("Child ", child_counter, " was frozen")
				for param in child.parameters():
					param.require_grad() = False
				# frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
            	print("Child ", child_counter, " was not frozen")
            	for param in child.parameters():
            		param.require_grad() =True
            child_counter += 1

def net_frozen(args, model):
	print('----------------------------------------------------------')
	model.frozen_until()
	init_lr = args.lr 
	if args.optim == 'adam':
		optimizer = optim.Adam(filter(lambda p: p.require_grad, model.parameters()), lr=init_lr, weight_decay=args.weight_decay)
	elif args.optim == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.require_grad, model.parameters()). lr=init_lr, weight_decay=args.weight_decay, momentum=0.9)
	print('----------------------------------------------------------')
	return model, optimizer