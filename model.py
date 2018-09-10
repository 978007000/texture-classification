import os
from torch.autograd import Variable
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import constants


class SDAE(nn.Module):
	"""Stack Denoising Autoencoder"""
	def __init__(self, img_size):
		super(SDAE, self).__init__()
		self.img_size = img_size
		

class MyAlexNet_SDAe(nn.Module):
	def __init__(self, args):

class MyAlexNet(nn.Module):
	def __init__(self, pretrained=False, num_classes = constants.NUM_LABELS):
		super(MyAlexNet, self).__init__()
		if pretrained:
			self.pretrained_model = models.AlexNet(pretrained=True)
			