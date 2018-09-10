import os
from torch.autograd import Variable
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim


class SDAE(nn.Module):
	"""Stack Denoising Autoencoder"""
	def __init__(self, img_size):
		super(SDAE, self).__init__()
		self.img_size = img_size
		

class MyAlexNet_SDAe(nn.Module)