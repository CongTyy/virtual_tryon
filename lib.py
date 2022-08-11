import os
import shutil
from glob import glob
import pandas as pd
from tqdm import tqdm
import time
import datetime
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
from torch import autograd
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils.spectral_norm as spectral_norm
# import tensorflow as tf

patch = (1, 256 // 2 ** 4, 192 // 2 ** 4)

lr = 1e-4
betas = (0.5, 0.999)
lambda_gp = 10

Tensor = torch.cuda.FloatTensor
EPOCH = 100
device = 'cuda:0'
batch_size = 32
n_critic = 5
F_BIAS = 1
I_BIAS = 1

random.seed(1111)
torch.manual_seed(1111)
