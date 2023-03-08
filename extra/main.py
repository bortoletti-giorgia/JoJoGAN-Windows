# Initial setup

import torch
from torchvision import transforms, utils
from PIL import Image
import math
import random
import os
import gdown
import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
import wandb

# JoJoGAN Specific Import
from model import *
from e4e_projection import projection as e4e_projection
from util import *

#matplotlib inline

from copy import deepcopy

import argparse
import matplotlib.pyplot as plt

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--modelname", help="where the model is saved", required=True)
parser.add_argument("--testimagename", help="name for the test image with extension", required=True)

args = parser.parse_args()

source_filename = args.testimagename 
modelname = args.modelname

# Creating local folders for local content creation and management
os.makedirs('inversion_codes', exist_ok=True)
os.makedirs('style_images', exist_ok=True)
os.makedirs('style_images_aligned', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# CUDA
torch.backends.cudnn.benchmark = True
device = 'cuda' #@param ['cuda', 'cpu']

# Load StyleGAN model
# Already trained using FFHQ dataset - 70K faces
latent_dim = 512

# Load original generator
original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
original_generator.load_state_dict(ckpt["g_ema"], strict=False)
mean_latent = original_generator.mean_latent(10000)

# to be finetuned generator
generator = deepcopy(original_generator)

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Source Image
filename = source_filename

filepath = f'test_input/{filename}'

name = strip_path_extension(filepath)+'.pt'

# aligns and crops face from the source image
aligned_face = align_face(filepath)

# my_w = restyle_projection(aligned_face, name, device, n_iters=1).unsqueeze(0)
my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)

plt.imsave("results/aligned_face.jpg", get_image(aligned_face))

# Test with pretrained style
pretrained = modelname

#@markdown Preserve color tries to preserve color of original image by limiting family of allowable transformations. Otherwise, the stylized image will inherit the colors of the reference images, leading to heavier stylizations.
preserve_color = False #@param{type:"boolean"}

ckpt = f'{pretrained}.pt'

ckpt = torch.load(os.path.join('models', ckpt), map_location=lambda storage, loc: storage)

generator.load_state_dict(ckpt, strict=False)

#@title Generate results
n_sample =  5#@param {type:"number"}
seed = 3000 #@param {type:"number"}

torch.manual_seed(seed)
with torch.no_grad():
    generator.eval()
    z = torch.randn(n_sample, latent_dim, device=device)
    my_sample = generator(my_w, input_is_latent=True)

# display reference images
style_path = f'style_images_aligned/{pretrained}.jpg'

style_image = transform(Image.open(style_path)).unsqueeze(0).to(device)
face = transform(aligned_face).unsqueeze(0).to(device)

my_output = torch.cat([style_image, face, my_sample], 0)
plt.imsave("results/final_sample.jpg", get_image(utils.make_grid(my_output, normalize=True, range=(-1, 1))))
