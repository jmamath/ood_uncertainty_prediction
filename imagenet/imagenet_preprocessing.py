# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:34:09 2022

@author: JeanMichelAmath
"""

import os
from pathlib import Path
import numpy as np
import torch
from training.utils import MyData
from torchvision import transforms as T

imagenet_multidomain = Path(os.getcwd(), '../data/imagenet-multidomain')

names = ["train", "val", "adversarial", "sketch", "sketch_2", "art", "painting", "graffiti", "graphic", "cartoon", "sculpture", "embroidery", "misc", "toy",  
               "brightness", "contrast", "defocus", "elastic_transform", "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise",
               "jpeg_compression", "pixelate", "shot_noise", "snow", "zoom_blur"]

directories = {name: os.path.join(imagenet_multidomain, name) for name in names}

# data0 = np.load(directories["adversarial"] + "/images.npy")
# targets0 = torch.LongTensor(np.load(directories["adversarial"] + "/labels.npy")).squeeze()


# data = np.load(directories["val"] + "/images.npy")
# targets = torch.LongTensor(np.load(directories["val"] + "/labels.npy")).squeeze()

# data1 = np.load(directories["val"] + "/images.npy")
# targets1 = torch.LongTensor(np.load(directories["val"] + "/labels.npy")).squeeze()



# data2 = np.load(directories["art"] + "/images.npy")
# targets2 = torch.LongTensor(np.load(directories["art"] + "/labels.npy")).squeeze()

# data3 = np.load(directories["shot_noise"] + "/images.npy")
# targets3 = torch.LongTensor(np.load(directories["shot_noise"] + "/labels.npy")).squeeze()

# data4 = np.load(directories["sketch_2"] + "/images.npy")
# targets4 = torch.LongTensor(np.load(directories["sketch_2"] + "/labels.npy")).squeeze()

# preprocess = T.Compose(
#     [T.RandomHorizontalFlip(),
#                 T.ToTensor(),  # Converting cropped images to tensors
#                 T.Normalize(mean=[0.485, 0.456, 0.406], 
#                             std=[0.229, 0.224, 0.225])])

# train_data = MyData(data, targets, "ImageNet", preprocess)