# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:46:09 2022

@author: JeanMichelAmath
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from training.utils import MyData


def compute_disagreement(y_a, y_b):
  total = len(y_a)
  _, pred_a = torch.max(F.softmax(y_a), 1)
  _, pred_b = torch.max(F.softmax(y_b), 1)
  disagreement = (pred_a != pred_b).sum().cpu().numpy() / total
  return disagreement  
    
def compute_gde_loader(model_a, model_b, loader, device):
    disagreement_loader = []
    for i, (x, y) in enumerate(loader):
        y = y.to(device)
        x = x.to(device)    
        y_a = model_a(x)
        y_b = model_b(x)
        disagreement = compute_disagreement(y_a, y_b)
        disagreement_loader.append(disagreement)
    disagreement_loader = np.array(disagreement_loader)
    return np.mean(disagreement_loader)


def compute_gde_mnist(model_a, model_b, corruption_dir, corruptions, preprocess, device):
  gde = []
  for directory, corruption in zip(corruption_dir, corruptions):
    # Reference to original data is mutated
    data = np.load(directory + "/train_images.npy")
    targets = torch.LongTensor(np.load(directory + "/train_labels.npy")).squeeze()
    corrupted_data = MyData(data, targets, "MNIST", preprocess)
    
    loader = torch.utils.data.DataLoader(
        corrupted_data,
        batch_size=512,
        shuffle=True)

    gde.append(compute_gde_loader(model_a, model_b, loader, device))
    print('{} GDE: {}'.format(corruption, gde[-1]))
  return gde


def compute_gde_cifar10(model_a, model_b, source_loader, base_path, corruptions, preprocess, device):
  gde = []
  gde.append(compute_gde_loader(model_a, model_b, source_loader, device))
  for corruption in corruptions:
    # Reference to original data is mutated
    data = np.load(base_path + corruption + '.npy')
    targets = torch.LongTensor(np.load(base_path + 'labels.npy')).squeeze()
    corrupted_data = MyData(data, targets, "CIFAR10", preprocess)
    
    loader = torch.utils.data.DataLoader(
        corrupted_data,
        batch_size=512,
        shuffle=True)

    gde.append(compute_gde_loader(model_a, model_b, loader, device))
    print('{} GDE: {}'.format(corruption, gde[-1]))
  return gde


def compute_gde_imagenet(model_a, model_b, directories, preprocess, device):
  directories.pop("train")
  gde = []
  for name, directory in directories.items():
    # Reference to original data is mutated
    data = np.load(directories[name] + "/images.npy")
    targets = torch.LongTensor(np.load(directories[name] + "/labels.npy")).squeeze()
    corrupted_data = MyData(data, targets, "IMAGENET", preprocess)
    
    loader = torch.utils.data.DataLoader(
        corrupted_data,
        batch_size=16,
        shuffle=True)

    gde.append(compute_gde_loader(model_a, model_b, loader, device))
    print('{} GDE: {}'.format(name, gde[-1]))
  return gde
       