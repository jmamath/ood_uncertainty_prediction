# -*- coding: utf-8 -*-
"""
Created on Mon May  9 07:47:10 2022

@author: JeanMichelAmath
"""

import numpy as np
import torch
from training.utils import MyData
from torchvision import transforms as T

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device
device = get_device()

# from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance

# Instantiate distance

def get_distance(src_dataset, target_dataset, maxsamples = 1000):
    dist = DatasetDistance(src_dataset ,target_dataset,
                           inner_ot_method = 'exact',
                           debiased_loss = True,
                           p = 2, entreg = 1e-1,
                           device='cuda')

    d = dist.distance(maxsamples = maxsamples)
    return d


# src = MyData(data, targets, "IMAGENET", preprocess)
# tgt = = MyData(data4, targets, "IMAGENET", preprocess)

def compute_otdd_specific(source_data, dataset_size, directories, name, preprocess, device):
  """
  compute the OTDD from source data to every corrupted data in MNIST-C
  """
  # Reference to original data is mutated
  data = np.load(directories[name] + "/images.npy")
  targets = torch.LongTensor(np.load(directories[name] + "/labels.npy")).squeeze()
  corrupted_data = MyData(data[:dataset_size], targets, "IMAGENET", preprocess)
    
  maxsamples = min(1000, len(data))
  otdd = get_distance(source_data, corrupted_data, maxsamples).item()
  print('{} OTDD: {}'.format(name, otdd))
  return otdd

# compute_otdd_specific(test_data, 5000, directories, "art", preprocess, device)

def compute_otdd_imagenet(source_data, dataset_size, directories, preprocess, device):
  """
  compute the OTDD from source data to every corrupted data in MNIST-C
  """
  otdd = []
  directories.pop("train")
  for name, directory in directories.items():
    # Reference to original data is mutated
    data = np.load(directories[name] + "/images.npy")
    targets = torch.LongTensor(np.load(directories[name] + "/labels.npy")).squeeze()
    corrupted_data = MyData(data[:dataset_size], targets, "IMAGENET", preprocess)
    
    maxsamples = min(1000, len(data))
    otdd.append(get_distance(source_data, corrupted_data, maxsamples).item())
    print('{} OTDD: {}'.format(name, otdd[-1]))
  return otdd

def compute_otdd_cifar(source_data, dataset_size, base_path, corruptions, preprocess, device):
  """
  compute the OTDD from source data to every corrupted data in CIFAR10-C
  """
  otdd = []
  otdd.append(get_distance(source_data, source_data).item())
  print('Source OTDD: {}'.format(otdd[-1]))
  for corruption in corruptions:
    # Reference to original data is mutated
    data = np.load(base_path + corruption + '.npy')
    targets = torch.LongTensor(np.load(base_path + 'labels.npy')).squeeze()
    corrupted_data = MyData(data[:dataset_size], targets[:dataset_size], "CIFAR10", preprocess)

    otdd.append(get_distance(source_data, corrupted_data).item())
    print('{} OTDD: {}'.format(corruption, otdd[-1]))
  return otdd

def compute_otdd_mnist(source_data, dataset_size, corruption_dir, corruptions, preprocess, device):
  """
  compute the OTDD from source data to every corrupted data in MNIST-C
  """
  otdd = []
  for directory, corruption in zip(corruption_dir, corruptions):
    # Reference to original data is mutated
    data = np.load(directory + "/train_images.npy")
    targets = torch.LongTensor(np.load(directory + "/train_labels.npy")).squeeze()
    corrupted_data = MyData(data[:dataset_size], targets[:dataset_size], "MNIST", preprocess)

    otdd.append(get_distance(source_data, corrupted_data).item())
    print("divergence matrix for {}  computed".format(corruption))
    print('{} OTDD: {}'.format(corruption, otdd[-1]))
  return otdd
