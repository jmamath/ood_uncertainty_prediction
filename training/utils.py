# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:37:20 2022

@author: JeanMichelAmath
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import numpy as np
import torch.nn as nn

from sklearn.model_selection import train_test_split
from tqdm import trange
from PIL import Image

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device
device = get_device()

######################## DATASET CLASS ######################## 
# @title Dataset class
class MyData(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, targets, dataset_name, preprocess=transforms.ToTensor(), num_class=10):
        """"Initialization, here display serves to show an example
        if false, it means that we intend to feed the data to a model"""
        self.targets = targets
        # if not isinstance(data, torch.Tensor):
        #     data = torch.Tensor(data)
        self.data = data
        self.preprocess = preprocess
        self.dataset_name = dataset_name
        self.num_class = num_class

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.targets)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X, y = self.data[index], self.targets[index]  
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image, this allow preprocessing specific to all dataset to be applied
        if self.dataset_name == "CIFAR":
            X = Image.fromarray(X)
        if self.dataset_name == "MNIST":
            X = Image.fromarray(X.squeeze())
        if self.preprocess:
            X = self.preprocess(X)        
        # Load data and get label
        return X,y       
       
        
######################## TRAINING FUNCTIONS ######################## 
def compute_accuracy(pred, y):
  _, predicted = torch.max(F.softmax(pred), 1)
  total = len(pred)
  correct = (predicted == y).sum()
  # we use accuracy in torch tensor because it is used to learn
  # in the simple training loop, otherwise use correct.cpu().numpy()
  accuracy = correct / total 
  return accuracy  

def evaluate_model(model, loader):
  with torch.no_grad():       
    acc_final = []
    for x, y in loader: # batch_level   
      x = x.to(device)
      y = y.to(device)        
      predictions = model(x)            
      accuracy = compute_accuracy(predictions, y).cpu().numpy()
      acc_final.append(accuracy)        
  return np.array(acc_final).mean()  

def train_model(train_loader, model, criterion, optimizer, scheduler, n_epochs, device):

    # to track the training log likelihood as the model trains
    train_log_likelihood = []
    # to track the average training log likelihood per epoch as the model trains
    avg_train_log_likelihood = [] 
    # to track the training accuracy as the model trains
    train_acc = []
    # to track the average acc per epoch as the model trains
    avg_train_acc = [] 
    # switch to train mode
    model.train()
    # for epoch in range(1, n_epochs + 1):
    with trange(n_epochs) as pbar:
      for epoch in pbar:
        for i, (x, target) in enumerate(train_loader):
            target = target.to(device)
            x = x.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            output = model(x)
            loss = criterion(output, target)
            # measure accuracy and record loss
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
             # record training log likelihood, KL and accuracy
            train_log_likelihood.append(loss.item())   
            acc = compute_accuracy(output, target).cpu().numpy() 
            train_acc.append(acc)
        # Get descriptive statistics of the training log likelihood, the training accuracy and the KL over MC_sample                       
        # Store the descriptive statistics to display the learning behavior 
        avg_train_log_likelihood.append( np.average(train_log_likelihood) )
        avg_train_acc.append(np.average(train_acc))
                
        # print training/validation statistics 
        pbar.set_postfix(train_log_likelihood=avg_train_log_likelihood[-1], acc=avg_train_acc[-1])
        
        # clear lists to track the monte carlo estimation for the next epoch
        train_log_likelihood = []  
        train_acc = []
    
    return avg_train_log_likelihood, avg_train_acc


######################## EVALUATION FUNCTIONS ########################

def test(net, test_loader, device):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.to(device), targets.to(device)
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


# def test_c_cifar(net, test_data, base_path, corruptions):
#   """Evaluate network on given corrupted dataset."""
#   corruption_accs = []
#   for corruption in corruptions:
#     # Reference to original data is mutated
#     test_data.data = np.load(base_path + corruption + '.npy')
#     test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
#     # import pdb; pdb.set_trace()
#     test_loader = torch.utils.data.DataLoader(
#         test_data,
#         batch_size=128,
#         shuffle=True)

#     test_loss, test_acc = test(net, test_loader)
#     corruption_accs.append(test_acc)
#     print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
#         corruption, test_loss, 100 - 100. * test_acc))
#   return corruption_accs


def test_c_cifar(net, base_path, corruptions, preprocess):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for corruption in corruptions:
    # Reference to original data is mutated
    data = np.load(base_path + corruption + '.npy')
    # data = torch.FloatTensor(data).permute(0,3,2,1)
    targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
    corrupted_data = MyData(data, targets, preprocess)

    # import pdb; pdb.set_trace()
    test_loader = torch.utils.data.DataLoader(
        corrupted_data,
        batch_size=128,
        shuffle=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))
  return corruption_accs

def test_c_mnist(net, corruption_dir, corruptions):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for directory, corruption in zip(corruption_dir, corruptions):  
    data = np.load(directory + "/test_images.npy")
    data = torch.FloatTensor(data).permute(0,3,2,1)
    targets = torch.LongTensor(np.load(directory + "/test_labels.npy")).squeeze()
    corrupted_data = MyData(data, targets)

    test_loader = torch.utils.data.DataLoader(
        corrupted_data,
        batch_size=512,
        shuffle=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))
  return corruption_accs


