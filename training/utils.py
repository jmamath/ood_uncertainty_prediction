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
from tqdm import trange
from PIL import Image

# def get_device():
#     if torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'
#     return device
# device = get_device()

######################## DATASET CLASS ######################## 
# @title Dataset class
class MyData(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, targets, dataset_name, preprocess=transforms.ToTensor(), num_class=10):
        """"Initialization, here display serves to show an example
        if false, it means that we intend to feed the data to a model"""
        self.targets = targets
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
        if self.dataset_name == "CIFAR10":
            # import pdb; pdb.set_trace()
            X = Image.fromarray(X)
        if self.dataset_name == "MNIST":
            X = Image.fromarray(X.squeeze())
        if self.dataset_name == "IMAGENET":
            # import pdb; pdb.set_trace()
            X = Image.fromarray(X)
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

def evaluate_model(model, loader, device):
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

    ###################
    # train the model #
    ################### 
    model.train()
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

def train_model_earlystopping(train_loader, valid_loader, model, criterion, optimizer, scheduler, patience, model_location, n_epochs, device):

    # to track the training log likelihood as the model trains
    train_log_likelihood = []
    # to track the validation log likelihood as the model trains
    valid_log_likelihood = []
    # to track the average training log likelihood per epoch as the model trains
    avg_train_log_likelihood = []
    # to track the average validation log likelihood per epoch as the model trains
    avg_valid_log_likelihood = [] 
    
    # to track the training accuracy as the model trains
    train_accuracies = []
    # to track the validation loss as the model trains
    valid_accuracies = []
    # to track the average training loss per epoch as the model trains
    avg_train_accuracies = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_accuracies = [] 
    
    # initialize the early_stopping object    
    early_stopping = EarlyStopping(metric="val_loss", patience=patience, location=model_location, verbose=True)

    ###################
    # train the model #
    ################### 
    model.train()
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
            train_accuracies.append(acc)
        # Get descriptive statistics of the training log likelihood, the training accuracy and the KL over MC_sample                       
        # Store the descriptive statistics to display the learning behavior 
        avg_train_log_likelihood.append(np.average(train_log_likelihood))
        avg_train_accuracies.append(np.average(train_accuracies))
        
        ######################    
        # validate the model #
        ######################    
        with torch.no_grad():          
          for j, (x, target) in enumerate(valid_loader,1):
              target = target.to(device)
              x = x.to(device)
              # calculate the loss and accuracy
              output = model(x)
              loss = criterion(output, target)  
              acc = compute_accuracy(output, target).cpu().numpy()
              # record validation loss
              valid_log_likelihood.append(loss.item())
              valid_accuracies.append(acc) 
                
       # calculate average loss, accuracie and whateve the divergence tracked over one epoch of validation, 
        # the average are stored in a separate variable        
        avg_valid_log_likelihood.append(np.average(valid_log_likelihood))
        avg_valid_accuracies.append(np.average(valid_accuracies))
        
        
        epoch_len = len(str(n_epochs))

        # print training/validation statistics 
        pbar.set_postfix(train_log_likelihood=train_log_likelihood[-1],
                         valid_log_likelihood=valid_log_likelihood[-1], 
                         train_acc=train_accuracies[-1], 
                         valid_acc=valid_accuracies[-1])                
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_log_likelihood[-1], model)

        # clear lists to track next epoch
        train_log_likelihood = []
        valid_log_likelihood = []
        train_accuracies = []  
        valid_accuracies = []
        
        if early_stopping.early_stop:
            print("Early stopping")
            break 
    
    return avg_train_log_likelihood, avg_train_accuracies



# @title Early Stopping class
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, metric="val_loss", patience=7, location=None, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            metric (string): the metric to track, validation loss or validation accuracy
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_min = np.Inf
        self.metric = metric
        self.delta = delta
        self.location = location

    def __call__(self, metric, model):
        # if self.metric == "val_loss":        
        score = -metric
        # else:
        #   score = metric
        # 1st iteration
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, self.location, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, self.location, model)
            self.counter = 0

    def save_checkpoint(self, metric, location, model):
        '''Saves model when validation loss decrease.
        Each model carries a name, so we use it to store the states'''
        
        if self.verbose:
            print(f'{self.metric} decreased ({self.metric_min:.6f} --> {metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), location)
        self.metric_min = metric


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

def test_imagenet_multidomain(net, directories, preprocess, device):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  directories.pop("train")
  for name, directory in directories.items():  
    data = np.load(directories[name] + "/images.npy")
    targets = torch.LongTensor(np.load(directories[name] + "/labels.npy")).squeeze()
    corrupted_data = MyData(data, targets, "IMAGENET", preprocess)
    
    test_loader = torch.utils.data.DataLoader(
        corrupted_data,
        batch_size=64,
        shuffle=True)

    test_loss, test_acc = test(net, test_loader, device)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        name, test_loss, 100 - 100. * test_acc))
  return corruption_accs

def test_c_cifar(net, base_path, corruptions, preprocess, device):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for corruption in corruptions:
    # Reference to original data is mutated
    data = np.load(base_path + corruption + '.npy')
    # data = torch.FloatTensor(data).permute(0,3,2,1)
    targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
    corrupted_data = MyData(data, targets, "CIFAR10", preprocess)

    # import pdb; pdb.set_trace()
    test_loader = torch.utils.data.DataLoader(
        corrupted_data,
        batch_size=128,
        shuffle=True)

    test_loss, test_acc = test(net, test_loader, device)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))
  return corruption_accs

def test_c_mnist(net, corruption_dir, corruptions, preprocess, device):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for directory, corruption in zip(corruption_dir, corruptions):  
    data = np.load(directory + "/test_images.npy")
    data = torch.FloatTensor(data).permute(0,3,2,1)
    targets = torch.LongTensor(np.load(directory + "/test_labels.npy")).squeeze()
    corrupted_data = MyData(data, targets, "MNIST", preprocess)

    test_loader = torch.utils.data.DataLoader(
        corrupted_data,
        batch_size=512,
        shuffle=True)

    test_loss, test_acc = test(net, test_loader, device)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))
  return corruption_accs


