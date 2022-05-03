# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:10:02 2022

@author: JeanMichelAmath
"""

import torch
import torch.nn.functional as F
from tqdm import trange
from training.utils import compute_accuracy, test, MyData
import numpy as np

######################## AVERAGE THRESHOLD CONFIDENCE ########################

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
      super(linearRegression, self).__init__()
      self.linear1 = torch.nn.Linear(inputSize, 128)
      self.linear2 = torch.nn.Linear(128, 128)
      self.linear3 = torch.nn.Linear(128, 1)

    def forward(self, x):
      x = F.relu(self.linear1(x))
      x = F.relu(self.linear2(x))      
      out = F.relu(self.linear3(x))
      return out 
  
def compute_entropy(p):
    '''
    Computes the entropy of a categorical probability distribution
    It handle the case where one of the class has probability 1. and all the other have 0.
    It sets the convention: 0 * log(0) = 0.
    The function can handle 
    Args:
      p: Float array of any dimension (:, ... ,:, d). The last dimension in each case must be a probability distribution
         i.e it must sums to 1.
    Return
      entropy: Float. The entropy of the probability distribution p.
    '''    
    zero_probs = p == 0
    log_p = torch.log(p)
    log_p[zero_probs] = 0    
    entropy = -(p * log_p).sum(-1)  
    return entropy

def compute_max(pred):    
  return torch.max(pred,1).values    

def train_regressor(regressor, architecture, train_loader, n_epochs, criterion, optimizer, device):
    """
    This function train a regressor that learns a threshold matching the accuracy obtained by architecture on 
    each batch.

    Parameters
    ----------
    regressor : torch Model
        A linear regression neural network taking a batch of scores (Float) as an input 
        and outputing a threshold (Float)
    architecture : torch Model
        A trained model that will transform a batch of real world data (torch Tensor) to a batch of logits 
    train_loader : torch Loader
        A regular torch loader. However the batch size must match the regressor input dim. Because the regressor will compute
        a threshold on a batch of score, approximating the accuracy.
    n_epochs : Int
        Number of epochs to train.
    criterion : torch loss function
        compute the loss obtained from the threshold computed by the regressor and the accuracy.
    optimizer : torch Optimizer
        Minimize the loss previously computed
    device : str
        whether to run on the CPU or the GPU.

    Returns
    -------
    regressor : torch Model
        Trained torch model
    avg_train_mae : list
        list of losses to evaluate training

    """
  
    # to track the training log likelihood as the model trains
    train_mae = []
    # to track the average training log likelihood per epoch as the model trains
    avg_train_mae = [] 
    
    # for epoch in range(1, n_epochs + 1):
    with trange(n_epochs) as pbar:
      for epoch in pbar:
        ###################
        # train the model #
        ###################        
        for _, (batch_x, batch_y) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()            
            batch_x = batch_x.to(device) 
            batch_y = batch_y.to(device)                                   
            logits = architecture(batch_x)
            score = compute_entropy(F.softmax(logits))
            acc = compute_accuracy(logits, batch_y)
            # import pdb; pdb.set_trace()
            # compute MC_sample Monte Carlo predictions       
            ## We perform many sample            
            threshold = regressor(score)                       
            # import pdb; pdb.set_trace()
            loss = criterion(threshold[0], acc)
            # import pdb; pdb.set_trace()            
           # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()                 
            # perform a single optimization step (parameter update)
            optimizer.step() 
            # record training log likelihood, KL and accuracy
            train_mae.append(loss.item())          

        # Get and store descriptive statistics of the training mean absolute error to display the learning behavior                         
        avg_train_mae.append(np.average(train_mae))        
        
        # print training/validation statistics 
        pbar.set_postfix(train_mae=avg_train_mae[-1])
        
        # clear lists to track the monte carlo estimation for the next epoch
        train_mae = []                       
        # if epoch % 20 == 0:
        #   print("Saving model at epoch ", epoch)
        #   torch.save(posterior_model.state_dict(), './'+'{}_state_{}.pt'.format(model.name, epoch))              
                  
    return  regressor, avg_train_mae

def estimate_target_risk(dataloader, architecture, regressor, device):
    """
    Given dataset, this function will compute the threshold to estimate the accuracy

    Parameters
    ----------
    dataloader : torch Loader
        A regular torch loader. However the batch size must match the regressor input dim. Because the regressor will compute
        a threshold on a batch of score, approximating the accuracy.
    architecture : torch Model
        A trained model that will transform a batch of real world data (torch Tensor) to a batch of logits
    regressor : torch Model
        A linear regression neural network taking a batch of scores (Float) as an input 
        and outputing a threshold (Float)
    device : str
        whether to run on the CPU or the GPU.

    Returns
    -------
    Float
        The mean of all threshold computed at every batch
    """
    threshold_final = []
    with torch.no_grad():    
      for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)                                        
        logits = architecture(batch_x)
        score = compute_entropy(F.softmax(logits))       
        threshold = regressor(score).cpu().numpy()
        threshold_final.append(threshold) 
    return np.array(threshold_final).mean()


def estimate_c_cifar(net, regressor, batch_size, base_path, corruptions, preprocess, device):
  """
    This function use the regressor to compute a threshold estimating the accuracy on a given dataset.
  We return both the accuracy and the estimation, to show to the user if the estimation is close to the real accuracy. It is specific to the
  CIFAR-10 and CIFAR-10-C database

    Parameters
    ----------
    net : torch model
        torch model trained on a source domain
    regressor : torch model
        this model predict the expected accuracy based on logits
    batch_size : Int
        the batch size is important because it should match the input size of the regressor
    base_path : string
        path to the corrupted database
    corruptions : list[string]
        a list with the name of every corruption
    preprocess : torch transforms
        to be applied on each corrupted dataset, so that it match training condition
    device : str
        whether to run on the CPU or the GPU.

    Returns
    -------
    corruption_accs : list[float]
        the list of accuracies comnputed on each corrupted domain
    corruption_est : list[float]
        the list of estimated accuracies computed on each corrupted domain
    """
  corruption_accs = []
  corruption_est = []
  for corruption in corruptions:
    # Reference to original data is mutated
    data = np.load(base_path + corruption + '.npy')
    targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
    corrupted_data = MyData(data, targets, preprocess)

    # import pdb; pdb.set_trace()
    test_loader = torch.utils.data.DataLoader(
        corrupted_data,
        batch_size,
        shuffle=True)

    test_loss, test_acc = test(net, test_loader, device)
    test_est = estimate_target_risk(test_loader, net, regressor, device)
    corruption_accs.append(test_acc)
    corruption_est.append(test_est)
    print('{}\n\tTest Loss {:.3f} | Test acc {:.3f} | Estimate acc {:.3f}'.format(
        corruption, test_loss, 100 * test_acc, test_est))
  return corruption_accs, corruption_est

def estimate_c_mnist(net, regressor, batch_size, corruption_dir, corruptions, preprocess, device):
  """
    This function use the regressor to compute a threshold estimating the accuracy on a given dataset.
  We return both the accuracy and the estimation, to show to the user if the estimation is close to the real accuracy. It is specific to the
  MNIST and MNIST-C database

    Parameters
    ----------
    net : torch model
        torch model trained on a source domain
    regressor : torch model
        this model predict the expected accuracy based on logits
    batch_size : Int
        the batch size is important because it should match the input size of the regressor
    corruption_dir : list[Path]
        a list of path to each of the corrupted dataset in the MNIST-C database
    corruptions : list[string]
        a list with the name of every corruption
    preprocess : torch transforms
        to be applied on each corrupted dataset, so that it match training condition
    device : str
        whether to run on the CPU or the GPU.

    Returns
    -------
    corruption_accs : list[float]
        the list of accuracies comnputed on each corrupted domain
    corruption_est : list[float]
        the list of estimated accuracies computed on each corrupted domain
    """
  corruption_accs = []
  corruption_est = []
  for directory, corruption in zip(corruption_dir, corruptions):  
    data = np.load(directory + "/test_images.npy")
    targets = torch.LongTensor(np.load(directory + "/test_labels.npy")).squeeze()
    corrupted_data = MyData(data, targets, 'MNIST', preprocess)

    test_loader = torch.utils.data.DataLoader(
        corrupted_data,
        batch_size,
        shuffle=True)

    test_loss, test_acc = test(net, test_loader, device)
    test_est = estimate_target_risk(test_loader, net, regressor, device)
    corruption_accs.append(test_acc)
    corruption_est.append(test_est)
    print('{}\n\tTest Loss {:.3f} | Test acc {:.3f} | Estimate acc {:.3f}'.format(
        corruption, test_loss, 100 * test_acc, test_est))
  return corruption_accs, corruption_est
