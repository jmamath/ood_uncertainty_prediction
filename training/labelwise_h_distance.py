# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:48:32 2022

@author: JeanMichelAmath
"""
from training.architectures.lenet import LeNet5
from training.architectures.resnet import CifarResNet, BasicBlock
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
from training.utils import MyData, evaluate_model, train_model
import copy

######################## LABELWISE H-DISTANCE ########################

class Labelwise_H_distance:
    def __init__(self, dataset_name, preprocess, id_label_fraction, ood_label_fraction, n_epochs, device, pretrained_model=None):
        """
        Class to compute the Labelwise H-ditance

        Parameters
        ----------
        dataset_name : string
            name of the dataset to use. It will be used in MyData to differentiate how preprocessing is applied 
        preprocess : torch transform
            preprocessing to apply to the dataset
        id_label_fraction : float in [0,1]
            the percentage of data from the in distribution dataset to keep
        ood_label_fraction : float in [0,1]
            Percentage of data from the out of distribution dataset to keep
        n_epochs : number of epoch for each cell of the matrix
            DESCRIPTION.
        device : string
            whether to use the cpu or the gpu

        """
        self.dataset_name = dataset_name
        self.preprocess = preprocess
        self.id_label_fraction = id_label_fraction
        self.ood_label_fraction = ood_label_fraction
        self.n_epochs = n_epochs
        self.device = device
        self.pretrained_model = pretrained_model

    def filter_label(self, dataset, label_to_keep, label_fraction):
      """
        This function is used to prepare a dataset for the labelwise h-distance. Indeed
        Given a torch dataset, this function will filter only the items associated with the targeted label_to_keep
        and will randomly sample label_fraction of it.

        Parameters
        ----------
        dataset : MyData dataset
            DESCRIPTION.
        label_to_keep : int
            label targeted by our data filtering
        label_fraction : float in [0,1]
            Percentage of data to keep from dataset

        Returns
        -------
        np array
            random label_fraction of dataset from label_to_keep

        """
      # Get all targets
      targets = np.array(dataset.targets)
      # Get indices of the class to keep from dataset
      idx_to_keep = targets==label_to_keep
      # Only keep your desired classes
      labeled_data_partitioned = dataset.data[idx_to_keep]
      idx = np.arange(len(labeled_data_partitioned))
      subset_size = int(len(labeled_data_partitioned) * label_fraction)
      random_subsample_idx = np.random.choice(idx, subset_size)  
      # import pdb; pdb.set_trace()
      return dataset.data[random_subsample_idx]
      
    
    def prepare_data(self, dataA, dataB):
      """
        Prepare data according to the Proximal A-distance method
        see https://arxiv.org/pdf/1505.07818.pdf section 3.2 
        
        This function will give label 0 to dataA and label 1 to dataB. 
        Furthermore it will return a training and test dataset
        The H-distance is basically the ability of an architecture to differentiate between dataA and dataB
        Parameters
        ----------
        dataA : torch dataset
            in distribution dataset
        dataB : torch dataset
            out of distribution dataset

        Returns
        -------
        train_data : torch dataset
            67 % of the merged dataA and dataB
        test_data : torch dataset
            33 % of the merged dataA and dataB
        """
      
      # import pdb; pdb.set_trace()
      labelA = np.zeros(dataA.shape[0],  dtype=np.longlong)
      labelB = np.ones(dataB.shape[0], dtype=np.longlong)      
      dataAB = np.concatenate((dataA, dataB), 0)
      labelAB = np.concatenate((labelA, labelB), 0)
      X_train, X_test, y_train, y_test = train_test_split(dataAB, labelAB, test_size=0.33, random_state=0)
      train_data = MyData(X_train, y_train, self.dataset_name, self.preprocess)
      test_data = MyData(X_test, y_test, self.dataset_name, self.preprocess)
      return train_data, test_data
    
    def proximal_A_distance(self, train_data, test_data):
      """
      This function mainly learns to distinguish two different datasets with the training set and report its evaluation
      on the unseen test set. If the class has a pretrained model, then we will only learn the last fully connected layer
      """
      # We simply train a classifier to distinguish between datasets for 10 epochs
      train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
      test_loader =  DataLoader(test_data, batch_size=512, shuffle=True)
      # import pdb; pdb.set_trace()
      # Training
      if self.dataset_name == "CIFAR10":
          if self.pretrained_model == None:
              model = CifarResNet(BasicBlock, [2,2,2], num_classes=2).to(self.device)
              optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
          else:
              model = self.prepare_pretrained_model()
              optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)
      if self.dataset_name == "MNIST":
          if self.pretrained_model == None:
              model = LeNet5(num_classes=2).to(self.device)
              optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
          else:
              model = self.prepare_pretrained_model()
              optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)
      if self.dataset_name == "IMAGENET":
          if self.pretrained_model == None:
              model = models.resnet50().to(self.device)
              num_ftrs = model.fc.in_features
              model.fc = nn.Linear(num_ftrs, 2).to(self.device)
              optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
          else:
              model = self.prepare_pretrained_model()
              optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)
      crit = nn.CrossEntropyLoss()
      _, _ = train_model(train_loader, model, crit, optimizer, None, self.n_epochs, self.device)
      proximal_distance = evaluate_model(model, test_loader, self.device)
      return proximal_distance
  
    def prepare_pretrained_model(self):
        """ Prepare model for transfer learning """
        model = copy.deepcopy(self.pretrained_model)
        for param in model.parameters():
            param.requires_grad = False    
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2).to(self.device)
        return model
    
    def compute_labeled_proxy_distance(self, A_k, B_m):
      """
      This function will take the k-th labeled data from dataset A and the m-th labeled data from dataset B, 
      and will further compute the proxymal A-distance between them.
      Inputs:
        A_k and B_m are tuples of (MyData datasets, Int)
      Outupt:
        proxymal A-distance between Ak and Bm.
      """
      A, k = A_k
      B, m = B_m
      Ak = self.filter_label(A, k, self.id_label_fraction)
      Bm = self.filter_label(B, m, self.ood_label_fraction)
      # import pdb; pdb.set_trace()
      AkBm_train, AkBm_test = self.prepare_data(Ak, Bm)
      return self.proximal_A_distance(AkBm_train, AkBm_test)
    
    def compute_labeled_matrix_proxy_distance(self, dataA, dataB, nb_label):
      """
      Computes a square matrix of dimension (nb_label, nb_label)
      each cell (i,j) of the matrix is the H-distance between dataA label i and dataB label j
      """
      distance_matrix = np.empty((nb_label, nb_label))
      for i in range(nb_label):
        for j in range(nb_label):
          distance_matrix[i,j] = self.compute_labeled_proxy_distance((dataA,i), (dataB,j))
      return distance_matrix
    
    # The following function are specific to datasets because those come with different database organisations
    
    def divergences_imagenet_c(self, source_data, directories):
      """
      Compute the h-distance between the original data and every corrupted dataset.
      We need to have a list with the all the divergence matrices including source vs source
      This will allow us to compute all the distances from the source
      """
      divergence_matrices = []
      directories.pop("train")
      for name, directory in directories.items():
        data = np.load(directories[name] + "/images.npy")
        targets = torch.LongTensor(np.load(directories[name] + "/labels.npy")).squeeze()
        corrupted_data = MyData(data, targets, self.dataset_name, self.preprocess)
        # import pdb; pdb.set_trace()
        divergence_matrix = self.compute_labeled_matrix_proxy_distance(source_data, corrupted_data, source_data.num_class)
        divergence_matrices.append(divergence_matrix)
        # np.save("{}".format(corruption), h_distance)
        print("divergence matrix for {}  computed".format(name))
      return divergence_matrices 
    
    def divergences_cifar10c(self, source_data, base_path, corruptions):
      """
      Compute the divergence matrices between the original data and every corrupted dataset.
      We need to have a list with the all the divergence matrices including source vs source
      This will allow us to compute all the distances from the source
      """
      divergence_matrices = []
      source_matrix = self.compute_labeled_matrix_proxy_distance(source_data, source_data, source_data.num_class)
      divergence_matrices.append(source_matrix)
      # np.save('source_matrix', source_matrix)
      print("divergence matrix for origin computed")
      for corr in corruptions:
         # Reference to original data is mutated
         data = np.load(base_path + corr + '.npy')
         targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
         corrupted_data = MyData(data, targets, self.dataset_name, self.preprocess)
    
         divergence_matrix = self.compute_labeled_matrix_proxy_distance(source_data, corrupted_data, source_data.num_class)
         divergence_matrices.append(divergence_matrix)
         # np.save("{}".format(corr), divergence_matrix)
         print("divergence matrix for {}  computed".format(corr))
      return divergence_matrices
   
    def divergences_mnist_c(self, source_data, corruption_dir, corruptions):
      """
      Compute the divergence matrices between the original data and every corrupted dataset.
      We need to have a list with the all the divergence matrices including source vs source
      This will allow us to compute all the distances from the source
      """
      divergence_matrices = []
      for directory, corruption in zip(corruption_dir, corruptions):
        # Reference to original data is mutated
        data = np.load(directory + "/train_images.npy")
        targets = torch.LongTensor(np.load(directory + "/train_labels.npy")).squeeze()
        corrupted_data = MyData(data, targets, self.dataset_name, self.preprocess)
        # import pdb; pdb.set_trace()
        divergence_matrix = self.compute_labeled_matrix_proxy_distance(source_data, corrupted_data, nb_label=source_data.num_class)
        divergence_matrices.append(divergence_matrix)
        # np.save("{}".format(corruption), divergence_matrix)
        print("divergence matrix for {}  computed".format(corruption))
      return divergence_matrices

######################## DISTRIBUTIONAL DISTANCES ######################## 
def KL_multivariate_Bernoulli(P,Q):
    """
    Computes the Kullback-Leibler divergence between two matrices

    Parameters
    ----------
    P : np array of shape N,N
        where N is the number of class
    Q : np array of shape N,N
        where N is the number of class

    Returns
    -------
    Float
        sum of all the kl div of every cell of the matrices.

    """
    def correct_log_P(P):
        # It sets the convention: 0 * log(0) = 0.
        # useful to compute the KL thereafeter
        zero_probs = P == 0
        log_P = np.log(P)
        log_P[zero_probs] = 0  
        return log_P

    def correct_ratio(P,Q):
        # This is to avoid division by 0
        zero_Q = Q == 0
        if Q[zero_Q].size != 0:
            Q[zero_Q] = 1e-6
        return P/Q

    first_term = P * correct_log_P(correct_ratio(P, Q))
    second_term = (1-P) * correct_log_P(correct_ratio((1-P), (1-Q))) 
    # import pdb; pdb.set_trace()
    return np.sum(first_term + second_term)

def distance_matrix_space(A, B, norm_type=None):
  """
  Computes a functional between two matrix A and B of same shape based on a norm type
  This functional weights more the diagonal of the matrix difference than the upper and lower triangular matrices differences.
  Because it seems that errors on the diagonals seems to be correlated to higher accuracy gap between domain A and B.

    Parameters
    ----------
    A : numpy matrix
    B : numpy matrix
    norm_type : {non-zero int, inf, -inf, ‘fro’, ‘nuc’, 'kl'}, optional
        Order of the numpy norm 

    Returns
    -------
    Float
        d(A,B)
    """
  diag_A = np.diag(A)
  diag_B = np.diag(B)
  C = A.shape[0]
  if norm_type == "kl":
      constant = np.diag(np.ones(C)/2)
      first_factor = KL_multivariate_Bernoulli(diag_A, diag_B)
      second_factor = KL_multivariate_Bernoulli( (A-np.diag(diag_A)+constant ), (B-np.diag(diag_B)+constant))
  else:
      first_factor = np.linalg.norm(diag_A-diag_B, norm_type)
      second_factor = np.linalg.norm( (A-np.diag(diag_A)) - (B-np.diag(diag_B)), norm_type)
  return ((C-1)/C) * first_factor + (1/C)* second_factor



def distances_c(divergence_matrices, norm_type="kl"):
    """
    Given a set of matrices, will compute the distance of every matrix from the first one

    Parameters
    ----------
    divergence_matrices : np.array of shape (C,N,N) with C the number of domains or corruptions considered
        and N the number of classes.
    norm_type : {non-zero int, inf, -inf, ‘fro’, ‘nuc’, 'kl'}, optional
        Order of the numpy norm

    Returns
    -------
    distances : list[float]
        a list of every distances from origin [d(A0,A0), d(A0,A1), ... d(A0,AC)]
    """
    distances = []
    origin = divergence_matrices[0]
    for matrix in divergence_matrices:
        distances.append(distance_matrix_space(origin, matrix, norm_type))
    return distances

