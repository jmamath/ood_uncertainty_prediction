# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:09:52 2022

@author: JeanMichelAmath
"""
from training.architectures.lenet import LeNet5
from training.architectures.resnet import CifarResNet, BasicBlock
import torch
from training.utils import MyData, evaluate_model, train_model
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn

######################## H-DISTANCE ########################
class H_distance:
    def __init__(self, dataset_name, preprocess, n_epochs, device):
        """
        Class to compute the Labelwise H-ditance

        Parameters
        ----------
        dataset_name : string
            name of the dataset to use. It will be used in MyData to differentiate how preprocessing is applied 
        preprocess : torch transform
            preprocessing to apply to the dataset
        n_epochs : number of epoch for each cell of the matrix
            DESCRIPTION.
        device : string
            whether to use the cpu or the gpu

        """
        self.dataset_name = dataset_name
        self.preprocess = preprocess
        self.n_epochs = n_epochs
        self.device = device
    
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
      labelA = np.zeros(dataA.shape[0],  dtype=np.longlong)
      labelB = np.ones(dataB.shape[0], dtype=np.longlong)      
      dataAB = np.concatenate((dataA, dataB), 0)
      labelAB = np.concatenate((labelA, labelB), 0)
      X_train, X_test, y_train, y_test = train_test_split(dataAB, labelAB, test_size=0.33, random_state=0)
      # import pdb; pdb.set_trace()
      train_data = MyData(X_train, y_train, self.dataset, self.preprocess)
      test_data = MyData(X_test, y_test, self.dataset, self.preprocess)
      return train_data, test_data
    
    def proximal_A_distance(self, train_data, test_data):
      """
      This function mainly learns to distinguish two different datasets with the training set and report its evaluation
      on the unseen test set
      """
      # We simply train a classifier to distinguish between datasets for 10 epochs
      train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
      test_loader =  DataLoader(test_data, batch_size=512, shuffle=True)
      # import pdb; pdb.set_trace()
      # Training
      if self.dataset_name == "CIFAR":
          model = CifarResNet(BasicBlock, [1,1,1]).to(self.device)
      if self.dataset_name == "MNIST":
          model = LeNet5().to(self.device)
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      crit = nn.CrossEntropyLoss()
      _, _ = train_model(train_loader, model, crit, optimizer, None, self.n_epochs, self.device)
      proximal_distance = evaluate_model(model, test_loader)
      return proximal_distance
    
    def compute_proxy_distance(self, A, B):
      """
      This function will prepare the datasets A and B so that we can compute the proxymal H-distance between them
      Inputs:
        A and B are MyData datasets
      Outupt:
        proxymal A-distance between Ak and Bm.
      """
      AB_train, AB_test = self.prepare_data(A.data, B.data)
      return self.proximal_A_distance(AB_train, AB_test)

    
    # The following function are specific to datasets because those come with different database organisations
    
    def distances_cifar10c(self, source_data, base_path, corruptions):
      """
      Compute the h-distance between the original data and every corrupted dataset.
      We need to have a list with the all the divergence matrices including source vs source
      This will allow us to compute all the distances from the source
      """
      h_distances = []
      source_distance = self.compute_proxy_distance(source_data, source_data)
      h_distances.append(source_distance)
      # np.save('source_matrix', source_distance)
      print("h-distance for origin computed")
      for corr in corruptions:
         # Reference to original data is mutated
         data = np.load(base_path + corr + '.npy')
         targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
         corrupted_data = MyData(data, targets, self.dataset_name, self.preprocess)
    
         h_distance = self.compute_proxy_distance(source_data, corrupted_data)
         h_distances.append(h_distance)
         # np.save("{}".format(corr), h_distance)
         print("h-distance for {}  computed".format(corr))
      return h_distances
   
    def distances_mnist_c(self, source_data, corruption_dir, corruptions):
      """
      Compute the h-distance between the original data and every corrupted dataset.
      We need to have a list with the all the divergence matrices including source vs source
      This will allow us to compute all the distances from the source
      """
      h_distances = []
      for directory, corruption in zip(corruption_dir, corruptions):
        # Reference to original data is mutated
        data = np.load(directory + "/train_images.npy")
        targets = torch.LongTensor(np.load(directory + "/train_labels.npy")).squeeze()
        corrupted_data = MyData(data, targets, self.dataset_name, self.preprocess)
        # import pdb; pdb.set_trace()
        h_distance = self.compute_proxy_distance(source_data, corrupted_data)
        h_distances.append(h_distance)
        # np.save("{}".format(corruption), h_distance)
        print("divergence matrix for {}  computed".format(corruption))
      return h_distances 