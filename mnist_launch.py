# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:42:21 2022

@author: JeanMichelAmath
"""

import os
from training.architectures.lenet import LeNet5
from pathlib import Path
import argparse
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mnist.mnist_c_preprocessing import corruption_dir, corruptions, identity_dir
from training.utils import MyData, train_model, test_c_mnist
from training.labelwise_h_distance import Labelwise_H_distance, distances_c
from training.h_distance import H_distance
from training.atc import linearRegression, train_regressor, estimate_c_mnist

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device
device = get_device()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, choices=["H-distance", "ATC", "Labelwise-H-distance"])
    config = parser.parse_args()
    ######################## TRAINING PROCESS ######################## 
    preprocess = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
    
    data = np.load(identity_dir + "/train_images.npy")
    targets = torch.LongTensor(np.load(identity_dir + "/train_labels.npy")).squeeze()
    train_data = MyData(data, targets, "MNIST", preprocess)
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    
    data = np.load(identity_dir + "/test_images.npy")
    targets = torch.LongTensor(np.load(identity_dir + "/test_labels.npy")).squeeze()
    test_data = MyData(data, targets, "MNIST", preprocess)
    
    # 1. Verify if there a trained model already exist
    curr_path = os.getcwd()
    mnist_path = os.path.join(curr_path, "mnist")
    
    mnist_model_loc = os.path.join(mnist_path, "mnist_model")    
    mnist_model_path = Path(mnist_model_loc)
    if not mnist_model_path.is_file():
        print("Training a base model on MNIST")
        model = LeNet5().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        n_epochs=10
        criterion = nn.CrossEntropyLoss()
        loss, acc = train_model(train_loader, model, criterion, optimizer, None, n_epochs, device)
        print("Saving the model...")
        torch.save(model.state_dict(),  os.path.join(mnist_path,"mnist_model"))
    else:
        print("Loading the model...")
        model = LeNet5().to(device)
        model.load_state_dict(torch.load(os.path.join(mnist_path,"mnist_model")))

    
    mnist_accuracies_loc = os.path.join(mnist_path, "mnist_accuracies")
    mnist_accuracies_path = Path(mnist_accuracies_loc)
    if not mnist_accuracies_path:
        print("Computing accuracies on corrupted domains...")
        ood_acc = test_c_mnist(model, corruption_dir, corruptions)
        accuracies_path = Path(os.path.join(mnist_path, "mnist_accuracies"))
        np.save(accuracies_path, np.array(ood_acc))
    
    ######################## COMPUTE H-DISTANCE ########################
    if config.algorithm == "H-distance":
        mnist_h_distances_loc = os.path.join(mnist_path, "mnist_h_distances")
        mnist_h_distances_path = Path(mnist_h_distances_loc)
        if not mnist_h_distances_path.is_file():
            h_dis = H_distance('MNIST', preprocess, n_epochs=10, device=device)
            h_distances = h_dis.distances_mnist_c(train_data, corruption_dir, corruptions)
            h_distances = np.array(h_distances)
            h_distances_path = Path(os.path.join(mnist_path, "mnist_h_distances"))
            np.save(h_distances_path, h_distances)

    ######################## COMPUTE LABELWISE H-DISTANCE ########################
    if config.algorithm == "Labelwise-H-distance":
        mnist_labelwise_h_distances_loc = os.path.join(mnist_path, "mnist_labelwise_h_distances")
        mnist_labelwise_h_distances_path = Path(mnist_labelwise_h_distances_loc)
        if not mnist_labelwise_h_distances_path.is_file():
            extended_h = Labelwise_H_distance('MNIST', preprocess, id_label_fraction=0.5, ood_label_fraction=0.1, n_epochs=10, device=device)
            divergence_matrices = extended_h.divergences_mnist_c(train_data, corruption_dir, corruptions)       
            divergence_matrices_path = Path(os.path.join(mnist_path, "divergence_matrices_mnist"))
            np.save(divergence_matrices_path, divergence_matrices)
            labelwise_h_distances = distances_c(divergence_matrices)
            labelwise_h_distances_path = Path(os.path.join(mnist_path, "mnist_labelwise_h_distances"))
            np.save(labelwise_h_distances_path, labelwise_h_distances)
            
    ######################## COMPUTE AVERAGE THRESHOLD CONFIDENCE ########################
    if config.algorithm == "ATC":
        mnist_atc_loc = os.path.join(mnist_path, "mnist_atc")
        mnist_atc_path = Path(mnist_atc_loc)
        if not mnist_atc_path.is_file():
            for param in model.parameters():
                param.requires_grad = False

            # Make sure that the regressor input dimension match the dataloader batch size
            regressor_input = 256       
            val_loader = DataLoader(test_data, batch_size=regressor_input, shuffle=True, drop_last=True)
            regressor = linearRegression(regressor_input, 1).to(device)
            optimizer = torch.optim.Adam(regressor.parameters(), lr=0.01)
            crit = nn.L1Loss()
            n_epochs = 20
            print("learning a threshold function ...")
            _, loss_mae = train_regressor(regressor, model, val_loader, n_epochs, crit, optimizer, device)            
            _, corruption_estimation = estimate_c_mnist(model, regressor, regressor_input, corruption_dir, corruptions, preprocess, device)            
            mnist_atc_path = Path(os.path.join(mnist_path, "mnist_atc"))
            np.save(mnist_atc_path, corruption_estimation)
            


if __name__=='__main__':
    main()
