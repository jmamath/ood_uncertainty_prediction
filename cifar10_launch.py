# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:46:48 2022

@author: JeanMichelAmath
"""

import os
from training.architectures.resnet import CifarResNet, BasicBlock
from pathlib import Path
from torchvision import datasets
import argparse
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cifar10.cifar10_c_preprocessing import corruptions, base_c_path
from training.utils import MyData, train_model, test, test_c_cifar
from training.labelwise_h_distance import Labelwise_H_distance, distances_c
from training.h_distance import H_distance
from training.atc import linearRegression, train_regressor, estimate_c_cifar, estimate_target_risk

# def get_device():
#     if torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'
#     return device
# device = get_device()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, choices=["H-distance", "ATC", "Labelwise-H-distance"])
    parser.add_argument('--device', required=True, choices=["cuda:0", "cuda:1"])
    config = parser.parse_args()
    device = config.device
    ######################## TRAINING PROCESS ######################## 
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    
    train_data = datasets.CIFAR10('../data/cifar', train=True, transform=preprocess, download=False)
    train_data = MyData(train_data.data, train_data.targets, 'CIFAR10', preprocess)
    
    test_data = datasets.CIFAR10('../data/cifar', train=False, transform=preprocess, download=False)
    test_data = MyData(test_data.data, test_data.targets, 'CIFAR10', preprocess)
    
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=True)
    
    # 1. Verify if there a trained model already exist
    curr_path = os.getcwd()
    cifar10_path = os.path.join(curr_path, "cifar10")
    
    cifar10_model_loc = os.path.join(cifar10_path, "cifar10_model")    
    cifar10_model_path = Path(cifar10_model_loc)
    if not cifar10_model_path.is_file():
        print("Training a base model on CIFAR10")
        model = CifarResNet(BasicBlock, [2,2,2]).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        n_epochs=10
        criterion = nn.CrossEntropyLoss()
        loss, acc = train_model(train_loader, model, criterion, optimizer, None, n_epochs, device)
        print("Saving the model...")
        torch.save(model.state_dict(),  os.path.join(cifar10_path,"cifar10_model"))
    else:
        print("Loading the model...")
        model = CifarResNet(BasicBlock, [2,2,2]).to(device)
        model.load_state_dict(torch.load(os.path.join(cifar10_path,"cifar10_model")))

    
    cifar10_accuracies_loc = os.path.join(cifar10_path, "cifar10_accuracies")
    cifar10_accuracies_path = Path(cifar10_accuracies_loc)
    if not cifar10_accuracies_path:
        print("Computing accuracies on corrupted domains...")
        iid_acc = [test(model, test_loader)]
        ood_acc = test_c_cifar(model, base_c_path, corruptions, preprocess)
        ood_acc = iid_acc + ood_acc
        accuracies_path = Path(os.path.join(cifar10_path, "cifar10_accuracies"))
        np.save(accuracies_path, np.array(ood_acc))
    
    ######################## COMPUTE H-DISTANCE ########################
    if config.algorithm == "H-distance":
        cifar10_h_distances_loc = os.path.join(cifar10_path, "cifar10_h_distances")
        cifar10_h_distances_path = Path(cifar10_h_distances_loc)
        if not cifar10_h_distances_path.is_file():
            h_dis = H_distance('CIFAR10', preprocess, n_epochs=10, device=device)
            h_distances = h_dis.distances_cifar10c(train_data, base_c_path, corruptions)
            h_distances = np.array(h_distances)
            h_distances_path = Path(os.path.join(cifar10_path, "cifar10_h_distances"))
            np.save(h_distances_path, h_distances)

    ######################## COMPUTE LABELWISE H-DISTANCE ########################
    if config.algorithm == "Labelwise-H-distance":
        cifar10_labelwise_h_distances_loc = os.path.join(cifar10_path, "cifar10_labelwise_h_distances")
        cifar10_labelwise_h_distances_path = Path(cifar10_labelwise_h_distances_loc)
        if not cifar10_labelwise_h_distances_path.is_file():
            extended_h = Labelwise_H_distance('CIFAR10', preprocess, id_label_fraction=0.5, ood_label_fraction=0.1, n_epochs=10, device=device)
            divergence_matrices = extended_h.divergences_cifar10c(train_data, base_c_path, corruptions)
            divergence_matrices_path = Path(os.path.join(cifar10_path, "divergence_matrices_cifar10"))
            np.save(divergence_matrices_path, divergence_matrices)
            labelwise_h_distances = distances_c(divergence_matrices)
            labelwise_h_distances_path = Path(os.path.join(cifar10_path, "cifar10_labelwise_h_distances"))
            np.save(labelwise_h_distances_path, labelwise_h_distances)
            
    ######################## COMPUTE AVERAGE THRESHOLD CONFIDENCE ########################
    if config.algorithm == "ATC":
        cifar10_atc_loc = os.path.join(cifar10_path, "cifar10_atc")
        cifar10_atc_path = Path(cifar10_atc_loc)
        if not cifar10_atc_path.is_file():
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
            _, corruption_estimation = estimate_c_cifar(model, regressor, regressor_input, base_c_path, corruptions, preprocess, device)            
            iid_est = estimate_target_risk(test_loader, model, regressor, device)
            corruption_estimation = [iid_est] + corruption_estimation
            cifar10_atc_path = Path(os.path.join(cifar10_path, "cifar10_atc"))
            np.save(cifar10_atc_path, corruption_estimation)
            


if __name__=='__main__':
    main()
