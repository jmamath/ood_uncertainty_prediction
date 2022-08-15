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
from training.otd_distance import compute_otdd_cifar
from training.gde import compute_gde_cifar10
from training.temperature_scaling import ModelWithTemperature
# def get_device():
#     if torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'
#     return device
# device = get_device()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, choices=["H-distance", "ATC", "Labelwise-H-distance", "OTDD", "GDE"])
    parser.add_argument('--device', type=int, default=0)
    config = parser.parse_args()    
    
    # Set device
    device = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")
    ######################## TRAINING PROCESS ######################## 
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    
    train_data = datasets.CIFAR10('../data/cifar', train=True, transform=preprocess, download=False)
    train_data = MyData(train_data.data, train_data.targets, 'CIFAR10', preprocess)
    
    test_data = datasets.CIFAR10('../data/cifar', train=False, transform=preprocess, download=False)
    test_data = MyData(test_data.data, torch.LongTensor(test_data.targets), 'CIFAR10', preprocess)
    
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=True)
    
    # 1. Verify if a trained model already exist
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
        print("Calibrating the model...")
        scaled_model = ModelWithTemperature(model)
        scaled_model.set_temperature(test_loader, device)
    
    cifar10_accuracies_loc = os.path.join(cifar10_path, "cifar10_accuracies.npy")
    cifar10_accuracies_path = Path(cifar10_accuracies_loc)
    if not cifar10_accuracies_path.is_file():
        print("Computing accuracies on corrupted domains...")
        iid_acc = [test(model, test_loader, device)]
        ood_acc = test_c_cifar(model, base_c_path, corruptions, preprocess, device)
        ood_acc = iid_acc + ood_acc
        np.save(cifar10_accuracies_path, np.array(ood_acc))
    
    ######################## COMPUTE H-DISTANCE ########################
    if config.algorithm == "H-distance":
        cifar10_h_distances_loc = os.path.join(cifar10_path, "cifar10_h_dist_pre.npy")
        cifar10_h_distances_path = Path(cifar10_h_distances_loc)
        if not cifar10_h_distances_path.is_file():
            h_dis = H_distance('CIFAR10', preprocess, n_epochs=10, device=device, pretrained_model=model)
            h_distances = h_dis.distances_cifar10c(train_data, base_c_path, corruptions)
            h_distances = np.array(h_distances)
            np.save(cifar10_h_distances_path, h_distances)

    ######################## COMPUTE LABELWISE H-DISTANCE ########################
    if config.algorithm == "Labelwise-H-distance":
        cifar10_labelwise_h_distances_loc = os.path.join(cifar10_path, "cifar10_l_h_dist_pre.npy")
        cifar10_labelwise_h_distances_path = Path(cifar10_labelwise_h_distances_loc)
        if not cifar10_labelwise_h_distances_path.is_file():
            extended_h = Labelwise_H_distance('CIFAR10', preprocess, id_label_fraction=0.5, ood_label_fraction=0.1, n_epochs=10, device=device, pretrained_model=model)
            divergence_matrices = extended_h.divergences_cifar10c(train_data, base_c_path, corruptions)
            divergence_matrices_path = Path(os.path.join(cifar10_path, "divergence_matrices_cifar10_pretrained"))
            np.save(divergence_matrices_path, divergence_matrices)
            labelwise_h_distances = distances_c(divergence_matrices)
            np.save(cifar10_labelwise_h_distances_path, labelwise_h_distances)
            
    ######################## COMPUTE AVERAGE THRESHOLD CONFIDENCE ########################
    if config.algorithm == "ATC":
        cifar10_atc_loc = os.path.join(cifar10_path, "cifar10_atc.npy")
        cifar10_atc_path = Path(cifar10_atc_loc)
        if not cifar10_atc_path.is_file():
            for param in scaled_model.parameters():
                param.requires_grad = False
            # Make sure that the regressor input dimension match the dataloader batch size
            regressor_input = 256       
            val_loader = DataLoader(test_data, batch_size=regressor_input, shuffle=True, drop_last=True)
            regressor = linearRegression(regressor_input, 1).to(device)
            optimizer = torch.optim.Adam(regressor.parameters(), lr=0.01)
            crit = nn.L1Loss()
            n_epochs = 20
            print("learning a threshold function ...")
            _, loss_mae = train_regressor(regressor, scaled_model, val_loader, n_epochs, crit, optimizer, device)            
            _, corruption_estimation = estimate_c_cifar(scaled_model, regressor, regressor_input, base_c_path, corruptions, preprocess, device)            
            iid_est = estimate_target_risk(test_loader, scaled_model, regressor, device)
            corruption_estimation = [iid_est] + corruption_estimation
            np.save(cifar10_atc_path, corruption_estimation)
            
    ######################## COMPUTE OPTIMAL TRANSPORT DATASET DISTANCE ########################
    if config.algorithm == "OTDD":
        cifar10_otdd_loc = os.path.join(cifar10_path, "cifar10_otdd.npy")
        cifar10_otdd_path = Path(cifar10_otdd_loc)
        if not cifar10_otdd_path.is_file():
            otdd = compute_otdd_cifar(test_data, 5000, base_c_path, corruptions, preprocess, device)
            np.save(cifar10_otdd_path, otdd)
    
    ######################## COMPUTE GENERALIZED DISAGREEMENT EQUALITY ########################
    if config.algorithm == "GDE":
        cifar10_gde_loc = os.path.join(cifar10_path, "cifar10_gde.npy")
        cifar10_gde_path = Path(cifar10_gde_loc)
        if not cifar10_gde_path.is_file():
            print("Train a second model...")  
            model = CifarResNet(BasicBlock, [2,2,2]).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            n_epochs=10
            criterion = nn.CrossEntropyLoss()
            loss, acc = train_model(train_loader, model, criterion, optimizer, None, n_epochs, device)
            print("Saving the model...")
            torch.save(model.state_dict(),  os.path.join(cifar10_path,"cifar10_model_gde"))
            # Evaluate the disagreement between the two models
            model_a = CifarResNet(BasicBlock, [2,2,2]).to(device)
            model_b = CifarResNet(BasicBlock, [2,2,2]).to(device)
            model_a.load_state_dict(torch.load(os.path.join(cifar10_path,"cifar10_model")))
            model_b.load_state_dict(torch.load(os.path.join(cifar10_path,"cifar10_model_gde")))
            
            gde = compute_gde_cifar10(model_a, model_b, test_loader, base_c_path, corruptions, preprocess, device)
            np.save(cifar10_gde_path,gde)            
            


if __name__=='__main__':
    main()
