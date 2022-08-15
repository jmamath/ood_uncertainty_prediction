# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:15:28 2022

@author: JeanMichelAmath
"""

import os
from pathlib import Path
import argparse
from torchvision import transforms as T
from torchvision import models
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from imagenet.imagenet_preprocessing import directories
from training.utils import MyData, train_model, test_imagenet_multidomain
from training.labelwise_h_distance import Labelwise_H_distance, distances_c
from training.h_distance import H_distance
from training.atc import linearRegression, train_regressor, estimate_target_risk, estimate_c_imagenet
from training.otd_distance import compute_otdd_imagenet
from training.gde import compute_gde_imagenet
from training.temperature_scaling import ModelWithTemperature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, choices=["H-distance", "ATC", "Labelwise-H-distance", "OTDD", "GDE"])
    parser.add_argument('--device', type=int, default=0)
    config = parser.parse_args()    
    
    # Set device
    device = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")
    # device_bis = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    ######################## TRAINING PROCESS ######################## 
    preprocess = T.Compose(
    [T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])
        
    
    data = np.load(directories["train"] + "/images.npy")
    targets = torch.LongTensor(np.load(directories["train"] + "/labels.npy")).squeeze()
    train_data = MyData(data, targets, "IMAGENET", preprocess)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    data = np.load(directories["val"] + "/images.npy")
    targets = torch.LongTensor(np.load(directories["val"] + "/labels.npy")).squeeze()
    test_data = MyData(data, targets, "IMAGENET", preprocess)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    # 1. Verify if a trained model already exist
    curr_path = os.getcwd()
    imagenet_path = os.path.join(curr_path, "imagenet")
    
    imagenet_model_loc = os.path.join(imagenet_path, "imagenet_model")    
    imagenet_model_path = Path(imagenet_model_loc)
    if not imagenet_model_path.is_file():
        print("Training a base model on ImageNet")
        model = models.resnet50(pretrained=True).to(device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 23).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        n_epochs=10
        criterion = nn.CrossEntropyLoss()
        loss, acc = train_model(train_loader, model, criterion, optimizer, None, n_epochs, device)
        print("Saving the model...")
        torch.save(model.state_dict(),  os.path.join(imagenet_path,"imagenet_model"))
    else:
        print("Loading the model...")
        model = models.resnet50().to(device)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 23).to(device)
        model.load_state_dict(torch.load(os.path.join(imagenet_path,"imagenet_model")))
        print("Calibrating the model...")
        scaled_model = ModelWithTemperature(model)
        scaled_model.set_temperature(test_loader, device)

    
    imagenet_accuracies_loc = os.path.join(imagenet_path, "imagenet_accuracies.npy")
    imagenet_accuracies_path = Path(imagenet_accuracies_loc)
    if not imagenet_accuracies_path.is_file():
        print("Computing accuracies on corrupted domains...")
        ood_acc = test_imagenet_multidomain(model, directories, preprocess, device)
        np.save(imagenet_accuracies_path, np.array(ood_acc))
    
    ######################## COMPUTE H-DISTANCE ########################
    if config.algorithm == "H-distance":
        imagenet_h_distances_loc = os.path.join(imagenet_path, "imagenet_h_dist_pre.npy")
        imagenet_h_distances_path = Path(imagenet_h_distances_loc)
        if not imagenet_h_distances_path.is_file():
            h_dis = H_distance('IMAGENET', preprocess, n_epochs=10, device=device, batch_size=64, pretrained_model=model)
            h_distances = h_dis.distances_imagenet_c(train_data, directories)
            h_distances = np.array(h_distances)
            np.save(imagenet_h_distances_path, h_distances)

    ######################## COMPUTE LABELWISE H-DISTANCE ########################
    if config.algorithm == "Labelwise-H-distance":
        imagenet_labelwise_h_distances_loc = os.path.join(imagenet_path, "imagenet_l_h_dist_pre.npy")
        imagenet_labelwise_h_distances_path = Path(imagenet_labelwise_h_distances_loc)
        if not imagenet_labelwise_h_distances_path.is_file():
            extended_h = Labelwise_H_distance('IMAGENET', preprocess, id_label_fraction=0.5, ood_label_fraction=1., n_epochs=10, device=device, pretrained_model=model)
            divergence_matrices = extended_h.divergences_imagenet_c(train_data, directories)
            divergence_matrices_path = Path(os.path.join(imagenet_path, "divergence_matrices_imagenet_pretrained"))
            np.save(divergence_matrices_path, divergence_matrices)
            labelwise_h_distances = distances_c(divergence_matrices)
            np.save(imagenet_labelwise_h_distances_path, labelwise_h_distances)
            
    ######################## COMPUTE AVERAGE THRESHOLD CONFIDENCE ########################
    if config.algorithm == "ATC":
        imagenet_atc_loc = os.path.join(imagenet_path, "imagenet_atc.npy")
        imagenet_atc_path = Path(imagenet_atc_loc)
        if not imagenet_atc_path.is_file():
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
            _, corruption_estimation = estimate_c_imagenet(scaled_model, regressor, regressor_input, directories, preprocess, device)            
            # iid_est = estimate_target_risk(test_loader, model, regressor, device)
            # corruption_estimation = [iid_est] + corruption_estimation
            np.save(imagenet_atc_path, corruption_estimation)
            
    ######################## COMPUTE OPTIMAL TRANSPORT DATASET DISTANCE ########################
    if config.algorithm == "OTDD":
        imagenet_otdd_loc = os.path.join(imagenet_path, "imagenet_otdd.npy")
        imagenet_otdd_path = Path(imagenet_otdd_loc)
        if not imagenet_otdd_path.is_file():
            otdd = compute_otdd_imagenet(test_data, 5000, directories, preprocess, device)
            np.save(imagenet_otdd_path, otdd)

    ######################## COMPUTE GENERALIZED DISAGREEMENT EQUALITY ########################
    if config.algorithm == "GDE":
        imagenet_gde_loc = os.path.join(imagenet_path, "imagenet_gde.npy")
        imagenet_gde_path = Path(imagenet_gde_loc)
        imagenet_model_bis_loc = os.path.join(imagenet_gde_path, "imagenet_model_gde")    
        imagenet_model_bis_path = Path(imagenet_model_bis_loc)        
        if not imagenet_gde_path.is_file():
            if not imagenet_model_path.is_file():
                print("Train a second model...")  
                model = models.resnet50(pretrained=True).to(device)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 23).to(device)
                optimizer = torch.optim.Adam(model.parameters())
                n_epochs=10
                criterion = nn.CrossEntropyLoss()
                loss, acc = train_model(train_loader, model, criterion, optimizer, None, n_epochs, device)
                print("Saving the model...")
                torch.save(model.state_dict(),  os.path.join(imagenet_path,"imagenet_model_gde"))
            else:
                pass
            # Evaluate the disagreement between the two models
            model_a = models.resnet50().to(device)
            num_ftrs = model_a.fc.in_features
            model_a.fc = nn.Linear(num_ftrs, 23).to(device)
            model_b = models.resnet50().to(device)
            num_ftrs = model_b.fc.in_features
            model_b.fc = nn.Linear(num_ftrs, 23).to(device)
            model_a.load_state_dict(torch.load(os.path.join(imagenet_path,"imagenet_model")))
            model_b.load_state_dict(torch.load(os.path.join(imagenet_path,"imagenet_model_gde")))
            
            gde = compute_gde_imagenet(model_a, model_b, directories, preprocess, device)
            np.save(imagenet_gde_path,gde)            
            

if __name__=='__main__':
    main()