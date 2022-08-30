# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:50:44 2022

@author: JeanMichelAmath
"""

import os
from pathlib import Path
import argparse
from torchvision import transforms as T
from torchvision import models
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.utils import train_model_earlystopping
from camelyon17.camelyon17_preprocessing import camelyon17_v1, get_hospital_data, metadata_df, Camelyon17Dataset, test_augmentations, test_nodes, compute_gde_camelyon_augmentations, compute_gde_camelyon_nodes, test_augmentations_all_severity, compute_gde_camelyon_augmentations_all_severity
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_node', type=int, required=True, choices=[0,1,2,3,4])
    parser.add_argument('--rebalanced', type=bool, default=False)
    parser.add_argument('--all_severity', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    config = parser.parse_args()    
    
    # Set device
    device = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")
    # device_bis = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    ######################## TRAINING PROCESS ########################             
    
    data, label = get_hospital_data(metadata_df, config.training_node)
    
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.16, random_state=0)
    
    train_dataset = Camelyon17Dataset(camelyon17_v1, y_train, X_train)
    test_dataset = Camelyon17Dataset(camelyon17_v1, y_test, X_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 1. Verify if a trained model already exist
    curr_path = os.getcwd()
    camelyon17_path = os.path.join(curr_path, "camelyon17")
    
    camelyon17_model_loc = os.path.join(camelyon17_path, "camelyon17_model_node_{}".format(config.training_node))    
    camelyon17_model_path = Path(camelyon17_model_loc)
    # import pdb; pdb.set_trace()
    if not camelyon17_model_path.is_file():
        print("Training a base model on Camelyon17")
        model = models.densenet121(num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        n_epochs=5
        patience=1
        criterion = nn.CrossEntropyLoss()
        loss, acc = train_model_earlystopping(train_loader, test_loader, model, criterion, optimizer, None, patience, camelyon17_model_loc, n_epochs, device)
        print("Saving the model...")
        # torch.save(model.state_dict(),  camelyon17_model_loc)
    else:
        print("Loading the model...")
        model = models.densenet121(num_classes=2).to(device)        
        model.load_state_dict(torch.load(camelyon17_model_loc))
    
    camelyon17_accuracies_loc = os.path.join(camelyon17_path, "camelyon17_accuracies_node_{}".format(config.training_node))
    camelyon17_accuracies_path = Path(camelyon17_accuracies_loc)
    if not camelyon17_accuracies_path.is_file():
        print("Computing accuracies on corrupted domains...")
        if config.all_severity:
            ood_acc = test_augmentations_all_severity(model, X_test, y_test, device) 
        else:
            ood_acc = test_augmentations(model, X_test, y_test, device) 
        node_acc = test_nodes(model, config.training_node, device, rebalanced=config.rebalanced)
        ood_acc.update(node_acc)
        ood_acc = pd.Series(ood_acc)
        ood_acc.to_csv(camelyon17_accuracies_path)
        
    ######################## COMPUTE GENERALIZED DISAGREEMENT EQUALITY ########################
    camelyon17_gde_loc = os.path.join(camelyon17_path, "camelyon17_gde_node_{}".format(config.training_node))
    camelyon17_gde_path = Path(camelyon17_gde_loc)
    camelyon17_model_bis_loc = os.path.join(camelyon17_path, "camelyon17_model_bis_node_{}".format(config.training_node))
    camelyon17_model_bis_path = Path(camelyon17_model_bis_loc)        
    if not camelyon17_gde_path.is_file():
        if not camelyon17_model_bis_path.is_file():
            print("Train a second model...")  
            model = models.densenet121(num_classes=2).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            n_epochs=5
            patience=1
            criterion = nn.CrossEntropyLoss()
            loss, acc = train_model_earlystopping(train_loader, test_loader, model, criterion, optimizer, None, patience, camelyon17_model_bis_loc, n_epochs, device)

            print("Saving the model...")
            # torch.save(model.state_dict(), camelyon17_model_bis_path)
        else:
            pass
        print("Evaluate the disagreement between the two models")
        model_a = models.densenet121(num_classes=2).to(device)
        model_b = models.densenet121(num_classes=2).to(device)
        model_a.load_state_dict(torch.load(camelyon17_model_loc))
        model_b.load_state_dict(torch.load(camelyon17_model_bis_loc))
        if config.all_severity:
            gde_aug = compute_gde_camelyon_augmentations_all_severity(model_a, model_b, X_test, y_test, device)
        else:
            gde_aug = compute_gde_camelyon_augmentations(model_a, model_b, X_test, y_test, device)
        gde_node = compute_gde_camelyon_nodes(model_a, model_b, config.training_node, device, rebalanced=config.rebalanced)
        gde_aug.update(gde_node)
        gde_aug = pd.Series(gde_aug)
        gde_aug.to_csv(camelyon17_gde_path)


if __name__=='__main__':
    main()    