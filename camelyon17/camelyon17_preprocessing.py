# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:53:31 2022

@author: JeanMichelAmath
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import training.augmentations as augmentations
from training.utils import test
from sklearn.model_selection import train_test_split
from training.gde import compute_gde_loader


def process_results_node(node):
    curr_path = os.getcwd()
    accuracies = pd.read_csv(os.path.join(curr_path, "camelyon17_accuracies_node_" + str(node)))
    accuracies.rename(columns={'Unnamed: 0':"domain", '0':"accuracy"}, inplace=True)
    accuracies = accuracies.set_index("domain")
    gde = pd.read_csv(os.path.join(curr_path, "camelyon17_gde_node_" + str(node)))
    gde.rename(columns={'Unnamed: 0':"domain", '0':"gde"}, inplace=True)
    gde = gde.set_index("domain")
    results = accuracies.join(gde)
    results.to_csv("camelyon17_node_"+str(node))
    return results

camelyon17_v1 = Path(os.getcwd(), '../data/camelyon17_v1.0')
metadata_df = pd.read_csv(os.path.join(camelyon17_v1, 'metadata.csv'), index_col=0, dtype={'patient': 'str'})



def get_hospital_data(metadata, hospital):
    input_array = [f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            for patient, node, x, y in
            metadata[metadata["node"]==hospital].loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]
    y_array = torch.LongTensor(metadata[metadata["node"]==hospital]['tumor'].values)
    return input_array, y_array


class Camelyon17Dataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_dir, targets, input_array, perturbation=None, num_class=2):
        """"Initialization, here display serves to show an example
        if false, it means that we intend to feed the data to a model"""
        self.targets = targets
        self.data_dir = data_dir
        self.input_array = input_array
        self.perturbation = perturbation
        self.num_class = num_class        

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.targets)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X, y = self.get_input(index), self.targets[index]  
        if self.perturbation is None:
            X = transforms.ToTensor()(X) 
        else: 
            if isinstance(X, Image.Image) and (self.perturbation in augmentations.augmentations_kornia):
                X = self.perturbation(transforms.ToTensor()(X), 3)
            elif isinstance(X, torch.Tensor) and (self.perturbation in augmentations.augmentations_pil):
                X = self.perturbation(transforms.functional.to_pil_image(X), 3)
            else:
                X = self.perturbation(X, 3)
            if isinstance(X, torch.Tensor):
                pass
            else:
                X = transforms.ToTensor()(X)
        return X,y   

  def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = os.path.join(
            self.data_dir,
            self.input_array[idx])
        x = Image.open(img_filename).convert('RGB')
        return x

# len(metadata_df[metadata_df["node"]==0])
# len(metadata_df[metadata_df["node"]==1])
# len(metadata_df[metadata_df["node"]==2])
# len(metadata_df[metadata_df["node"]==3])
# len(metadata_df[metadata_df["node"]==4])

def get_class_distribution(node):
    len_data = len(metadata_df[metadata_df["node"]==node])
    empirical = metadata_df[metadata_df["node"]==node]["tumor"]
    class_0 = (empirical==0).sum()/len_data
    class_1 = (empirical==1).sum()/len_data
    return np.array([class_0, class_1])

node_0_pmf = get_class_distribution(0)
node_1_pmf = get_class_distribution(1)
node_2_pmf = get_class_distribution(2)
node_3_pmf = get_class_distribution(3)
node_4_pmf = get_class_distribution(4)

def ugly_kl(p,q):
    return p[0]*np.log(p[0]/q[0]) + p[1]*np.log(p[1]/q[1])

kl_node_0 = ugly_kl(node_0_pmf, node_1_pmf), ugly_kl(node_0_pmf, node_2_pmf), ugly_kl(node_0_pmf, node_3_pmf), ugly_kl(node_0_pmf, node_4_pmf)
kl_node_1 = ugly_kl(node_1_pmf, node_0_pmf), ugly_kl(node_1_pmf, node_2_pmf), ugly_kl(node_1_pmf, node_3_pmf), ugly_kl(node_1_pmf, node_4_pmf)
kl_node_2 = ugly_kl(node_2_pmf, node_0_pmf), ugly_kl(node_2_pmf, node_1_pmf), ugly_kl(node_2_pmf, node_3_pmf), ugly_kl(node_2_pmf, node_4_pmf)
kl_node_3 = ugly_kl(node_3_pmf, node_0_pmf), ugly_kl(node_3_pmf, node_1_pmf), ugly_kl(node_3_pmf, node_2_pmf), ugly_kl(node_3_pmf, node_4_pmf)
kl_node_4 = ugly_kl(node_4_pmf, node_0_pmf), ugly_kl(node_4_pmf, node_1_pmf), ugly_kl(node_4_pmf, node_2_pmf), ugly_kl(node_4_pmf, node_3_pmf)



def get_target_nodes(current_node):
    remaining_nodes = [0,1,2,3,4]
    remaining_nodes.remove(current_node)
    return remaining_nodes
    
    
def test_augmentations(net, X_test, y_test, device):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = {}
  for corruption in augmentations.augmentations_all:
    corruption_name = corruption.__name__
    # Reference to original data is mutated
    dataset = Camelyon17Dataset(camelyon17_v1, y_test, X_test, corruption)
    loader = DataLoader(dataset, batch_size=32)

    test_loss, test_acc = test(net, loader, device)
    corruption_accs[corruption_name] = test_acc
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption_name, test_loss, 100 - 100. * test_acc))
  return corruption_accs


def test_nodes(net, current_node, device):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = {}
  remaining_nodes = get_target_nodes(current_node)
  for node in remaining_nodes:
    node_name = "node_" + str(node)
    # Reference to original data is mutated
    data, label = get_hospital_data(metadata_df, node)
    _, X_test, _, y_test = train_test_split(data, label, test_size=0.16, random_state=0)
    dataset = Camelyon17Dataset(camelyon17_v1, y_test, X_test)
    loader = DataLoader(dataset, batch_size=32)

    test_loss, test_acc = test(net, loader, device)
    corruption_accs[node_name] = test_acc
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        node_name, test_loss, 100 - 100. * test_acc))
  return corruption_accs
 

    
def compute_gde_camelyon_augmentations(model_a, model_b, X_test, y_test, device):
  gde = {}
  for corruption in augmentations.augmentations_all:
    corruption_name = corruption.__name__
    # Reference to original data is mutated
    dataset = Camelyon17Dataset(camelyon17_v1, y_test, X_test, corruption)
    loader = DataLoader(dataset, batch_size=32)

    gde[corruption_name] = compute_gde_loader(model_a, model_b, loader, device)
    print('{} GDE: {}'.format(corruption_name, gde[corruption_name]))
  return gde 

def compute_gde_camelyon_nodes(model_a, model_b, current_node, device):
  gde = {}
  remaining_nodes = get_target_nodes(current_node)
  for node in remaining_nodes:
    node_name = "node_" + str(node)
    # Reference to original data is mutated
    data, label = get_hospital_data(metadata_df, node)
    _, X_test, _, y_test = train_test_split(data, label, test_size=0.16, random_state=0)
    dataset = Camelyon17Dataset(camelyon17_v1, y_test, X_test)
    loader = DataLoader(dataset, batch_size=32)

    gde[node_name] = compute_gde_loader(model_a, model_b, loader, device)
    print('{} GDE: {}'.format(node_name, gde[node_name]))
  return gde 

def rebalance_data(target_data, target_label, source_distribution, proportion):
    """
    This function will take a list of target data and a list of target labels, and subsample only a given proportion
    of it according to the source distribution.

    Parameters
    ----------
    target_data : List
        A list of filenames related to images
    target_label : List
        A list of labels from the target distribution
    source_distribution : Tuple
        A binary distribution (p0,p1) with p0+p1 = 1, 0<= p0,p1 <= 1
    proportion : Float
        Between 0 and 1.

    Returns
    -------
    X : List
        New list of filenemaes subsampled accordingly to the source distribution.
    y : TYPE
        New list of labels subsampled accordingly to the source distribution.

    """
    p0, p1 = source_distribution
    len_target = len(target_data) * proportion
    len_p0_target = int(p0 * len_target)
    len_p1_target = int(p1 * len_target)
    index_p0_target = np.flatnonzero(target_label == 0)
    index_p1_target = np.flatnonzero(target_label == 1)
    # import pdb ; pdb.set_trace()
    index_p0_target_rebalanced = np.random.choice(index_p0_target, len_p0_target)
    index_p1_target_rebalanced = np.random.choice(index_p1_target, len_p1_target)
    X0, y0 = target_data[index_p0_target_rebalanced], target_label[index_p0_target_rebalanced]
    X1, y1 = target_data[index_p1_target_rebalanced], target_label[index_p1_target_rebalanced]
    X = np.concatenate((X0, X1), 0)
    y = np.concatenate((y0, y1), 0)
    return X, y 