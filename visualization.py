# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:03:02 2022

@author: JeanMichelAmath
"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import os

def normalize(data):
    return (data - data.min())/ (data.max() - data.min())

def plot_distance_to_acc(distance, accuracies, plot_name):
    gap = np.abs(accuracies-accuracies[0])
    sns.regplot(x=normalize(distance), y=gap)
    plt.ylim([0,1])
    plt.ylabel("| acc(T)-acc(S) |")
    plt.xlabel("Dataset distance")
    plt.title("Loss of performance as dataset shifts")
    plt.savefig("{}".format(plot_name), dpi=200)
    plt.show()

def plot_atc_differences(estimation, accuracies, plot_name):
    """Because estimations were designed to approximate accuracies in percent between 0 and 100
    we need to"""
    gap = np.abs(accuracies-accuracies[0])
    estimation_gap = np.abs(accuracies-estimation) # divide estimation per 100 if in the hundreds scale
    sns.regplot(x=normalize(estimation_gap), y=gap)
    plt.ylim([0,1])
    plt.ylabel("| acc(T)-acc(S) |")
    plt.xlabel("ATC")
    plt.title("Loss of performance as dataset shifts")
    plt.savefig("{}".format(plot_name), dpi=200)
    plt.show()
    
    
accuracies = np.load("mnist_accuracies.npy")
h_distances = np.load("mnist_h_distances.npy")
labelwise_h_distances = np.load("mnist_labelwise_h_distances.npy")
atc = np.load("mnist_atc.npy")

plot_distance_to_acc(h_distances, accuracies, "mnist_h_distance_plot")
plot_distance_to_acc(-labelwise_h_distances, accuracies, "mnist_labelwise_h_distance_plot")
plot_atc_differences(atc, accuracies, "mnist_atc_plot")
    
######################## VISUALIZE DIVERGENCE MATRICES ########################

def plot_divergence_matrix(divergence_matrices, distance , accuracy ,corruptions):
    fig = plt.figure(figsize=(32., 32.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(8, 4),  # creates 4x4 grid of axes
                     axes_pad=0.25,  # pad between axes in inch.
                     cbar_mode="edge",
                     cbar_pad=0.5,
                     share_all=True,
                     cbar_location="right",
                     cbar_set_cax=True)
    i=0
    for ax, im in zip(grid, divergence_matrices):
        # Iterating over the grid returns the Axes.
        img = ax.imshow(im)  
        ax.set_title("Id vs {} - Acc: {} - Div: {}".format(corruptions[i], accuracy[i], np.around(distance[i],1)))
        # import pdb; pdb.set_trace()
        # if i % 4:
        # ax.cbar_axes[i].colorbar(img)
        i = i+ 1
    # plt.savefig("div_matrices_plot", dpi=300)
    plt.show()


