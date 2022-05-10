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
from pathlib import Path

# Look at https://machinelearningmastery.com/robust-regression-for-machine-learning-in-python/
# for a robust regression, and an approach to compare all methods.
def normalize(data):
    return (data - data.min())/ (data.max() - data.min())

def plot_distance_to_acc(distance, accuracies, plot_name):
    gap = np.abs(accuracies-accuracies[0])
    sns.regplot(x=normalize(distance), y=gap)
    plt.ylim([0,1])
    plt.ylabel("| acc(T)-acc(S) |")
    plt.xlabel("Dataset distance")
    plt.title("{}".format(plot_name))
    # plt.savefig("{}".format(plot_name), dpi=200)
    plt.show()

def plot_atc_differences(estimation, accuracies, plot_name):
    """Because estimations were designed to approximate accuracies in percent between 0 and 100
    we need to"""
    gap = np.abs(accuracies-accuracies[0])
    estimation_gap = np.abs(accuracies-estimation) # divide estimation per 100 if in the hundreds scale
    sns.regplot(x=estimation_gap, y=gap)
    plt.ylim([0,1])
    plt.ylabel("| acc(T)-acc(S) |")
    plt.xlabel("ATC")
    plt.title("{}".format(plot_name))
    # plt.savefig("{}".format(plot_name), dpi=200)
    plt.show()
    
    
mnist_accuracies = np.load("mnist_accuracies.npy")
mnist_h_distances = np.load("mnist_h_distances.npy")
mnist_labelwise_h_distances = np.load("mnist_labelwise_h_distances.npy")
mnist_otdd = np.load("mnist_otdd.npy")
mnist_atc = np.load("mnist_atc.npy")

cifar10_accuracies = np.load("cifar10_accuracies.npy")
cifar10_h_distances = np.load("cifar10_h_distances.npy")
cifar10_labelwise_h_distances = np.load("cifar10_labelwise_h_distances.npy")
cifar10_otdd = np.load("cifar10_otdd.npy")
cifar10_atc = np.load("cifar10_atc.npy")



plot_distance_to_acc(mnist_h_distances, mnist_accuracies, "mnist_h_distance_plot")
plot_distance_to_acc(-mnist_labelwise_h_distances, mnist_accuracies, "mnist_labelwise_h_distance_plot")
plot_distance_to_acc(mnist_otdd, mnist_accuracies, "mnist_otdd")
plot_atc_differences(mnist_atc, mnist_accuracies, "mnist_atc_plot")

plot_distance_to_acc(cifar10_h_distances, cifar10_accuracies, "cifar10_h_distances")
plot_distance_to_acc(cifar10_labelwise_h_distances, cifar10_accuracies, "cifar10_labelwise_h_distances")
plot_distance_to_acc(cifar10_otdd, cifar10_accuracies, "cifar10_otdd")
plot_atc_differences(cifar10_atc, cifar10_accuracies, "cifar10_atc")


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression, HuberRegressor
# evaluate a model
def evaluate_model(X, y, model):
	# define model evaluation method
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	# force scores to be positive
	return np.absolute(scores)

# plot the dataset and the model's line of best fit
def plot_best_fit(X, y, model):
	# fut the model on all data
	model.fit(X, y)
	# plot the dataset
	plt.scatter(X, y)
	# plot the line of best fit
	xaxis = np.arange(X.min(), X.max(), 0.01)
	yaxis = model.predict(xaxis.reshape((len(xaxis), 1)))
	plt.plot(xaxis, yaxis, color='r')
	# show the plot
	plt.title(type(model).__name__)
	plt.show()

model = LinearRegression()
model = HuberRegressor()
# evaluate model
h_distance_prediction = evaluate_model(cifar10_h_distances[:,np.newaxis], cifar10_accuracies, model)
labelwise_h_distance_prediction = evaluate_model(-cifar10_labelwise_h_distances[:,np.newaxis], cifar10_accuracies, model)
otdd_prediction = evaluate_model(cifar10_otdd[:,np.newaxis], cifar10_accuracies, model)
atc_prediction = evaluate_model(cifar10_atc[:,np.newaxis], cifar10_accuracies, model)


print('H-distance Prediction - Mean MAE: %.3f (%.3f)' % (np.mean(h_distance_prediction), np.std(h_distance_prediction)))
# plot the line of best fit
plot_best_fit(cifar10_h_distances[:,np.newaxis], cifar10_accuracies, model)

print('Labelwise H-distance Prediction - Mean MAE: %.3f (%.3f)' % (np.mean(labelwise_h_distance_prediction), np.std(labelwise_h_distance_prediction)))
# plot the line of best fit
plot_best_fit(cifar10_labelwise_h_distances[:,np.newaxis], cifar10_accuracies, model)

print('OTDD Prediction - Mean MAE: %.3f (%.3f)' % (np.mean(otdd_prediction), np.std(otdd_prediction)))
# plot the line of best fit
plot_best_fit(cifar10_otdd[:,np.newaxis], cifar10_accuracies, model)

print('ATC Prediction - Mean MAE: %.3f (%.3f)' % (np.mean(atc_prediction), np.std(atc_prediction)))
# plot the line of best fit
plot_best_fit(cifar10_atc[:,np.newaxis], cifar10_accuracies, model)

results = dict()
results["H-distance"] = h_distance_prediction
results["Labelwise H-distance"] = labelwise_h_distance_prediction
results["OTDD"] = otdd_prediction
results["ATC"] = atc_prediction


# plot model performance for comparison
plt.boxplot(results.values(), labels=results.keys(), showmeans=True)
plt.show()
    
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


