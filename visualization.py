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

def plot_distance_to_acc(accuracies, distance, plot_name):
    gap = np.abs(accuracies-accuracies[0])
    sns.regplot(x=normalize(distance), y=gap)
    plt.ylim([0,1])
    plt.ylabel("| acc(T)-acc(S) |")
    plt.xlabel("Dataset distance")
    plt.title("{}".format(plot_name))
    # plt.savefig("{}".format(plot_name), dpi=200)
    plt.show()

def plot_atc_differences(accuracies, estimation, plot_name):
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
    

def load_results(dataset, names):
    curr_path = os.getcwd()
    data_path = os.path.join(curr_path, dataset)
    results = {}
    accuracies = np.load(os.path.join(data_path, dataset + "_" + "accuracies" + ".npy"))
    for name in names:
        name_path = os.path.join(data_path, dataset + "_" + name + ".npy")
        results[name] = np.load(name_path)
    return results, accuracies


cifar10_results, cifar10_accuracies = load_results("cifar10", ["h_dist_pre", "atc", "otdd"])
mnist_results, mnist_accuracies =  load_results("mnist", ["h_dist_pre", "atc", "otdd"])
imagenet_results, imagenet_accuracies =  load_results("imagenet", ["h_dist_pre", "atc", "otdd"])



def plot_all_results(accuracies, data):
    for key in data.keys():
        plot_distance_to_acc(accuracies, data[key], key)

plot_all_results(cifar10_accuracies, cifar10_results)     
plot_all_results(mnist_accuracies, mnist_results)    
plot_all_results(imagenet_accuracies, imagenet_results)       





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
def plot_best_fit(X, y, model, name):
	# fut the model on all data
	model.fit(X, y)
	# plot the dataset
	plt.scatter(X, y)
	# plot the line of best fit
	xaxis = np.arange(X.min(), X.max(), 0.01)
	yaxis = model.predict(xaxis.reshape((len(xaxis), 1)))
	plt.plot(xaxis, yaxis, color='r')
	# show the plot
	plt.title(name)
	plt.show()

model = LinearRegression()
model = HuberRegressor()
# evaluate model

def get_prediction(results, accuracy, model):
    predictions = {}
    for key in results.keys():
        predictions[key] = evaluate_model(results[key][:,np.newaxis], accuracy, model)
    return predictions

mnist_prediction = get_prediction(mnist_results, mnist_accuracies, model)
cifar10_prediction = get_prediction(cifar10_results, cifar10_accuracies, model)
imagenet_prediction = get_prediction(imagenet_results, imagenet_accuracies, model)


def display_results(predictions, accuracy, results, model):
    for key in results.keys():
        mean = np.mean(predictions[key])
        std = np.std(predictions[key])
        print('{} Prediction - Mean MAE: {} +- {}'.format(key, np.round(mean,2), np.round(std,2)))
        # plot the line of best fit
        plot_best_fit(results[key][:,np.newaxis], accuracy, model, key)

display_results(mnist_prediction, mnist_accuracies, mnist_results, model)
display_results(cifar10_prediction, cifar10_accuracies, cifar10_results, model)
display_results(imagenet_prediction, imagenet_accuracies, imagenet_results, model)


plt.boxplot(mnist_prediction.values(), labels=mnist_prediction.keys(), showmeans=True)
plt.ylabel("Mean Absolute Error")
plt.title("MNIST")
plt.savefig("mnist_predictions", dpi=300)

plt.boxplot(cifar10_prediction.values(), labels=cifar10_prediction.keys(), showmeans=True)
plt.ylabel("Mean Absolute Error")
plt.title("CIFAR-10")
plt.savefig("cifar10_predictions", dpi=300)

plt.boxplot(imagenet_prediction.values(), labels=imagenet_prediction.keys(), showmeans=True)
plt.ylabel("Mean Absolute Error")
plt.title("ImageNet")
plt.savefig("imagenet_predictions", dpi=300)

## Subexperiment, determine whether or not to pretrained on H-distance
# previously I forgot to specify to the model to have 2 class output, so it used to have 10 class. I checked 2 things
# whether or not changing the number of output class have an effect, whether or not pretraining have an effect
# it seems that pretraining works better, but is isn't as fast. Why ?
mnist_h_10class = np.load("mnist_h_distances_10classes.npy")
mnist_h_2class = np.load("mnist_h_distances.npy")
mnist_h_pretrained = np.load("mnist_h_distances_pretrained.npy")

plt.plot(mnist_h_10class/100, label="10 class")
plt.plot(mnist_h_2class, label="2 class")
plt.plot(mnist_h_pretrained, label="pretrained")
plt.legend()

plot_distance_to_acc(mnist_h_10class/100, mnist_accuracies, "10 class")
plot_distance_to_acc(mnist_h_2class, mnist_accuracies, "2 class")
plot_distance_to_acc(mnist_h_pretrained, mnist_accuracies, "pretrained")

model = LinearRegression()
h_distance_10class = evaluate_model(mnist_h_10class[:,np.newaxis], mnist_accuracies, model)
h_distance_2class= evaluate_model(mnist_h_2class[:,np.newaxis], mnist_accuracies, model)
h_distance_pretrained = evaluate_model(mnist_h_pretrained[:,np.newaxis], mnist_accuracies, model)

print('H-distance 10 class - Mean MAE: %.3f (%.3f)' % (np.mean(h_distance_10class), np.std(h_distance_10class)))
plot_best_fit(mnist_h_10class[:,np.newaxis], mnist_accuracies, model)
print('H-distance 2 class - Mean MAE: %.3f (%.3f)' % (np.mean(h_distance_2class), np.std(h_distance_2class)))
plot_best_fit(mnist_h_2class[:,np.newaxis], mnist_accuracies, model)
print('H-distance pretrained- Mean MAE: %.3f (%.3f)' % (np.mean(h_distance_pretrained), np.std(h_distance_pretrained)))
plot_best_fit(mnist_h_pretrained[:,np.newaxis], mnist_accuracies, model)

    
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


