# Deployment Error Estimation

## Data
You are required to have this ood_uncertainty_prediction repository in a folder containing a data folder for all the dataset considered. 
As the time being, the code works only for mnist_c. 

```
├── project
│   ├── data
│   │   ├── mnist_c
│   │   ├── cifar_c
│   │   ├── imagenet
│   │   ├── Camelyon17
│   ├── ood_uncertainty_prediction

```


## Command line interface
Now set your working directory in the *ood_uncertainty_prediction*.  
You can get the results by executing the following commnand to get result from MNIST, CIFAR and IMAGENET datasets:
```
python mnist_launch.py --algorithm --device
```
After the keyword `algorithm` you can add one of the following three algorithms:
- `ATC`: Garg, Saurabh, et al. "Leveraging unlabeled data to predict out-of-distribution performance." arXiv preprint arXiv:2201.04234 (2022).
- `H-distance`: Ben-David, Shai, et al. "Analysis of representations for domain adaptation." Advances in neural information processing systems 19 (2006).
- `GDE`: Jiang, Yiding, et al. "Assessing generalization of SGD via disagreement." arXiv preprint arXiv:2106.13799 (2021).
- `OTDD`: Alvarez-Melis, David, and Nicolo Fusi. "Geometric dataset distances via optimal transport." Advances in Neural Information Processing Systems 33 (2020): 21428-21439.

Device count only if you know the name of your GPU, and if you have many of them

In case you are using the Camelyon17 dataset, it will look more like:
```
python camelyon17_launch.py --device --training_node --rebalanced --all_severity
```
Here we have only implemented the algorithm GDE. 
- `training_node` which corresponds to the hospital taken as a training set. 
- `rebalanced` ask whether or not to rebalance the deployment nodes according to the distribution of the training node.
- `all_severity` ask whether to run on every severity of the kornia augmentations (selected because they have at least five levels), or just on a severity of 3.




