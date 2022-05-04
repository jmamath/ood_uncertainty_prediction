# OOD Uncertainty Prediction

## Data
You are required to have this ood_uncertainty_prediction repository in a folder containing a data folder for all the dataset considered. 
As the time being, the code works only for mnist_c. 

```
├── project
│   ├── data
│   │   ├── mnist_c
│   ├── ood_uncertainty_prediction

```

## Command line interface
Now set your working directory in the *ood_uncertainty_prediction*.  
You can get the results by executing the following commnand:
```
python mnist_launch.py --algorithm
```
After the keyword `algorithm` you can add one of the following three algorithms:
- `ATC`
- `H-distance`
- `Labelwise-H-distance`

### Training 
The first task is to train the model (here a LeNet-5 architecture) on the identity dataset, and to save the model.
Then to compute the accuracy on the test set and every other corrupted dataset in mnist_c.

### Distance from identity
Then depending on the algorithm you choose, a distance will be computed from the identity dataset to every other dataset in the *mnist_c* folder. 
These distances are stored in the *ood_uncertainty_prediction/mnist* folder and respectively named:
- `mnist_atc.npy`
- `mnist_h_distances.npy`
- `mnist_divergence_matrices.npy` and `mnist_labelwise_h_distances.npy`

## TODO
### Add datasets
- Cifar10
- ImageNet
- Amazon-Wilds

### Add methods
- Optimal Transport Dataset Distance

### Documentation
- Add documentation on visualization
