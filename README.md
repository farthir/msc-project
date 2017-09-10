# UCL MSc Computer Science Summer Project
Building neural networks for materials science!

I wrote this code for my MSc project to predict the tensile behaviour of cellular solids but it's generic enough to be used for other purposes.

It implements a feedforward, multilayer perceptron (MLP) neural network with gradient descent backpropagation learning. It uses stochastic learning and does not (yet) include anything 'fancy' like mini-batching, momentum, dropout, regularisation etc. However, it does include data standardisation which can be used on inputs or outputs. It is built from the ground up and does not use existing neural network libraries.

## Prerequisites
Requires Python 3 and the following packages:
* matplotlib
* progressbar
* numpy
* pandas

## Usage
    python main.py [input_prefix] [output_prefix] [num_networks] [enable_summary]

* `input_prefix` string that corresponds to the prefix on all input files. The required input files with naming conventions are detailed below.
* `output_prefix` string that corresponds to the prefix added to all output files.
* `num_networks` integer that specifies the number of neural network iterations to run.
* `enable_summary` boolean that enables or disables the summary error graphs presented at the end of all runs. Also saved to an svg file.

### Input Naming Conventions
Input file naming convention and directory placement relative to repository root is as follows:
* `parameters/[input_prefix].json` for neural network parameters file. Usage detailed below.
* `data/[input_prefix].csv` for training data (also used for validation if enabled).
* `data/[input_prefix]_test.csv` for test data.
* `data/[input_prefix]_structure.csv` describes the structure of training and test data files. Usage detailed below.

### Structure CSV File
The structure CSV file describes how each column of a dataset should be handled by the system. It works across both input and output columns in the file. The options that can be used are:
* `none` do not modify the column,
* `numeric` standardise the column,
* `binary` convert binary outputs {0, 1} to {-1, 1} for use with any symmetric firing function.


Example:
```
numeric,numeric,binary
```

### Parameters JSON File
Below is an example parameters JSON file with value descriptions (# is not a valid JSON comment so these must be removed prior to usage).

    {
        "training_validation_ratio": 0.75, # how much data to use for validation
        "validating": true, # whether to enable validation
        "testing": true, # whether to enable testing
        "best_weights": true, # whether to use the best weights for testing (requires validation)
        "input_dimensions": 3, # number of input dimensions/features
        "output_dimensions": 1, # number of output dimensions
        "weight_init_mean": 0, # mean value for random weight initialisation
        "weight_init_range": 0.5, # spread of random weight initialisation about mean
        "random_numbers_seed": null, # sets a specific random number generator seed for both weight initialisation and training pattern randomisation. null for disabled. (debugging only)
        "hidden_layers_function": "lecun_tanh", # activation function to use in hidden layers (logistic, tanh, lecun_tanh)
        "output_function": "linear", # activation function to use in output layer (logistic, tanh, lecun_tanh, linear)
        "hidden_nodes": [10,10], # architecture of hidden layers. [] to disable hidden layers
        "training_rate": 0.05, # neural network training rate
        "target_training_error": 0.01, # target neural network training error
        "max_epochs": 100, # maximum number of epochs
        "stop_at_target_training_error": true, # whether to stop at target neural network training error, otherwise training will stop at max epochs
        "save_summary": true, # save summary of neural network results
        "save_detailed": true, # save detailed progression of training and validation errors for each epoch
        "save_testing": true, # save results of testing
        "save_network": false, # save neuron weights and delta values at at neural network resolution specified (debugging only)
        "save_network_resolution": 1 # how frequently, in epochs, to save the neuron weights and delta values (debugging only)
    }