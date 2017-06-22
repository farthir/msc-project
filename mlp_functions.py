# mlp_functions module
# assumes all input features preceed output(s)
import csv
import random
import config
import math


def read_patterns(data_filename):

    # num_output_neurons = num_output_dimensions -- no! -- depends on
    # whether classifier or regression and then if classifier, the encoding

    with open(data_filename, 'r', encoding='utf-8-sig') as data_file:
        reader = csv.reader(data_file)
        patterns = []
        for row in reader:
            patterns.append(row)

        # randomise csv input to make sure training/validation sets are truely
        # random
        random.shuffle(patterns)

    return patterns


def initialise_bias(neuron_layers):
    neuron_outputs_l_j = []

    for l in range(neuron_layers):
        neuron_outputs_l_j.append([1.0])

    # set 0 output on output neuron layer to None as this won't have bias
    neuron_outputs_l_j.append([None])

    return neuron_outputs_l_j


def initialise_weights(neurons_l):
    weights_l_i_j = []

    for l, neuron_l in enumerate(neurons_l):
        weights_i_j = []
        for i in range(neuron_l):
            weights_j = []
            if l == 0:
                weights_j.append(None)
            else:
                for j in range(neurons_l[l - 1]):
                    weight = random.random()
                    weights_j.append(weight)
            weights_i_j.append(weights_j)
        weights_l_i_j.append(weights_i_j)

    return weights_l_i_j


'''
def forward_pass(neurons_l):
    for l, neuron_l in enumerate(neurons_l):
        for i in range(1, neuron_l):
'''


def get_firing_function(function_name):
    if function_name == 'sigmoid':
        def firing_function(activation):
            return (1 / (1 + math.exp(-activation)))
    elif function_name == 'sigmoid_derivative':
        def firing_function(activation):
            return ((math.exp(-activation)) /
                    (math.pow((1 + math.exp(-activation)), 2)))
    elif function_name == 'tanh':
        def firing_function(activation):
            return ((math.exp(activation) - math.exp(-activation)) /
                    (math.exp(activation) + math.exp(-activation)))
    elif function_name == 'tanh_derivative':
        def firing_function(activation):
            return (1 - math.pow(((math.exp(activation)
                                 - math.exp(-activation)) /
                    (math.exp(activation) + math.exp(-activation))), 2))

    return firing_function
