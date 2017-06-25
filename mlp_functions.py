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
        reader = csv.reader(data_file, quoting=csv.QUOTE_NONNUMERIC)
        patterns = []
        for row in reader:
            patterns.append(row)

        # randomise csv input to make sure training/validation sets are truely
        # random
        random.shuffle(patterns)

    return patterns


def initialise_bias(neuron_layers):
    outputs_l_j = []

    for l in range(neuron_layers):
        outputs_l_j.append([1.0])

    # set 0 output on first output neuron layer to None as this won't have bias
    outputs_l_j.append([None])

    return outputs_l_j


def initialise_weights(neurons_l):
    weights_l_i_j = []

    for l, neuron_l in enumerate(neurons_l):
        weights_i_j = []
        # no weights in input layer
        if l == 0:
            weights_i_j.append(None)
        else:
            # plus one for bias
            for i in range(neuron_l + 1):
                weights_j = []
                # no weights on bias i = 0
                if i == 0:
                    weights_j.append(None)
                else:
                    # add one to accomodate bias
                    for j in range(neurons_l[l - 1] + 1):
                        weight = random.random()
                        weights_j.append(weight)

                weights_i_j.append(weights_j)

        weights_l_i_j.append(weights_i_j)

    return weights_l_i_j


def forward_pass(neurons_l, weights_l_i_j, outputs_l_j):
    for l, neuron_l in enumerate(neurons_l):
        # skip first layer as it's already populated and has no neurons
        if (l != 0):
            for i in range(neuron_l):
                # neuron count starts at 1
                i += 1
                # calculate activation
                activation = 0.0
                # plus one for bias
                for j in range(neurons_l[l - 1] + 1):
                    activation += (weights_l_i_j[l][i][j] *
                                   outputs_l_j[l-1][j])
                # set outputs in hidden layers
                if (l < (len(neurons_l) - 1)):
                    # apply squashing to hidden layers
                    firing_function = get_firing_function(
                        config.params['hidden_layers_function'])
                    outputs_l_j[l].append(firing_function(activation))
                # set outputs in output layer
                elif (l == (len(neurons_l) - 1)):
                    # check output layer function
                    if (config.params['output_function'] == 'linear'):
                        # linear output
                        outputs_l_j[l].append(activation)
                    else:
                        # squashed output
                        firing_function = get_firing_function(
                            config.params['output_function'])
                        outputs_l_j[l].append(firing_function(activation))
    return outputs_l_j


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
