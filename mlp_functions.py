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
    L = len(neurons_l) - 1
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
                if (l < L):
                    # apply squashing to hidden layers
                    firing_function = get_firing_function(
                        config.params['hidden_layers_function'])
                    outputs_l_j[l].append(firing_function(activation))
                # set outputs in output layer
                elif (l == L):
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


def update_ms_error(neurons_l, error, teacher_i, outputs_l_j):
    # get number of output layer neurons
    L = len(neurons_l) - 1
    output_neurons = neurons_l[L]
    for i in range(output_neurons):
        # neuron count starts at 1
        i += 1
        error += math.pow(teacher_i[i] - outputs_l_j[L][i], 2)
    return error


def calculate_errors(neurons_l, weights_l_i_j, teacher_i, outputs_l_j):
    L = len(neurons_l) - 1
    reversed_errors_l_i = []
    for l, neuron_l in reversed(list(enumerate(neurons_l))):
        # skip first layer as it's the input layer
        reversed_errors_i = []
        # indexing starts at 1 for neurons
        reversed_errors_i.append(None)
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
                # output layer with known targets
                if (l == L):
                    # check output layer function
                    difference = 0.0
                    difference = teacher_i[i] - outputs_l_j[l][i]
                    if (config.params['output_function'] == 'linear'):
                        # linear output
                        reversed_errors_i.append(difference)
                    else:
                        # squashed output
                        firing_function = get_firing_function(
                            (config.params['output_function'] + '_derivative'))
                        reversed_errors_i.append(difference *
                                                 firing_function(activation))
                # hidden layers
                else:
                    # calculate sum
                    error_sum = 0.0
                    # start at 1
                    for k in range(neurons_l[l + 1]):
                        # neuron count starts at 1
                        k += 1
                        # reversed_errors_i is backward
                        error_sum += (reversed_errors_l_i[L - (l + 1)][k] *
                                      weights_l_i_j[l + 1][k][i])
                    # hidden layers squashing
                    firing_function = get_firing_function(
                       config.params['hidden_layers_function'] + '_derivative')
                    reversed_errors_i.append(error_sum *
                                             firing_function(activation))
        reversed_errors_l_i.append(reversed_errors_i)

    return reversed_errors_l_i[::-1]


def update_weights(neurons_l, weights_l_i_j, errors_l_i, outputs_l_j):
        for l, neuron_l in enumerate(neurons_l):
            # skip first layer as it has no neurons
            if (l != 0):
                for i in range(neuron_l):
                    # neuron count starts at 1
                    i += 1
                    # plus one for bias
                    for j in range(neurons_l[l - 1] + 1):
                        weights_l_i_j[l][i][j] += (
                            config.params['training_rate'] *
                            errors_l_i[l][i] *
                            outputs_l_j[l - 1][j])
        return weights_l_i_j
