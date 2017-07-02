# mlp_functions module
# assumes all input features preceed output(s)
import random
import math


def initialise_bias(params):
    outputs_l_j = []

    # plus one for input layer
    for l in range(len(params['hidden_nodes']) + 1):
        outputs_l_j.append([1.0])

    # set 0 output on first output neuron layer to None as this won't have bias
    outputs_l_j.append([None])

    return outputs_l_j


def initialise_weights(params, neurons_l):
    weights_l_i_j = []

    if params['fixed_weight_seed'] is not None:
        rng = random.Random(params['fixed_weight_seed'])
    else:
        rng = random.Random()

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
                        weight = (params['weight_init_mean'] +
                                  params['weight_init_range'] * (
                                    (2 * rng.random()) - 1))
                        weights_j.append(weight)

                weights_i_j.append(weights_j)

        weights_l_i_j.append(weights_i_j)
    return weights_l_i_j


def forward_pass(params, neurons_l, weights_l_i_j, outputs_l_j):
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
                        params['hidden_layers_function'])
                    outputs_l_j[l].append(firing_function(activation))
                # set outputs in output layer
                elif (l == L):
                    # check output layer function
                    if (params['output_function'] == 'linear'):
                        # linear output
                        outputs_l_j[l].append(activation)
                    else:
                        # squashed output
                        firing_function = get_firing_function(
                            params['output_function'])
                        outputs_l_j[l].append(firing_function(activation))
    return outputs_l_j


def get_firing_function(function_name):
    if function_name == 'logistic':
        def firing_function(activation):
            return (1 / (1 + math.exp(-activation)))
    elif function_name == 'logistic_derivative':
        def firing_function(activation):
            return ((math.exp(-activation)) /
                    (math.pow((1 + math.exp(-activation)), 2)))
    elif function_name == 'tanh':
        def firing_function(activation):
            return (math.tanh(activation))
    elif function_name == 'tanh_derivative':
        def firing_function(activation):
            return (1 - math.pow(math.tanh(activation), 2))
    elif function_name == 'lecun_tanh':
        def firing_function(activation):
            return (1.7159 * math.tanh((2/3) * activation))
    elif function_name == 'lecun_tanh_derivative':
        def firing_function(activation):
            return (1.14393 / math.pow(math.cosh((2/3) * activation), 2))

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


def calculate_errors(params, neurons_l, weights_l_i_j, teacher_i, outputs_l_j):
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
                    if (params['output_function'] == 'linear'):
                        # linear output
                        reversed_errors_i.append(difference)
                    else:
                        # squashed output
                        firing_function = get_firing_function(
                            (params['output_function'] + '_derivative'))
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
                       params['hidden_layers_function'] + '_derivative')
                    reversed_errors_i.append(error_sum *
                                             firing_function(activation))
        reversed_errors_l_i.append(reversed_errors_i)

    return reversed_errors_l_i[::-1]


def update_weights(params, neurons_l, weights_l_i_j, errors_l_i, outputs_l_j):
        for l, neuron_l in enumerate(neurons_l):
            # skip first layer as it has no neurons
            if (l != 0):
                for i in range(neuron_l):
                    # neuron count starts at 1
                    i += 1
                    # plus one for bias
                    for j in range(neurons_l[l - 1] + 1):
                        weights_l_i_j[l][i][j] += (
                            params['training_rate'] *
                            errors_l_i[l][i] *
                            outputs_l_j[l - 1][j])
        return weights_l_i_j
