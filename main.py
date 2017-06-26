# driver module for programme
import sys
import mlp_functions
import config
import math
import matplotlib.pyplot as plt

filename = sys.argv[1]

# read in patterns
patterns = mlp_functions.read_patterns('data/%s.csv' % filename)

# read in test patterns
if (config.params['testing']):
    test_patterns = mlp_functions.read_patterns('data/%s_test.csv' % filename)

if (config.params['validating'] and
        config.params['training_validation_ratio'] == 1):
    training_patterns = patterns
    validation_patterns = patterns

elif (config.params['validating'] and
        config.params['training_validation_ratio'] < 1):
    num_training_patterns = int(len(patterns) *
                                config.params['training_validation_ratio'])
    num_validation_patterns = len(patterns) - num_training_patterns
    training_patterns = patterns[:num_training_patterns]
    validation_patterns = patterns[num_training_patterns:]

elif not config.params['validating']:
    training_patterns = patterns

# network initialisation
# set number of neurons in each layer
# layer '0' is the input layer and defines number of inputs
neurons_l = []
neurons_l.append(config.params['input_dimensions'])

if config.params['neuron_layers'] > 1:
    for l in range(config.params['neuron_layers'] - 1):
        neurons_l.append(config.params['hidden_nodes'][l])

# this may not always be the case but is set here
neurons_l.append(config.params['output_dimensions'])

# initialise weights
weights_l_i_j = mlp_functions.initialise_weights(neurons_l)

# backpropagation loop
epoch = 0
training_errors = []
repeat = True
if config.params['validating']:
    validation_error_best = 1000.0
    validation_errors = []

while (repeat):
    training_error = 0.0

    for p in training_patterns:
        # load pattern
        input_pattern = p[:config.params['input_dimensions']]
        # set bias 'output'
        outputs_l_j = mlp_functions.initialise_bias(
                        config.params['neuron_layers'])
        # add input pattern to 'output' of layer 0 (i.e. set the input to p)
        outputs_l_j[0].extend(input_pattern)

        # forward pass
        outputs_l_j = mlp_functions.forward_pass(neurons_l, weights_l_i_j,
                                                 outputs_l_j)

        # update training_error
        output_pattern = p[config.params['input_dimensions']:]
        teacher_i = []
        # account for i = 0
        teacher_i.append(None)
        teacher_i.extend(output_pattern)

        training_error = mlp_functions.update_ms_error(
                            neurons_l, training_error, teacher_i, outputs_l_j)

        # calculate errors
        errors_l_i = mlp_functions.calculate_errors(neurons_l, weights_l_i_j,
                                                    teacher_i, outputs_l_j)

        # update weights
        weights_l_i_j = mlp_functions.update_weights(neurons_l, weights_l_i_j,
                                                     errors_l_i, outputs_l_j)

    # normalise mean squared training error into [0,1] and convert to rms
    training_error = math.sqrt(training_error / (neurons_l[-1] *
                                                 len(training_patterns)))

    # write out epoch training_error
    training_errors.append(training_error)

    if config.params['validating']:
        validation_error = 0.0

        for p in validation_patterns:
            # load pattern
            input_pattern = p[:config.params['input_dimensions']]
            # set bias 'output'
            outputs_l_j = mlp_functions.initialise_bias(
                            config.params['neuron_layers'])
            # add input pattern to 'output' of layer 0 (i.e. set the input to p
            outputs_l_j[0].extend(input_pattern)

            # forward pass
            outputs_l_j = mlp_functions.forward_pass(neurons_l, weights_l_i_j,
                                                     outputs_l_j)

            # update validation error
            output_pattern = p[config.params['input_dimensions']:]
            teacher_i = []
            # account for i = 0
            teacher_i.append(None)
            teacher_i.extend(output_pattern)
            validation_error = mlp_functions.update_ms_error(
                        neurons_l, validation_error, teacher_i, outputs_l_j)

        # normalise mean squared validation error into [0,1] and convert to rms
        validation_error = math.sqrt(validation_error / (neurons_l[-1] * (
                                                    len(validation_patterns))))

        # make sure validation error is dropping
        if (validation_error < validation_error_best):
            validation_error_best = validation_error
            best_weights_l_i_j = list(weights_l_i_j)

        validation_errors.append(validation_error)

    epoch += 1
    if (training_error < config.params['target_training_error'] or
            epoch == config.params['max_epochs']):
        repeat = False

# testing loop
if (config.params['testing']):
    testing_error = 0.0
    testing_errors = []

    for p in test_patterns:
        # load pattern
        input_pattern = p[:config.params['input_dimensions']]
        # set bias 'output'
        outputs_l_j = mlp_functions.initialise_bias(
                        config.params['neuron_layers'])
        # add input pattern to 'output' of layer 0 (i.e. set the input to p)
        outputs_l_j[0].extend(input_pattern)

        # forward pass
        # use weight at lowest validation error
        if (config.params['validating'] and config.params['best_weights']):
            outputs_l_j = mlp_functions.forward_pass(neurons_l,
                                                     best_weights_l_i_j,
                                                     outputs_l_j)
        # use weight at lowest training error
        else:
            outputs_l_j = mlp_functions.forward_pass(neurons_l,
                                                     weights_l_i_j,
                                                     outputs_l_j)

        # update test error
        output_pattern = p[config.params['input_dimensions']:]
        teacher_i = []
        # account for i = 0
        teacher_i.append(None)
        teacher_i.extend(output_pattern)

        testing_error = mlp_functions.update_ms_error(
                            neurons_l, testing_error, teacher_i, outputs_l_j)
        testing_errors.append(testing_error)

    # normalise mean squared testing error into [0,1] and convert to rms
    testing_error = math.sqrt(testing_error / (neurons_l[-1] *
                                               len(test_patterns)))

# plot some data
print(testing_errors)

plt.subplot(211)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.plot(training_errors, 'b', validation_errors, 'r--')

plt.subplot(212)
plt.xlabel('Error')
plt.ylabel('Number')

plt.hist(testing_errors, 10, (0, 1))
plt.show()
