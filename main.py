# driver module for programme
import mlp_functions
import config
from pprint import pprint

# read in patterns
patterns = mlp_functions.read_patterns('data/xor.csv')

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

# separate input vector from teaching/output vector
# this entire section can probably be removed
training_x = []
training_t = []
for p in training_patterns:
    training_x.append(p[:config.params['input_dimensions']])
    training_t.append(p[config.params['input_dimensions']:])

if config.params['validating']:
    # separate input vector from teaching/output vector
    validation_x = []
    validation_t = []
    for p in validation_patterns:
        validation_x.append(p[:config.params['input_dimensions']])
        validation_t.append(p[config.params['input_dimensions']:])

# network initialisation
# should probably create a neuron class to do this properly as it's a mess
# set number of neurons in each layer
# layer '0' is the input layer and defines number of inputs
neurons_l = []
neurons_l.append(config.params['input_dimensions'])

if config.params['neuron_layers'] > 1:
    for l in range(1, config.params['neuron_layers']):
        neurons_l.append(3)  # this needs to be input somehow

# this may not always be the case but is set here
neurons_l.append(config.params['output_dimensions'])

# get weights
weights_l_i_j = mlp_functions.initialise_weights(neurons_l)

# backpropagation loop
epoch = 0

if config.params['validating']:
    error_validation_prev = 1000.0

repeat = True
while (repeat):
    error = 0.0
    for p in training_patterns:
        # load pattern
        input_pattern = p[:config.params['input_dimensions']]
        outputs_l_j = []
        # set bias 'output'
        outputs_l_j = mlp_functions.initialise_bias(
                        config.params['neuron_layers'])
        # add input pattern to 'output' of layer 0 (i.e. set the input to p)
        outputs_l_j[0].extend(input_pattern)

    repeat = False

    if (error < config.params['target_training_error'] or
            epoch == config.params['max_epochs']):
        repeat = False

# temp
pprint(config.params)
print(neurons_l)
print(weights_l_i_j)
