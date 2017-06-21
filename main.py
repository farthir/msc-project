# driver module for programme

# import file_handling
import mlp_functions
# import sys
import json
from pprint import pprint

# static parameters
parameters_filename = 'parameters.json'

# read in parameters
with open(parameters_filename) as parameters_file:
    parameters = json.load(parameters_file)

if parameters['training_test_ratio'] == 1:
    parameters['validating'] = False
else:
    parameters['validating'] = True

# read in patterns
patterns = mlp_functions.read_patterns('data/xor.csv')

# get training patterns
num_training_patterns = int(len(patterns) * parameters['training_test_ratio'])
training_patterns = patterns[:num_training_patterns]

# separate input vector from teaching/output vector
training_x = []
training_t = []
for p in training_patterns:
    training_x.append(p[:parameters['input_dimensions']])
    training_t.append(p[parameters['input_dimensions']:])

if parameters['validating']:
    num_test_patterns = len(patterns) - num_training_patterns
    test_patterns = patterns[num_training_patterns:]

    # separate input vector from teaching/output vector
    test_x = []
    test_t = []
    for p in test_patterns:
        test_x.append(p[:parameters['input_dimensions']])
        test_t.append(p[parameters['input_dimensions']:])

# temp
pprint(parameters)
print(patterns)
print(training_x)
print(training_t)
print(test_x)
print(test_t)
