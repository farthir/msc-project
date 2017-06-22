# module to load config from parameters file so it's globally accessible
import json

# static parameters
parameters_filename = 'parameters.json'

# read in parameters
with open(parameters_filename) as parameters_file:
    params = json.load(parameters_file)
