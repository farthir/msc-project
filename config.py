# module to load config from parameters file so it's globally accessible
import json
import sys

# read in parameters
parameters_filename = sys.argv[2]
with open('parameters/%s.json' % parameters_filename) as parameters_file:
    params = json.load(parameters_file)
