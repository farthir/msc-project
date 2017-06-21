# mlp_functions module
# assumes all input features preceed output(s)
import csv
import random


def read_patterns(data_file_name):

    # num_output_neurons = num_output_dimensions -- no! -- depends on
    # whether classifier or regression and then if classifier, the encoding

    with open(data_file_name, 'r', encoding='utf-8-sig') as data_file:
        reader = csv.reader(data_file)
        patterns = []
        for row in reader:
            patterns.append(row)

            # input pattern
            # x_vector = row[0:num_input_dimensions]
            # t_vector = row[num_input_dimensions:num_output_dimensions]

            # x_patterns.append(x_vector)

        # randomise csv input to make sure training/test sets are truely random
        random.shuffle(patterns)

    return patterns
