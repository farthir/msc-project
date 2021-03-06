"""Module containing helper functions for reading from and writing to files"""
import csv
import random

from pathlib import Path

def read_patterns_structure(structure_filepath):
    """Function to read data structure file"""

    with open(structure_filepath, 'r', encoding='utf-8-sig') as structure_file:
        reader = csv.reader(structure_file)

        structure = next(reader)

    return structure

def read_patterns(data_filepath, params):
    """Function to read data patterns from a file"""
    # num_output_neurons = num_output_dimensions -- no! -- depends on
    # whether classifier or regression and then if classifier, the encoding

    with open(data_filepath, 'r', encoding='utf-8-sig') as data_file:
        reader = csv.reader(data_file)
        patterns = []
        for row in reader:
            patterns.append(row)

        # randomise csv input to make sure training/validation sets are truely random
        rng = random.Random(params['random_numbers_seed'])
        rng.shuffle(patterns)

    return patterns


def write_result_row(results_filepath, headers, result):
    """Function to write a single row to a file"""
    existing_path = Path(results_filepath)

    if existing_path.is_file():
        with open(results_filepath, 'a', encoding='utf-8-sig') as results_file:
            writer = csv.writer(results_file)
            writer.writerow(result)
    else:
        with open(results_filepath, 'w', encoding='utf-8-sig') as results_file:
            writer = csv.writer(results_file)
            if headers is not None:
                writer.writerow(headers)
            writer.writerow(result)
