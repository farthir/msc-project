"""Module containing classes and methods for input data processing"""

import numpy as np
import pandas as pd

class InputProcessingError(Exception):
    """Base class for exceptions in this module."""
    pass

class VariableTypeError(InputProcessingError):
    """Raise error if type of variable not handled."""
    pass

class Standardiser(object):
    """Class to standardise input data.
    Assumes correctly formatted patterns.
    """

    # variable types determine how the standardiser handles the data
    # choose from numeric, category, binary for each column

    def __init__(self, params, patterns, variable_types, variables_mean=None, variables_std=None):
        self.params = params
        self.patterns = patterns
        self.variable_types = variable_types

        if variables_mean is None and variables_std is None:
            self.variables_mean = []
            self.variables_std = []
            self.training_data = True
        else:
            self.variables_mean = variables_mean
            self.variables_std = variables_std
            self.training_data = False

    def standardise_all_by_type(self):
        """Method to standardise all patterns in class instance."""

        # loop through columns in data (currently ordered by row in list)
        for i in range(self.params['input_dimensions']):
            variable_data = [item[i] for item in self.patterns]
            variable_type = self.variable_types[i]

            if variable_type == 'numeric':
                if self.training_data:
                    standardised_data, mean, std = self.__standardise_numeric(variable_data)
                    self.variables_mean.append(mean)
                    self.variables_std.append(std)
                else:
                    standardised_data, mean, std = self.__standardise_numeric(
                        variable_data, self.variables_mean[i], self.variables_std[i])
            elif variable_type == 'category':
                standardised_data = self.__standardise_category(variable_data)
                if self.training_data:
                    self.variables_mean.append(None)
                    self.variables_std.append(None)
            elif variable_type == 'binary':
                standardised_data = self.__standardise_binary(variable_data)
                if self.training_data:
                    self.variables_mean.append(None)
                    self.variables_std.append(None)
            else:
                raise VariableTypeError('ERROR: variable type "' + variable_type + '" unhandled.')

            for pattern in range(len(self.patterns)):
                # handle category data containing more than one row
                if variable_type == 'category':
                    self.patterns[pattern][i:i+1] = standardised_data[pattern]
                else:
                    self.patterns[pattern][i] = standardised_data[pattern]

    def __standardise_numeric(self, variable_data, mean=None, std=None):
        """Method that standardises numeric data using gaussian coding with
        mean and standard deviation."""
        # calculate mean and standard deviation if training else use existing (provided)
        if self.training_data:
            np_array = np.array(variable_data)
            mean = np.mean(np_array)
            std = np.std(np_array)

        # return new list containing standardised data and
        # mean and std for use on other data
        return [((item - mean) / std) for item in variable_data], mean, std

    def __standardise_category(self, variable_data, effects_coding=True):
        """Method that standardises category data by finding C-1 encoding
        and then converting ommitted category (all 0's) to -1 for effects encoding.
        """
        if effects_coding:
            # get 1-of-(C-1) encoded data
            # convert to int as default is uint
            # look for row of zeros and modify to -1 for effects coding
            pd_enc = pd.get_dummies(variable_data, drop_first=True).astype(int)
            pd_enc[(pd_enc.T == 0).all()] = -1
        else:
            # get 1-of-(C) encoded data
            # convert to int as default is uint
            # result is 1-of-(C) dummy coding
            pd_enc = pd.get_dummies(variable_data).astype(int)

        # return as a list of lists to replace and extend original data
        return pd_enc.values.T.tolist()

    def __standardise_binary(self, variable_data):
        """Method that standardises binary data by changing 0 to -1."""
        # return new list containing standardised data

        return [(-1 if item == 0 else 1) for item in variable_data]
