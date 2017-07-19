"""Module containing classes and methods for data processing"""
import copy
import numpy as np
import pandas as pd

class ProcessingError(Exception):
    """Base class for exceptions in this module."""
    pass

class VariableTypeError(ProcessingError):
    """Raise error if type of variable not handled."""
    pass

class NotFullyImplementedError(ProcessingError):
    """Raise this error if there is an attempt to call functionality 
    that is not fully implemented"""
    pass

class Standardiser(object):
    """Class to standardise input data.
    Assumes correctly formatted patterns.
    """

    # variable types determine how the standardiser handles the data
    # choose from numeric, category, binary, none, or float type for each column

    def __init__(
            self, patterns_in, variable_types, variables_mean=None, variables_std=None):
        self.patterns_in = patterns_in
        self.patterns_out = copy.deepcopy(patterns_in)
        self.variable_types = variable_types

        if variables_mean is None and variables_std is None:
            self.variables_mean = []
            self.variables_std = []
            self.training_data = True
        else:
            self.variables_mean = variables_mean
            self.variables_std = variables_std
            self.training_data = False

    def standardise_by_type(self):
        """Method to standardise all patterns in class instance."""

        # loop through columns in data (currently ordered by row in list)
        # two counters, one for input structure and one for output structure
        variable_position_out = 0
        for variable_position_in in range(len(self.variable_types)):
            variable_data = [item[variable_position_in] for item in self.patterns_in]
            variable_type = self.variable_types[variable_position_in]

            # usually one standardised variable for one variable
            variable_count = 1

            if self.__is_scale_type(variable_type):
                standardised_data = self.__scale(variable_data, float(variable_type))
            elif variable_type == 'numeric':
                if self.training_data:
                    standardised_data, mean, std = self.__standardise_numeric(variable_data)
                    self.variables_mean.append(mean)
                    self.variables_std.append(std)
                else:
                    standardised_data, mean, std = self.__standardise_numeric(
                        variable_data,
                        self.variables_mean[variable_position_out],
                        self.variables_std[variable_position_out])
            elif variable_type == 'category_effects':
                raise VariableTypeError(
                    'ERROR: categories not fully implemented. Disable exception at own risk.')
                # also get a variable_count for categories as it will be more than
                # input variable count
                standardised_data, variable_count = self.__standardise_category(
                    variable_data)

                if self.training_data:
                    for _ in range(variable_count):
                        self.variables_mean.append(None)
                        self.variables_std.append(None)
            elif variable_type == 'category_dummy':
                raise VariableTypeError(
                    'ERROR: categories not fully implemented. Disable exception at own risk.')
                standardised_data, variable_count = self.__standardise_category(
                    variable_data, effects_coding=False)

                if self.training_data:
                    for _ in range(variable_count):
                        self.variables_mean.append(None)
                        self.variables_std.append(None)
            elif variable_type == 'binary':
                standardised_data = self.__standardise_binary(variable_data)

                if self.training_data:
                    self.variables_mean.append(None)
                    self.variables_std.append(None)
            elif variable_type == 'none':
                # serves as a way to disable standardisation for variables (e.g. on output)
                standardised_data = [float(item) for item in variable_data]
                
                if self.training_data:
                    for _ in range(variable_count):
                        self.variables_mean.append(None)
                        self.variables_std.append(None)

            else:
                raise VariableTypeError(
                    'ERROR: variable type "' + variable_type + '" not implemented.')

            for pattern in range(len(self.patterns_in)):
                # handle category data containing more than one row
                if variable_type.startswith('category'):
                    self.patterns_out[pattern][
                        variable_position_out:variable_position_out+1] = standardised_data[pattern]
                else:
                    self.patterns_out[pattern][variable_position_out] = standardised_data[pattern]
            variable_position_out += variable_count

    def __standardise_numeric(self, variable_data, mean=None, std=None):
        """Method that standardises numeric data using gaussian coding with
        mean and standard deviation."""
        # convert values in list to float
        variable_data = [float(item) for item in variable_data]

        # calculate mean and standard deviation if training else use existing (provided)
        if self.training_data:
            np_array = np.array(variable_data)
            mean = np.mean(np_array).item()
            std = np.std(np_array).item()

        # return new list containing standardised data and
        # mean and std for use on other data
        return [((item - mean) / std) for item in variable_data], mean, std

    def __standardise_category(self, variable_data, effects_coding=True):
        """Method that standardises category data by finding C-1 encoding
        and then converting ommitted category (all 0's) to -1 for effects encoding.
        """
        if effects_coding:
            # get 1-of-(C-1) encoded data
            # convert to float as default is uint
            # look for row of zeros and modify to -1 for effects coding
            pd_enc = pd.get_dummies(variable_data, drop_first=True).astype(float)
            pd_enc[(pd_enc.T == 0).all()] = -1
        else:
            # get 1-of-(C) encoded data
            # convert to float to match data types as default is uint
            # result is 1-of-(C) dummy coding
            pd_enc = pd.get_dummies(variable_data).astype(float)

        # return as a list of lists to replace and extend original data
        # also return number of columns now used to categorisation
        return pd_enc.values.tolist(), pd_enc.shape[1]

    def __standardise_binary(self, variable_data):
        """Method that standardises binary data by changing 0 to -1."""
        # return new list containing standardised data

        return [float((-1 if item == '0' else 1)) for item in variable_data]

    def __scale(self, variable_data, multiplier):
        """Method that applies linear scaling to data"""
        # convert values in list to float
        variable_data = [float(item) for item in variable_data]

        return [item * multiplier for item in variable_data]

    def __is_scale_type(self, variable_type):
        """Method that checks to see if variable_type is a float"""
        try:
            float(variable_type)
            return True
        except ValueError:
            return False

class Destandardiser(object):
    """Class to reverse the standardisation process to produce interpretable outputs.
    Assumes correctly formatted patterns.
    """

    # variable types determine how the destandardiser handles the data
    # choose from numeric, category, binary, none, or float type for each column

    def __init__(
            self, patterns_in, variable_types, variables_mean=None, variables_std=None):
        self.patterns_in = patterns_in
        self.patterns_out = copy.deepcopy(patterns_in)
        self.variable_types = variable_types

        self.variables_mean = variables_mean
        self.variables_std = variables_std

    def destandardise_single(self, pattern_number):
        """Method to destandardise a single pattern."""

        # loop through columns in data (currently ordered by row in list)
        # two counters, one for input structure and one for output structure
        variable_position_out = 0

        pattern_out = []

        for variable_position_in in range(len(self.variable_types)):
            variable_data = [self.patterns_in[pattern_number][variable_position_in]]
            variable_type = self.variable_types[variable_position_in]

            # usually one standardised variable for one variable
            variable_count = 1

            if is_scale_type(variable_type):
                destandardised_data = self.__descale(variable_data, float(variable_type))
            elif variable_type == 'numeric':
                destandardised_data = self.__destandardise_numeric(
                    variable_data,
                    self.variables_mean[variable_position_out],
                    self.variables_std[variable_position_out])
            elif variable_type == 'binary':
                destandardised_data = self.__destandardise_binary(variable_data)
            elif variable_type == 'none':
                # serves as a way to disable destandardisation for variables (e.g. on output)
                destandardised_data = variable_data
            else:
                raise VariableTypeError(
                    'ERROR: variable type "' + variable_type + '" not implemented.')

            pattern_out.append(destandardised_data)

            variable_position_out += variable_count

        return pattern_out

    def destandardise_by_type(self):
        """Method to destandardise all patterns in class instance."""

        # loop through columns in data (currently ordered by row in list)
        # two counters, one for input structure and one for output structure
        variable_position_out = 0
        for variable_position_in in range(len(self.variable_types)):
            variable_data = [item[variable_position_in] for item in self.patterns_in]
            variable_type = self.variable_types[variable_position_in]

            # usually one standardised variable for one variable
            variable_count = 1

            if is_scale_type(variable_type):
                destandardised_data = self.__descale(variable_data, float(variable_type))
            elif variable_type == 'numeric':
                destandardised_data = self.__destandardise_numeric(
                    variable_data,
                    self.variables_mean[variable_position_out],
                    self.variables_std[variable_position_out])
            elif variable_type == 'binary':
                destandardised_data = self.__destandardise_binary(variable_data)
            elif variable_type == 'none':
                # serves as a way to disable destandardisation for variables (e.g. on output)
                destandardised_data = variable_data
            else:
                raise VariableTypeError(
                    'ERROR: variable type "' + variable_type + '" not implemented.')

            for pattern in range(len(self.patterns_in)):
                self.patterns_out[pattern][variable_position_out] = destandardised_data[pattern]
            variable_position_out += variable_count

    def __destandardise_numeric(self, variable_data, mean, std):
        """Method that destandardises numeric data using gaussian coding with
        mean and standard deviation."""

        # return new list containing destandardised data
        return [((item * std) + mean) for item in variable_data]

    def __destandardise_binary(self, variable_data):
        """Method that destandardises binary data by changing -1 to 0."""
        # return new list containing destandardised data

        return [float((0 if item == -1 else 1)) for item in variable_data]

    def __descale(self, variable_data, multiplier):
        """Method that reverses the linear scaling applied during data preprocessing"""
        return [float(item / multiplier) for item in variable_data]

def is_scale_type(variable_type):
    """Utility method that checks to see if variable_type is a float"""
    try:
        float(variable_type)
        return True
    except ValueError:
        return False
