"""Module defining multi-layer perceptron backpropagation neural network class"""
import math
import json
import progressbar

import mlp_functions
import io_functions
import data_processing


class Multilayer_perceptron(object):
    def __init__(self, unified_filename, results_filename):
        self.__read_parameters(unified_filename)
        self.__read_structure(unified_filename)
        self.__read_data(unified_filename)
        self.__data_preprocessing()
        self.__initialise_network()
        self.__backpropagation_loop()
        self.__testing_loop()
        self.__save_results(results_filename)

    def __read_parameters(self, parameters_filename):
        with open('parameters/%s.json'
                  % parameters_filename) as parameters_file:
            self.params = json.load(parameters_file)

    def __read_structure(self, structure_filename):
        self.variable_types = io_functions.read_patterns_structure('data/%s_structure.csv'
                                                                   % structure_filename)

    def __read_data(self, data_filename):
        patterns = io_functions.read_patterns('data/%s.csv' % data_filename)

        # read in test patterns
        if (self.params['testing']):
            self.test_patterns = io_functions.read_patterns('data/%s_test.csv'
                                                            % data_filename)

        if (self.params['validating'] and
                self.params['training_validation_ratio'] == 1):
            self.training_patterns = patterns
            self.validation_patterns = patterns

        elif (self.params['validating'] and
                self.params['training_validation_ratio'] < 1):
            num_training_patterns = int(
                len(patterns) * self.params['training_validation_ratio'])
            self.training_patterns = patterns[:num_training_patterns]
            self.validation_patterns = patterns[num_training_patterns:]

        elif not self.params['validating']:
            self.training_patterns = patterns
        
        # useful variable
        last_output = (self.params['input_dimensions'] + self.params['output_dimensions'])
        self.last_output = last_output

    def __data_preprocessing(self):
        """Method to standardise data"""

        # training patterns
        if self.params['standardise_input']:
            input_training_patterns = (
                [item[:self.params['input_dimensions']] for item in self.training_patterns])
            input_training_standardiser = data_processing.Standardiser(
                input_training_patterns, self.variable_types[:self.params['input_dimensions']])

            input_training_standardiser.standardise_by_type()

            # input_training_standardiser.patterns

        if self.params['standardise_output']:
            output_training_patterns = (
                [item[self.params['input_dimensions']:self.last_output]
                 for item in self.training_patterns])
            output_training_standardiser = data_processing.Standardiser(
                output_training_patterns,
                self.variable_types[self.params['input_dimensions']:self.last_output])

            output_training_standardiser.standardise_by_type()

        # validation patterns
        if self.params['validating']:
            if self.params['standardise_input']:
                input_validation_patterns = (
                    [item[:self.params['input_dimensions']] for item in self.validation_patterns])
                input_validation_standardiser = data_processing.Standardiser(
                    input_validation_patterns,
                    self.variable_types[:self.params['input_dimensions']],
                    variables_mean=input_training_standardiser.variables_mean,
                    variables_std=input_training_standardiser.variables_std)

                input_validation_standardiser.standardise_by_type()

            if self.params['standardise_output']:
                output_validation_patterns = (
                    [item[self.params['input_dimensions']:self.last_output]
                     for item in self.validation_patterns])
                output_validation_standardiser = data_processing.Standardiser(
                    output_validation_patterns,
                    self.variable_types[self.params['input_dimensions']:self.last_output],
                    variables_mean=output_training_standardiser.variables_mean,
                    variables_std=output_training_standardiser.variables_std)

                output_validation_standardiser.standardise_by_type()

        # test patterns
        if self.params['testing']:
            if self.params['standardise_input']:
                input_test_patterns = (
                    [item[:self.params['input_dimensions']] for item in self.test_patterns])
                input_testing_standardiser = data_processing.Standardiser(
                    input_test_patterns,
                    self.variable_types[:self.params['input_dimensions']],
                    variables_mean=input_training_standardiser.variables_mean,
                    variables_std=input_training_standardiser.variables_std)

                input_testing_standardiser.standardise_by_type()

            if self.params['standardise_output']:
                output_test_patterns = (
                    [item[self.params['input_dimensions']:self.last_output]
                     for item in self.test_patterns])
                output_test_standardiser = data_processing.Standardiser(
                    output_test_patterns,
                    self.variable_types[self.params['input_dimensions']:self.last_output],
                    variables_mean=output_training_standardiser.variables_mean,
                    variables_std=output_training_standardiser.variables_std)

                output_test_standardiser.standardise_by_type()

    def __initialise_network(self):
        # network initialisation
        # set number of neurons in each layer
        # layer '0' is the input layer and defines number of inputs
        neurons_l = []
        neurons_l.append(self.params['input_dimensions'])

        for l in range(len(self.params['hidden_nodes'])):
            neurons_l.append(self.params['hidden_nodes'][l])

        # this may not always be the case but is set here
        neurons_l.append(self.params['output_dimensions'])

        # initialise weights
        weights_l_i_j = mlp_functions.initialise_weights(self.params,
                                                         neurons_l)

        self.neurons_l = neurons_l
        self.weights_l_i_j = weights_l_i_j

    def __backpropagation_loop(self):
        # backpropagation loop
        epoch = 0
        training_errors = []
        repeat = True
        if self.params['validating']:
            validation_errors = []
            validation_error_best = 1000.0
            training_error_best = 0.0
            epoch_best = 0

        # initialise progress bar for console
        bar = progressbar.ProgressBar(
            maxval=self.params['max_epochs'],
            widgets=[progressbar.Bar(
                '=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        while (repeat):
            training_error = 0.0
            bar.update(epoch)

            for p in self.training_patterns:
                # load pattern
                input_pattern = p[:self.params['input_dimensions']]
                # set bias 'output'
                outputs_l_j = mlp_functions.initialise_bias(self.params)
                # add input pattern to 'output' of layer 0
                outputs_l_j[0].extend(input_pattern)

                # forward pass
                outputs_l_j = mlp_functions.forward_pass(
                    self.params, self.neurons_l, self.weights_l_i_j,
                    outputs_l_j)

                # update training_error
                output_pattern = p[self.params['input_dimensions']:
                                   self.last_output]
                teacher_i = []
                # account for i = 0
                teacher_i.append(None)
                teacher_i.extend(output_pattern)

                training_error = mlp_functions.update_ms_error(
                    self.neurons_l, training_error, teacher_i, outputs_l_j)

                # calculate errors
                errors_l_i = mlp_functions.calculate_errors(
                    self.params, self.neurons_l, self.weights_l_i_j,
                    teacher_i, outputs_l_j)

                # update weights
                self.weights_l_i_j = mlp_functions.update_weights(
                    self.params, self.neurons_l, self.weights_l_i_j,
                    errors_l_i, outputs_l_j)

            # normalise training error into [0,1] and convert to rms
            if self.params['output_function'] == "sigmoid":
                training_error = math.sqrt(
                    training_error / (
                        self.neurons_l[-1] * len(self.training_patterns)))
            elif self.params['output_function'] == "tanh":
                training_error = math.sqrt(
                    training_error / (
                        2 * self.neurons_l[-1] * len(self.training_patterns)))

            # write out epoch training_error
            training_errors.append(training_error)

            if self.params['validating']:
                validation_error = 0.0

                for p in self.validation_patterns:
                    # load pattern
                    input_pattern = p[:self.params['input_dimensions']]
                    # set bias 'output'
                    outputs_l_j = mlp_functions.initialise_bias(self.params)
                    # add input pattern to 'output' of layer 0
                    outputs_l_j[0].extend(input_pattern)

                    # forward pass
                    outputs_l_j = mlp_functions.forward_pass(
                        self.params, self.neurons_l, self.weights_l_i_j,
                        outputs_l_j)

                    # update validation error
                    output_pattern = p[self.params['input_dimensions']:
                                       self.last_output]

                    teacher_i = []
                    # account for i = 0
                    teacher_i.append(None)
                    teacher_i.extend(output_pattern)
                    validation_error = mlp_functions.update_ms_error(
                        self.neurons_l, validation_error, teacher_i,
                        outputs_l_j)

                # normalise validation error into [0,1] and convert to rms
                if self.params['output_function'] == "logistic":
                    validation_error = math.sqrt(
                        validation_error / (
                            self.neurons_l[-1] * len(self.validation_patterns)))
                elif self.params['output_function'] == "tanh":
                    validation_error = math.sqrt(
                        validation_error / (
                            2 * self.neurons_l[-1] * len(self.validation_patterns)))

                # make sure validation error is dropping
                if (validation_error < validation_error_best):
                    validation_error_best = validation_error
                    best_weights_l_i_j = list(self.weights_l_i_j)
                    training_error_best = training_error
                    epoch_best = epoch

                validation_errors.append(validation_error)


            epoch += 1
            if (training_error < self.params['target_training_error'] or
                    epoch == self.params['max_epochs']):
                repeat = False

        self.training_errors = training_errors
        self.epoch_end = epoch
        self.training_error_end = training_error

        if self.params['validating']:
            self.best_weights_l_i_j = best_weights_l_i_j
            self.validation_errors = validation_errors
            self.validation_error_end = validation_error
            self.training_error_best = training_error_best
            self.validation_error_best = validation_error_best
            self.epoch_best = epoch_best

    def __testing_loop(self):
        # testing loop
        if (self.params['testing']):
            testing_errors = []

            for p in self.test_patterns:
                testing_error = 0.0
                # load pattern
                input_pattern = p[:self.params['input_dimensions']]
                # set bias 'output'
                outputs_l_j = mlp_functions.initialise_bias(self.params)
                # add input pattern to 'output' of layer 0
                outputs_l_j[0].extend(input_pattern)

                # forward pass
                # use weight at lowest validation error
                if (self.params['validating'] and self.params['best_weights']):
                    outputs_l_j = mlp_functions.forward_pass(
                        self.params, self.neurons_l, self.best_weights_l_i_j,
                        outputs_l_j)
                # use weight at lowest training error
                else:
                    outputs_l_j = mlp_functions.forward_pass(
                        self.params, self.neurons_l, self.weights_l_i_j,
                        outputs_l_j)

                # update test error
                output_pattern = p[self.params['input_dimensions']:
                                   self.last_output]
                teacher_i = []
                # account for i = 0
                teacher_i.append(None)
                teacher_i.extend(output_pattern)

                testing_error = mlp_functions.update_ms_error(
                    self.neurons_l, testing_error, teacher_i, outputs_l_j)

                # normalise testing error into [0,1] and convert to rms
                if self.params['output_function'] == "logistic":
                    testing_error = math.sqrt(
                        testing_error / (
                            self.neurons_l[-1]))
                elif self.params['output_function'] == "tanh":
                    testing_error = math.sqrt(
                        testing_error / (
                            self.neurons_l[-1])) / 2

                testing_errors.append(testing_error)

            self.testing_errors = testing_errors

    def __save_results(self, results_filename):
        # save some data
        headers = ['weight_init_mean', 'weight_init_range',
                   'fixed_weight_seed', 'hidden_layers_function',
                   'output_function', 'training_rate',
                   'hidden_nodes', 'epoch_end', 'training_error_end',
                   'validation_error_end', 'epoch_best',
                   'training_error_best', 'validation_error_best'
                  ]
        result = []
        result.append(self.params['weight_init_mean'])
        result.append(self.params['weight_init_range'])
        result.append(self.params['fixed_weight_seed'])
        result.append(self.params['hidden_layers_function'])
        result.append(self.params['output_function'])
        result.append(self.params['training_rate'])
        result.append(self.params['hidden_nodes'])
        if self.params['validating']:
            result.extend([self.epoch_end, self.training_error_end,
                           self.validation_error_end, self.epoch_best,
                           self.training_error_best, self.validation_error_best])
        else:
            result.extend([self.epoch_end, self.training_error_end])
        io_functions.write_result_row(
            'results/%s.csv' % results_filename, headers, result)
