"""Module defining multi-layer perceptron backpropagation neural network class"""
import json
import progressbar

import mlp_functions
import io_functions
import data_processing

class MLPNetwork(object):
    """Class to build an MLP network using stochastic gradient descent backpropagation learning"""
    def __init__(self, unified_filename, results_filename):
        self.results_filename = results_filename

        self.__read_parameters(unified_filename)
        self.__read_structure(unified_filename)
        self.__read_data(unified_filename)
        self.__data_preprocessing()
        self.__initialise_network()
        self.__backpropagation_loop()
        self.average_testing_error = 0
        self.__testing_loop()

        if self.params['save_summary']:
            self.__save_summary(results_filename)

    def __read_parameters(self, parameters_filename):
        with open('parameters/%s.json'
                  % parameters_filename) as parameters_file:
            self.params = json.load(parameters_file)

    def __read_structure(self, structure_filename):
        self.variable_types = io_functions.read_patterns_structure('data/%s_structure.csv'
                                                                   % structure_filename)

    def __read_data(self, data_filename):
        # read in training/validation patterns
        self.patterns = io_functions.read_patterns('data/%s.csv' % data_filename, self.params)

        # read in test patterns
        if self.params['testing']:
            self.test_patterns = io_functions.read_patterns('data/%s_test.csv'
                                                            % data_filename, self.params)

        # useful variable
        self.last_output = (self.params['input_dimensions'] + self.params['output_dimensions'])

    def __data_preprocessing(self):
        """Method to pre-process the data"""

        # split the data into training and validation patterns first
        if (self.params['validating'] and
                self.params['training_validation_ratio'] == 1):
            self.training_patterns = self.patterns
            self.validation_patterns = self.patterns

        elif (self.params['validating'] and
              self.params['training_validation_ratio'] < 1):
            num_training_patterns = int(
                len(self.patterns) * self.params['training_validation_ratio'])
            self.training_patterns = self.patterns[:num_training_patterns]
            self.validation_patterns = self.patterns[num_training_patterns:]

        elif not self.params['validating']:
            self.training_patterns = self.patterns

        # standardise the data
        # training patterns
        training_standardiser = data_processing.Standardiser(
            self.training_patterns, self.variable_types)

        training_standardiser.standardise_by_type()
        self.training_patterns = training_standardiser.patterns_out

        # validation patterns
        if self.params['validating']:
            validation_standardiser = data_processing.Standardiser(
                self.validation_patterns,
                self.variable_types,
                variables_mean=training_standardiser.variables_mean,
                variables_std=training_standardiser.variables_std)

            validation_standardiser.standardise_by_type()
            self.validation_patterns = validation_standardiser.patterns_out

        # test patterns
        if self.params['testing']:
            test_standardiser = data_processing.Standardiser(
                self.test_patterns,
                self.variable_types,
                variables_mean=training_standardiser.variables_mean,
                variables_std=training_standardiser.variables_std)

            test_standardiser.standardise_by_type()
            self.test_patterns = test_standardiser.patterns_out

        self.training_standardiser = training_standardiser

        # if scaling the output, adjust target training error accordingly
        if self.params['output_dimensions'] == 1:
            if data_processing.is_scale_type(self.variable_types[-1]):
                # to add effect of scalar, multiply by scale value
                target_training_error = (
                    self.params['target_training_error'] * float(self.variable_types[-1]))
            elif self.variable_types[-1] == 'numeric':
                # to add effect of numeric standardisation, divide by standard deviation
                target_training_error = (
                    self.params['target_training_error'] /
                    self.training_standardiser.variables_std[-1])
            else:
                target_training_error = self.params['target_training_error']
        else:
            target_training_error = self.params['target_training_error']

        self.target_training_error = target_training_error

    def __initialise_network(self):
        # network initialisation
        # set number of neurons in each layer
        # layer '0' is the input layer and defines number of inputs
        neurons_l = []
        neurons_l.append(self.params['input_dimensions'])

        for hidden_node_index in range(len(self.params['hidden_nodes'])):
            neurons_l.append(self.params['hidden_nodes'][hidden_node_index])

        # this may not always be the case but is set here
        neurons_l.append(self.params['output_dimensions'])

        # initialise weights
        weights_l_i_j = mlp_functions.initialise_weights(self.params,
                                                         neurons_l)

        self.neurons_l = neurons_l
        self.weights_l_i_j = weights_l_i_j

    def __backpropagation_loop(self):
        # backpropagation loop
        # epoch count starts at one
        epoch = 1
        training_errors = []
        repeat = True
        target_training_error_reached = False

        if self.params['validating']:
            validation_errors = []
            validation_error_best = 1000.0
            training_error_best = 0.0
            epoch_best = 0

        # initialise progress bar for console
        progress_bar = progressbar.ProgressBar(
            maxval=self.params['max_epochs'],
            widgets=[progressbar.Bar(
                '=', '[', ']'), ' ', progressbar.Percentage()])
        progress_bar.start()
        while repeat:
            training_error = 0.0
            progress_bar.update(epoch)

            for pattern in self.training_patterns:
                # load pattern
                input_pattern = pattern[:self.params['input_dimensions']]
                # set bias 'output'
                outputs_l_j = mlp_functions.initialise_bias(self.params)
                # add input pattern to 'output' of layer 0
                outputs_l_j[0].extend(input_pattern)

                # forward pass
                outputs_l_j = mlp_functions.forward_pass(
                    self.params, self.neurons_l, self.weights_l_i_j,
                    outputs_l_j)

                # update training_error
                output_pattern = pattern[self.params['input_dimensions']:
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

            # calculate rms training error
            training_error = mlp_functions.calculate_rms_error(
                self.params['output_function'],
                training_error,
                self.neurons_l[-1],
                len(self.training_patterns)
            )

            # write out epoch training_error
            training_errors.append(training_error)

            # Write out weights and errors if specified
            if self.params['save_network']:
                if epoch % self.params['save_network_resolution'] == 0:
                    # append results to file
                    headers = (['epoch'] +
                               ['weight_%s_%s_%s' % (l+1, i+1, j)
                                for l in range(len(self.weights_l_i_j[1:]))
                                for i in range(len(self.weights_l_i_j[l+1][1:]))
                                for j in range(len(self.weights_l_i_j[l+1][i+1]))] +
                               ['error_%s_%s' % (l+1, i+1)
                                for l in range(len(errors_l_i[1:]))
                                for i in range(len(errors_l_i[l+1][1:]))])

                    result = [epoch]
                    result.extend(
                        [j for l in range(len(self.weights_l_i_j[1:]))
                         for i in range(len(self.weights_l_i_j[l+1][1:]))
                         for j in self.weights_l_i_j[l+1][i+1]])
                    result.extend(
                        [i for l in range(len(errors_l_i[1:]))
                         for i in errors_l_i[l+1][1:]])

                    io_functions.write_result_row(
                        'results/%s_weights.csv' % self.results_filename, headers, result)

            if self.params['validating']:
                validation_error = 0.0

                for pattern in self.validation_patterns:
                    # load pattern
                    input_pattern = pattern[:self.params['input_dimensions']]
                    # set bias 'output'
                    outputs_l_j = mlp_functions.initialise_bias(self.params)
                    # add input pattern to 'output' of layer 0
                    outputs_l_j[0].extend(input_pattern)

                    # forward pass
                    outputs_l_j = mlp_functions.forward_pass(
                        self.params, self.neurons_l, self.weights_l_i_j,
                        outputs_l_j)

                    # update validation error
                    output_pattern = pattern[self.params['input_dimensions']:
                                             self.last_output]

                    teacher_i = []
                    # account for i = 0
                    teacher_i.append(None)
                    teacher_i.extend(output_pattern)
                    validation_error = mlp_functions.update_ms_error(
                        self.neurons_l, validation_error, teacher_i,
                        outputs_l_j)

                # calculate rms validation error
                validation_error = mlp_functions.calculate_rms_error(
                    self.params['output_function'],
                    validation_error,
                    self.neurons_l[-1],
                    len(self.validation_patterns)
                )

                # make sure validation error is dropping
                if validation_error < validation_error_best:
                    validation_error_best = validation_error
                    best_weights_l_i_j = list(self.weights_l_i_j)
                    epoch_best = epoch

                validation_errors.append(validation_error)

            # record when target training error was reached
            if not target_training_error_reached:
                epoch_target_training_error = epoch
                if training_error < self.target_training_error:
                    target_training_error_reached = True

            # network training halting conditions
            if self.params['stop_at_target_training_error']:
                if (training_error < self.target_training_error or
                        epoch == self.params['max_epochs']):
                    repeat = False
            else:
                if epoch == self.params['max_epochs']:
                    repeat = False

            # finally, increment the epoch
            epoch += 1

        # reverse effects of standardiser on error when we only have a single output
        # only modifies error with numeric standardised outputs and scaled outputs
        if self.params['output_dimensions'] == 1:
            if data_processing.is_scale_type(self.variable_types[-1]):
                training_destandardiser_error = data_processing.Destandardiser(
                    [[item] for item in training_errors],
                    [self.variable_types[-1]])
                training_destandardiser_error.destandardise_by_type()
                training_errors = [
                    item[0] for item in training_destandardiser_error.patterns_out]
                if self.params['validating']:
                    validation_destandardiser_error = data_processing.Destandardiser(
                        [[item] for item in validation_errors],
                        [self.variable_types[-1]])
                    validation_destandardiser_error.destandardise_by_type()
                    validation_errors = [
                        item[0] for item in validation_destandardiser_error.patterns_out]
            elif self.variable_types[-1] == 'numeric':
                training_destandardiser_error = data_processing.Destandardiser(
                    [[item] for item in training_errors],
                    [self.variable_types[-1]],
                    variables_mean=[0],
                    variables_std=[self.training_standardiser.variables_std[-1]])
                training_destandardiser_error.destandardise_by_type()
                training_errors = [
                    item[0] for item in training_destandardiser_error.patterns_out]
                if self.params['validating']:
                    validation_destandardiser_error = data_processing.Destandardiser(
                        [[item] for item in validation_errors],
                        [self.variable_types[-1]],
                        variables_mean=[0],
                        variables_std=[self.training_standardiser.variables_std[-1]])
                    validation_destandardiser_error.destandardise_by_type()
                    validation_errors = [
                        item[0] for item in validation_destandardiser_error.patterns_out]

        # data for summary results
        self.training_errors = training_errors
        self.epoch_end = epoch - 1 # subtract one as increment occurs before while loop ends
        self.epoch_target_training_error = epoch_target_training_error
        self.training_error_end = training_errors[-1]

        if self.params['validating']:
            self.best_weights_l_i_j = best_weights_l_i_j
            self.validation_errors = validation_errors
            self.validation_error_end = validation_errors[-1]
            self.training_error_best = training_errors[epoch_best - 1] # epoch indexed from 1
            self.validation_error_best = validation_errors[epoch_best - 1] # epoch indexed from 1
            self.epoch_best = epoch_best

        # write out detailed results if specified
        if self.params['save_detailed']:
            headers = (['epoch'] +
                       ['training_error'] +
                       ['validation_error']
                      )
            for epoch_index, training_error in enumerate(training_errors):
                result = []
                if self.params['validating']:
                    result.append(epoch_index + 1) # start epoch count at one
                    result.append(training_error)
                    result.append(validation_errors[epoch_index])
                else:
                    result.append(epoch_index + 1)
                    result.append(training_error)

                io_functions.write_result_row(
                    'results/%s_detailed.csv' % self.results_filename, headers, result)

    def __testing_loop(self):
        # testing loop
        if self.params['testing']:
            testing_errors = []
            all_outputs_l_j = []

            for pattern in self.test_patterns:
                testing_error = 0.0
                # load pattern
                input_pattern = pattern[:self.params['input_dimensions']]

                # set bias 'output'
                outputs_l_j = mlp_functions.initialise_bias(self.params)
                # add input pattern to 'output' of layer 0
                outputs_l_j[0].extend(input_pattern)

                # forward pass
                # use weight at lowest validation error
                if self.params['validating'] and self.params['best_weights']:
                    outputs_l_j = mlp_functions.forward_pass(
                        self.params, self.neurons_l, self.best_weights_l_i_j,
                        outputs_l_j)
                # use weight at lowest training error
                else:
                    outputs_l_j = mlp_functions.forward_pass(
                        self.params, self.neurons_l, self.weights_l_i_j,
                        outputs_l_j)

                # update test error
                output_pattern = pattern[self.params['input_dimensions']:
                                         self.last_output]

                teacher_i = []
                # account for i = 0
                teacher_i.append(None)
                teacher_i.extend(output_pattern)

                testing_error = mlp_functions.update_ms_error(
                    self.neurons_l, testing_error, teacher_i, outputs_l_j)

                # calculate rms testing error
                testing_error = mlp_functions.calculate_rms_error(
                    self.params['output_function'],
                    testing_error,
                    self.neurons_l[-1],
                    1
                )

                all_outputs_l_j.append(outputs_l_j)
                testing_errors.append(testing_error)

            # remove standardisation effects from data
            test_destandardiser_data = data_processing.Destandardiser(
                self.test_patterns,
                self.variable_types,
                variables_mean=self.training_standardiser.variables_mean,
                variables_std=self.training_standardiser.variables_std)

            test_destandardiser_data.destandardise_by_type()

            # remove standardisation effects from net outputs
            test_destandardiser_net = data_processing.Destandardiser(
                [item[-1][1:] for item in all_outputs_l_j],
                self.variable_types[self.params['input_dimensions']:self.last_output],
                variables_mean=self.training_standardiser.variables_mean[
                    self.params['input_dimensions']:self.last_output],
                variables_std=self.training_standardiser.variables_std[
                    self.params['input_dimensions']:self.last_output])

            test_destandardiser_net.destandardise_by_type()

            # reverse effects of standardiser on error when we only have a single output
            # only modifies error with numeric standardised outputs and scaled outputs
            if self.params['output_dimensions'] == 1:
                if data_processing.is_scale_type(self.variable_types[-1]):
                    test_destandardiser_error = data_processing.Destandardiser(
                        [[item] for item in testing_errors],
                        [self.variable_types[-1]])
                    test_destandardiser_error.destandardise_by_type()
                elif self.variable_types[-1] == 'numeric':
                    test_destandardiser_error = data_processing.Destandardiser(
                        [[item] for item in testing_errors],
                        [self.variable_types[-1]],
                        variables_mean=[0],
                        variables_std=[self.training_standardiser.variables_std[-1]])
                    test_destandardiser_error.destandardise_by_type()
                else:
                    test_destandardiser_error = None
            else:
                test_destandardiser_error = None

            # append testing results to file
            if self.params['save_testing']:
                headers = (['input_%s' % i for i in range(len(input_pattern))] +
                           ['output_%s' % i for i in range(len(output_pattern))] +
                           ['test_output_%s' % i for i in range(len(outputs_l_j[-1][1:]))] +
                           ['testing_error']
                          )
                for pattern_number in range(len(test_destandardiser_data.patterns_out)):
                    if test_destandardiser_error is not None:
                        result = (test_destandardiser_data.patterns_out[pattern_number] +
                                  test_destandardiser_net.patterns_out[pattern_number] +
                                  test_destandardiser_error.patterns_out[pattern_number])
                    else:
                        result = (test_destandardiser_data.patterns_out[pattern_number] +
                                  test_destandardiser_net.patterns_out[pattern_number] +
                                  [testing_errors[pattern_number]])

                    io_functions.write_result_row(
                        'results/%s_testing.csv' % self.results_filename, headers, result)

            # calculate average testing error
            if test_destandardiser_error is not None:
                self.average_testing_error = (
                    sum([item[0] for item in test_destandardiser_error.patterns_out]) /
                    float(len(testing_errors)))
                self.testing_errors = [item[0] for item in test_destandardiser_error.patterns_out]
            else:
                self.average_testing_error = sum(testing_errors)/float(len(testing_errors))
                self.testing_errors = testing_errors

    def __save_summary(self, results_filename):
        # save some data
        headers = ['weight_init_mean', 'weight_init_range',
                   'random_numbers_seed', 'hidden_layers_function',
                   'output_function', 'training_rate',
                   'hidden_nodes', 'epoch_end', 'training_error_end',
                   'epoch_target_training_error', 'average_testing_error',
                   'validation_error_end', 'epoch_best',
                   'training_error_best', 'validation_error_best'
                  ]
        result = []
        result.append(self.params['weight_init_mean'])
        result.append(self.params['weight_init_range'])
        result.append(self.params['random_numbers_seed'])
        result.append(self.params['hidden_layers_function'])
        result.append(self.params['output_function'])
        result.append(self.params['training_rate'])
        result.append(self.params['hidden_nodes'])
        result.extend([self.epoch_end, self.training_error_end,
                       self.epoch_target_training_error, self.average_testing_error])

        if self.params['validating']:
            result.extend([self.validation_error_end, self.epoch_best,
                           self.training_error_best, self.validation_error_best])

        io_functions.write_result_row(
            'results/%s.csv' % results_filename, headers, result)
