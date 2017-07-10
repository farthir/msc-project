"""Module defining multi-layer perceptron backpropagation neural network class"""
import json
import progressbar

import matplotlib.pyplot as plt

import mlp_functions
import io_functions
import data_processing

class MLPNetwork(object):
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
        # read in training/validation patterns
        self.patterns = io_functions.read_patterns('data/%s.csv' % data_filename)

        # read in test patterns
        if self.params['testing']:
            self.test_patterns = io_functions.read_patterns('data/%s_test.csv'
                                                            % data_filename)

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
        # WARN: Categorical standardisation is not implemented fully.
        #       Categories need to be found across whole pattern set and proper
        #       handling for test set if new categories are found as well as
        #       totals for number of inputs and outputs.
        # training patterns
        training_standardiser = data_processing.Standardiser(
            self.training_patterns, self.variable_types)

        training_standardiser.standardise_by_type()
        self.training_patterns = training_standardiser.patterns_out

        print(self.training_patterns)

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
        # TEMP - weights store
        weights_epoch_1_1_0=[]
        weights_epoch_1_1_1=[]
        weights_epoch_1_1_2=[]
        weights_epoch_1_2_0=[]
        weights_epoch_1_2_1=[]
        weights_epoch_1_2_2=[]
        weights_epoch_2_1_0=[]
        weights_epoch_2_1_1=[]
        weights_epoch_2_1_2=[]
        errors_epoch_1_1=[]
        errors_epoch_1_2=[]
        errors_epoch_2_1=[]
        

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
        while repeat:
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

            # calculate rms training error
            training_error = mlp_functions.calculate_rms_error(
                self.params['output_function'],
                training_error,
                self.neurons_l[-1],
                len(self.training_patterns)
            )

            # write out epoch training_error
            training_errors.append(training_error)

            # TEMP
            if (epoch % 100 == 0):
                weights_epoch_1_1_0.append(self.weights_l_i_j[1][1][0])
                weights_epoch_1_1_1.append(self.weights_l_i_j[1][1][1])
                weights_epoch_1_1_2.append(self.weights_l_i_j[1][1][2])
                weights_epoch_1_2_0.append(self.weights_l_i_j[1][2][0])
                weights_epoch_1_2_1.append(self.weights_l_i_j[1][2][1])
                weights_epoch_1_2_2.append(self.weights_l_i_j[1][2][2])
                weights_epoch_2_1_0.append(self.weights_l_i_j[2][1][0])
                weights_epoch_2_1_1.append(self.weights_l_i_j[2][1][1])
                weights_epoch_2_1_2.append(self.weights_l_i_j[2][1][2])
                errors_epoch_1_1.append(errors_l_i[1][1])
                errors_epoch_1_2.append(errors_l_i[1][2])
                errors_epoch_2_1.append(errors_l_i[2][1])

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

        plt.subplot(211)
        plt.xlabel('Epoch*100')
        plt.ylabel('Weight')
        plt.plot(weights_epoch_1_1_0, 'bs', weights_epoch_1_1_1, 'b--', weights_epoch_1_1_2, 'b^',
                 weights_epoch_1_2_0, 'rs', weights_epoch_1_2_1, 'r--', weights_epoch_1_2_2, 'r^',
                 weights_epoch_2_1_0, 'gs', weights_epoch_2_1_1, 'g--', weights_epoch_2_1_2, 'g^', ms=1)

        plt.subplot(212)
        plt.xlabel('Epoch*100')
        plt.ylabel('Error')
        plt.plot(errors_epoch_1_1, 'b', errors_epoch_1_2, 'r', errors_epoch_2_1, 'g', ms=1)
        plt.show()

    def __testing_loop(self):
        # testing loop
        if self.params['testing']:
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
                if self.params['validating'] and self.params['best_weights']:
                    outputs_l_j = mlp_functions.forward_pass(
                        self.params, self.neurons_l, self.best_weights_l_i_j,
                        outputs_l_j)
                # use weight at lowest training error
                else:
                    outputs_l_j = mlp_functions.forward_pass(
                        self.params, self.neurons_l, self.weights_l_i_j,
                        outputs_l_j)

                print(outputs_l_j[len(self.neurons_l) - 1])

                # update test error
                output_pattern = p[self.params['input_dimensions']:
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
