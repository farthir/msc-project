# import mlp_functions
# import config
import math
import json
import mlp_functions


class Multilayer_perceptron(object):
    def __init__(self, parameters_filename, data_filename):
        self.__read_parameters(parameters_filename)
        self.__read_data(data_filename)
        self.__initialise_network()
        self.__backpropagation_loop()
        self.__testing_loop()

    def __read_parameters(self, parameters_filename):
        with open('parameters/%s.json'
                  % parameters_filename) as parameters_file:
            self.params = json.load(parameters_file)

    def __read_data(self, data_filename):
        patterns = mlp_functions.read_patterns('data/%s.csv' % data_filename)

        # read in test patterns
        if (self.params['testing']):
            self.test_patterns = mlp_functions.read_patterns('data/%s_test.csv'
                                                             % data_filename)

        if (self.params['validating'] and
                self.params['training_validation_ratio'] == 1):
            self.training_patterns = patterns
            self.validation_patterns = patterns

        elif (self.params['validating'] and
                self.params['training_validation_ratio'] < 1):
            num_training_patterns = int(len(patterns) *
                                        self.params['training_validation_ratio'])
            self.training_patterns = patterns[:num_training_patterns]
            self.validation_patterns = patterns[num_training_patterns:]

        elif not self.params['validating']:
            self.training_patterns = patterns

    def __initialise_network(self):
        # network initialisation
        # set number of neurons in each layer
        # layer '0' is the input layer and defines number of inputs
        neurons_l = []
        neurons_l.append(self.params['input_dimensions'])

        if self.params['neuron_layers'] > 1:
            for l in range(self.params['neuron_layers'] - 1):
                neurons_l.append(self.params['hidden_nodes'][l])

        # this may not always be the case but is set here
        neurons_l.append(self.params['output_dimensions'])

        # initialise weights
        weights_l_i_j = mlp_functions.initialise_weights(neurons_l)

        self.neurons_l = neurons_l
        self.weights_l_i_j = weights_l_i_j

    def __backpropagation_loop(self):
        # backpropagation loop
        epoch = 0
        training_errors = []
        repeat = True
        if self.params['validating']:
            validation_error_best = 1000.0
            validation_errors = []

        while (repeat):
            training_error = 0.0

            for p in self.training_patterns:
                # load pattern
                input_pattern = p[:self.params['input_dimensions']]
                # set bias 'output'
                outputs_l_j = mlp_functions.initialise_bias(
                    self.params['neuron_layers'])
                # add input pattern to 'output' of layer 0 (i.e. set the input to p)
                outputs_l_j[0].extend(input_pattern)

                # forward pass
                outputs_l_j = mlp_functions.forward_pass(self.neurons_l, self.weights_l_i_j,
                                                         outputs_l_j)

                # update training_error
                output_pattern = p[self.params['input_dimensions']:]
                teacher_i = []
                # account for i = 0
                teacher_i.append(None)
                teacher_i.extend(output_pattern)

                training_error = mlp_functions.update_ms_error(
                    self.neurons_l, training_error, teacher_i, outputs_l_j)

                # calculate errors
                errors_l_i = mlp_functions.calculate_errors(self.neurons_l, self.weights_l_i_j,
                                                            teacher_i, outputs_l_j)

                # update weights
                self.weights_l_i_j = mlp_functions.update_weights(self.neurons_l, self.weights_l_i_j,
                                                                  errors_l_i, outputs_l_j)

            # normalise mean squared training error into [0,1] and convert to rms
            training_error = math.sqrt(training_error / (self.neurons_l[-1] *
                                                         len(self.training_patterns)))

            # write out epoch training_error
            training_errors.append(training_error)

            if self.params['validating']:
                validation_error = 0.0

                for p in self.validation_patterns:
                    # load pattern
                    input_pattern = p[:self.params['input_dimensions']]
                    # set bias 'output'
                    outputs_l_j = mlp_functions.initialise_bias(
                        self.params['neuron_layers'])
                    # add input pattern to 'output' of layer 0 (i.e. set the input to p
                    outputs_l_j[0].extend(input_pattern)

                    # forward pass
                    outputs_l_j = mlp_functions.forward_pass(self.neurons_l, self.weights_l_i_j,
                                                             outputs_l_j)

                    # update validation error
                    output_pattern = p[self.params['input_dimensions']:]
                    teacher_i = []
                    # account for i = 0
                    teacher_i.append(None)
                    teacher_i.extend(output_pattern)
                    validation_error = mlp_functions.update_ms_error(
                        self.neurons_l, validation_error, teacher_i, outputs_l_j)

                # normalise mean squared validation error into [0,1] and convert to rms
                validation_error = math.sqrt(validation_error / (self.neurons_l[-1] * (
                    len(self.validation_patterns))))

                # make sure validation error is dropping
                if (validation_error < validation_error_best):
                    validation_error_best = validation_error
                    best_weights_l_i_j = list(self.weights_l_i_j)

                validation_errors.append(validation_error)

            epoch += 1
            if (training_error < self.params['target_training_error'] or
                    epoch == self.params['max_epochs']):
                repeat = False

        if validation_error_best < self.params['target_validation_error']:
            success = True
        else:
            success = False

        self.best_weights_l_i_j = best_weights_l_i_j
        self.training_errors = training_errors
        self.validation_errors = validation_errors
        self.success = success

    def __testing_loop(self):
        # testing loop
        if (self.params['testing']):
            testing_error = 0.0
            testing_errors = []

            for p in self.test_patterns:
                # load pattern
                input_pattern = p[:self.params['input_dimensions']]
                # set bias 'output'
                outputs_l_j = mlp_functions.initialise_bias(
                    self.params['neuron_layers'])
                # add input pattern to 'output' of layer 0 (i.e. set the input to p)
                outputs_l_j[0].extend(input_pattern)

                # forward pass
                # use weight at lowest validation error
                if (self.params['validating'] and self.params['best_weights']):
                    outputs_l_j = mlp_functions.forward_pass(self.neurons_l,
                                                             self.best_weights_l_i_j,
                                                             outputs_l_j)
                # use weight at lowest training error
                else:
                    outputs_l_j = mlp_functions.forward_pass(self.neurons_l,
                                                             self.weights_l_i_j,
                                                             outputs_l_j)

                # update test error
                output_pattern = p[self.params['input_dimensions']:]
                teacher_i = []
                # account for i = 0
                teacher_i.append(None)
                teacher_i.extend(output_pattern)

                testing_error = mlp_functions.update_ms_error(
                    self.neurons_l, testing_error, teacher_i, outputs_l_j)
                testing_errors.append(testing_error)

            # normalise mean squared testing error into [0,1] and convert to rms
            testing_error = math.sqrt(testing_error / (self.neurons_l[-1] *
                                                       len(self.test_patterns)))
        self.testing_errors = testing_errors
