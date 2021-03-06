"""Entry module for application"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import progressbar

from mlp_network import MLPNetwork as Mlp

def main():
    """Entry function for application to prevent execution if imported as module"""
    results_file = Path('results/%s.csv' % sys.argv[2])
    if results_file.is_file():
        print('ERROR: results file already exists')
    else:
        number_networks = int(sys.argv[3])

        # initialise progress bar for console
        progress_bar = progressbar.ProgressBar(
            maxval=number_networks,
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        progress_bar.start()

        # build and train networks and determine success rate based on target_training_error
        network_training_errors = []

        # success parameters
        number_successes = 0
        epochs_success = 0
        average_epochs_success = 0
        training_error_success = 0
        average_training_error_success = 0
        testing_error_success = 0
        average_testing_error_success = 0

        for network_index in range(number_networks):
            progress_bar.update(network_index)
            print()
            network = Mlp(sys.argv[1], sys.argv[2])

            network_training_errors.append(network.training_error_end)

            # stats for successful networks only
            if network.training_error_end < network.params['target_training_error']:
                number_successes += 1
                epochs_success += network.epoch_target_training_error
                training_error_success += network.training_error_end
                if network.params['testing']:
                    testing_error_success += network.average_testing_error

        percentage_successful = (number_successes / number_networks) * 100
        if number_successes > 0:
            average_epochs_success = epochs_success / number_successes
            average_training_error_success = training_error_success / number_successes
            if network.params['testing']:
                average_testing_error_success = testing_error_success / number_successes

        # print out vital network stats
        print()
        print('Network architecture: ', network.neurons_l)
        print('Percentage reaching target training error: ', percentage_successful)
        print('Average epochs to reach target training error: ', average_epochs_success)
        print('Average training error at end of all epochs: ', average_training_error_success)
        if network.params['testing']:
            print('Average testing error: ', average_testing_error_success)

        # visualise training progress
        if sys.argv[4] == '1':
            plt.subplot(311)
            plt.title('End training error for all networks')
            plt.xlabel('Error')
            plt.ylabel('Number')
            plt.hist(network_training_errors, 10)

            plt.subplot(312)
            plt.title('Training error progress for final network')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            if network.params['validating']:
                plt.plot(network.training_errors, 'b', network.validation_errors, 'r--')
            else:
                plt.plot(network.training_errors, 'b')

            if network.params['testing']:
                plt.subplot(313)
                plt.title('Test errors for final network')
                plt.xlabel('Error')
                plt.ylabel('Number')
                plt.hist(network.testing_errors, 10)

            plt.ylim(ymin=0)
            plt.tight_layout()
            plt.savefig('results/%s.svg' % sys.argv[2])
            plt.show()

if __name__ == "__main__":
    main()
