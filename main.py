import sys
import matplotlib.pyplot as plt
import progressbar

from mlp_network import Multilayer_perceptron as Mlp

nets = int(sys.argv[4])

# initialise progress bar for console
bar = progressbar.ProgressBar(
    maxval=nets,
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

successes = 0
epochs_success = 0
average_epochs_success = 0
for t in range(nets):
    bar.update(t + 1)
    net = Mlp(sys.argv[1], sys.argv[2], sys.argv[3])

    if net.params['validating']:
        if net.validation_error_best < net.params['target_validation_error']:
            successes += 1
            epochs_success += net.epoch_best
    else:
        if net.training_error_end < net.params['target_training_error']:
            successes += 1
            epochs_success += net.epoch_end

percentage_successful = (successes / nets) * 100
if successes > 0:
    average_epochs_success = epochs_success / successes
print()
print('Percentage successful: ', percentage_successful)
print('Average epochs (successful): ', average_epochs_success)

if (sys.argv[5] == '1'):
    if net.params['testing']:
        print('Testing errors: ')
        print(net.testing_errors)

        plt.subplot(212)
        plt.xlabel('Error')
        plt.ylabel('Number')
        plt.hist(net.testing_errors, 10, (0, 1))

    plt.subplot(211)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    if net.params['validating']:
        plt.plot(net.training_errors, 'b', net.validation_errors, 'r--')
    else:
        plt.plot(net.training_errors, 'b')

    plt.show()
