import sys
import matplotlib.pyplot as plt
import progressbar

from mlp_network import Multilayer_perceptron as Mlp

successes = 0
nets = int(sys.argv[4])

# initialise progress bar for console
bar = progressbar.ProgressBar(
    maxval=nets,
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

for t in range(nets):
    bar.update(t + 1)
    net = Mlp(sys.argv[1], sys.argv[2], sys.argv[3])

    if net.training_error_best < net.params['target_training_error']:
        successes += 1

percentage_successful = (successes / nets) * 100
print()
print('Percentage successful: ', percentage_successful)

if (sys.argv[5] == '1'):
    print(net.testing_errors)

    plt.subplot(211)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.plot(net.training_errors, 'b', net.validation_errors, 'r--')

    plt.subplot(212)
    plt.xlabel('Error')
    plt.ylabel('Number')

    plt.hist(net.testing_errors, 10, (0, 1))
    plt.show()
