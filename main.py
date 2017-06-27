import sys
import matplotlib.pyplot as plt
import progressbar

from mlp_network import Multilayer_perceptron as Mlp

net = Mlp(sys.argv[1], sys.argv[2])

successes = 0
nets = int(sys.argv[3])

bar = progressbar.ProgressBar(maxval=nets, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for t in range(nets):
    bar.update(t + 1)
    net = Mlp(sys.argv[1], sys.argv[2])
    if net.success:
        successes += 1

percentage_successful = (successes / nets) * 100
print()
print('Percentage successful: ', percentage_successful)

if (sys.argv[4] == '1'):
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
