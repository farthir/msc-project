from keras.models import Sequential
from keras.layers import Dense, Dropout
import sys
import numpy as np

def main():
    input_filename = sys.argv[1]
    num_networks = int(sys.argv[2])

    training = np.loadtxt('data/%s.csv' % input_filename, delimiter=',')
    test = np.loadtxt('data/%s_test.csv' % input_filename, delimiter=',')
    y_train = training[:, 3:4]
    x_train = training[:, 0:3]
    y_test = test[:, 3:4]
    x_test = test[:, 0:3]

    test_score = 0
    result = np.zeros((1,5))

    for _ in range(num_networks):
        model = Sequential()
        model.add(Dense(10, activation='tanh', input_dim=3))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=20, shuffle=True)
        y_network = model.predict_on_batch(x_test)
        result = np.concatenate((result, np.concatenate((x_test, y_test, y_network), axis=1)), axis=0)
        test_score += model.evaluate(x_test, y_test)

    print()
    print('Test score: ', test_score / num_networks)

    result = np.delete(result, 0, 0)
    np.savetxt('results/%s_kera.csv' % input_filename, result, delimiter=',')

if __name__ == "__main__":
    main()
