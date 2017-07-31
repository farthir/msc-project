from keras.models import Sequential
from keras.layers import Dense, Dropout
import sys
import numpy as np

def main():
    input_filename = sys.argv[1]

    training = np.loadtxt('data/%s.csv' % input_filename, delimiter=',')
    test = np.loadtxt('data/%s_test.csv' % input_filename, delimiter=',')
    y_train = training[:,3:4]
    x_train = training[:,0:3]
    y_test = test[:,3:4]
    x_test = test[:,0:3]

    model = Sequential()
    model.add(Dense(10, activation='tanh', input_dim=3))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100)

    print('Test score: ', model.evaluate(x_test, y_test))
    y_network = model.predict(x_test)

    out = np.concatenate((x_test, y_test, y_network), axis=1)
    np.savetxt('results/%s_kera.csv' % input_filename, out, delimiter=',')

if __name__ == "__main__":
    main()
