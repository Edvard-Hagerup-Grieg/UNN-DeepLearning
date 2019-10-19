from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
import time


class NeuralNetwork(object):

    def __init__(self, N=784, K=50, M=10):
        self.w1 = np.random.randn(N, K) / 10.0
        self.bias1 = np.zeros((K))

        self.w2 = np.random.randn(K, M) / 10.0
        self.bias2 = np.zeros((M))
        pass


    def phi_1(self, X):
        return X * (X > 0.0)

    def phi_2(self, X):

        new_X = []
        for i in np.arange(X.shape[0]):
            x = X[i,:]
            max_x = np.max(x)

            exp_x = np.exp(x - max_x)
            sum_exp = np.sum(exp_x)
            exp_x /= sum_exp

            new_X.append(exp_x)

        return np.array(new_X)

    def dphi_1(self, X):
        return np.ones(X.shape) * (X > 0.0)

    def loss(self, u, y):
        u_ = np.where(u == 0, 1, u)
        return -np.sum(y * np.log(u_))


    def predict(self, X):

        y = self.phi_2(np.dot(self.phi_1(np.dot(X, self.w1) + self.bias1), self.w2) + self.bias2)

        return np.array(y)

    def get_metrics(self, u, y):
        batch_size = y.shape[0]

        acc = 0.0
        err = 0.0
        for i in np.arange(batch_size):
            if np.argmax(u[i, :]) == np.argmax(y[i, :]):
                acc += 1.0

            err += self.loss(u[i, :], y[i, :])

        return err / batch_size, acc / batch_size


    def fit(self, X, Y, x_test, y_test, batch_size=1, iter=100, eta=0.01, validation_split=0.0):

        #x_train = X[:X.shape[0](1.0 - validation_split), :]

        for it in np.arange(iter):
            print("\nepoch %s/%s" % (it + 1, iter))

            Acc = []
            Err = []
            for l in np.arange(batch_size, X.shape[0], batch_size):
                y = Y[l - batch_size:l, :]
                x = X[l - batch_size:l, :]

                # forward pass
                v = np.dot(x,self.w1) + self.bias1
                v = self.phi_1(v)

                u = np.dot(v, self.w2) + self.bias2
                u = self.phi_2(u)

                err, acc = self.get_metrics(u, y)
                Acc.append(acc)
                Err.append(err)

                print("\rbatch %s/%s" % (l, X.shape[0]), end='')


                # reverse pass
                uy = u - y

                dE_2 = np.dot(v.T,uy) / batch_size
                dE_1 = np.dot(np.dot(self.w2, uy.T) * self.dphi_1(v).T,x) / batch_size

                dE_bias2 = np.sum(uy, axis=0) / batch_size
                dE_bias1 = np.sum(np.dot(self.w2, uy.T) * self.dphi_1(v).T, axis=1).T / batch_size

                # update weight
                self.w1 -= eta * dE_1.T
                self.w2 -= eta * dE_2

                self.bias1 -= eta * dE_bias1
                self.bias2 -= eta * dE_bias2


            print("\nTrain: err = %f\tacc = %f" % (np.sum(Err) * batch_size / X.shape[0],
                                                   np.sum(Acc) * batch_size / X.shape[0]), end='\n')

            y_predict = self.predict(x_test)
            err, acc = self.get_metrics(y_predict, y_test)

            print("Test: err = %f, acc = %f" % (err, acc))


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(x_train.shape)
    print(y_train.shape)

    model = NeuralNetwork()
    start_time = time.time()
    model.fit(X=x_train, Y=y_train, x_test=x_test, y_test=y_test, iter=20, batch_size=100, eta=0.1)
    finish_time = time.time()
    print("Time: %s" % (finish_time - start_time))