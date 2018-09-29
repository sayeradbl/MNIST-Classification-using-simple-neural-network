import numpy as np
import random

class NeuralNetwork:

    def __init__(self, topology, learning_rate = 0.01, momentum = 0, reg_param = 0.1, batch_size = 100, epochs=100):
        self.topology = topology
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.reg_param = reg_param
        self.batch_size = batch_size
        self.epochs = epochs

        #randomly initialize the weights and biases
        self.weights = [2*np.random.random(size).astype('f') - 1 for size in zip(topology[:-1], topology[1:])]    #get the weights of size - consecutive pairs of the topology
        self.biases = [2*np.random.random((1, size)).astype('f') - 1  for size in topology[1:]]    #one bias per neuron except the input neuron

        #variables to keep track of the gradients
        self.delta_weights = [np.zeros(weight.shape).astype('f') for weight in self.weights]
        self.delta_biases = [np.zeros(bias.shape).astype('f') for bias in self.biases]

    def softmax(self, x):
#         Compute the softmax of vector x
#         return np.exp(x) / np.sum(np.exp(x), axis = 1).reshape((-1,1))
        e = np.exp(x - np.max(x))
        return e / np.sum(e, axis = 1).reshape((-1,1))

    def softmax_der(self, x):
        y = self.softmax_generalized(x)
        return y * (1-y)

    def softmax_generalized(self, x, theta = 1.0, axis = 1):
        """
            With axis=1, apply softmax for each row
        """
        # make X at least 2d
        y = np.atleast_2d(x)

        # find axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

        y = y * float(theta)

        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis = axis), axis)
        y = np.exp(y)

        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

        p = y / ax_sum

        # flatten if X was 1D
        if len(x.shape) == 1: p = p.flatten()

        return p


    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def sigmoid_der(self, x):
        sig = self.sigmoid(x)
        return sig * (1-sig)

    def tanh_der(self, x):
        return (1 - (np.tanh(x))**2)

    def cross_entropy(self, target, predicted):
        #Cross Entropy  Loss
#         return -1/len(target) * np.sum(target * np.log(predicted) + (1-target) * np.log( 1 - predicted))
        return -1/len(target) * np.sum(target*np.log(predicted), axis = 1)


    def cross_entropy_der(self, target, predicted):
        return -np.sum(target/predicted, axis=1)
#         return - (target - predicted) / ( predicted * (1-predicted) )


    def forward(self, x):
        #forward pass through all the layers
        #returns the final output along with the activated output in each layer

        cache_z = []    #variable to store the values of inputs after multiplication with each of the weights
        cache_h = []    #variable to store the values after activation at each layer

        ip = x    #variable to store the instant input to different layers

#         cache_h.append(np.array(ip))    #adding the input to the list of output from previous hidden layer

        #process for hidden layers
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(ip, weight) + bias
            cache_z.append(z)
            h = np.tanh(z)
            cache_h.append(h)
            ip = h

        y = np.dot(ip, self.weights[-1]) + self.biases[-1]
        cache_z.append(y)
        softmax_out = self.softmax(y)
        cache_h.append(softmax_out)

        return softmax_out, cache_z, cache_h

    def backpropagate(self, y_train, cache_z, cache_h):
        '''from homework1 and other resources

        At outer layer:
            dJ/dWl = ( delta = error * softmax_prime(zl) )
                    * (previous layer activation (hl-1) )
        At inner layer:
            dJ/dWl-1 = ( delta = delta * tanh_prime(zl-1) * Wl)
                        * (previous layer activation (hl-1))

        '''

        grad_weights = [np.zeros(weight.shape).astype('f') for weight in self.weights]
        grad_biases = [np.zeros(bias.shape).astype('f') for bias in self.biases]

        output = cache_h[-1]
        cost = self.cross_entropy(y_train, output)

#         errors = self.cross_entropy_der(y_train, output)
#         delta = errors * self.softmax_der(cache_z[-1])

        delta = output - y_train         #from hw1 and https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function

        grad_weights[-1] = np.dot(cache_h[-2].T, delta)
        grad_biases[-1] = delta

        for i in range(2, len(self.topology)):
            z = cache_z[-i]
            der = self.tanh_der(z)
            delta = np.dot(delta, self.weights[-i+1].T)*der
            grad_biases[-i] = delta
            grad_weights[-i] = np.dot(cache_h[-i-1].T, delta)

        return grad_weights, grad_biases, cost


    def train(self, x_train_whole, y_train_whole, batch_size=100):
        size = len(x_train_whole)
        costs = []
        for j in range(self.epochs):
            costs_epoch = []
            print("Epoch: {}".format(j))
            dataset = list(zip(x_train_whole, y_train_whole))
            np.random.shuffle(dataset)
            x_train_whole, y_train_whole = zip(*dataset)
            x_train_whole = np.array(x_train_whole)
            y_train_whole = np.array(y_train_whole)

            for i, k in enumerate(range(0, size, batch_size)):
                cost =  self.train_mini_batch(x_train_whole[k : k + batch_size],y_train_whole[k : k + batch_size])
                costs_epoch.append(cost)
#             print("{}: {}".format(j, costs_epoch))
            cost = np.mean(costs_epoch)
            costs.append(cost)
        return costs

    def train_mini_batch(self, x_train, y_train):

        n = len(x_train)

        mu = self.momentum
        lr = self.learning_rate

        predicted, cache_z, cache_h = self.forward(x_train)
        cache_h = [x_train] + cache_h
        grad_weights, grad_biases, cost = self.backpropagate(y_train, cache_z, cache_h)

        '''
            v_prev = v
            v = mu * v_prev - lr * grad    --> standard momentum
            w = w + v

            v_prev = v
            v = mu * v - lr * grad
            w += - mu * v_prev + (1 + mu) * v    --> Neterov momentum
        '''

        for i, weight in enumerate(self.weights):
#              Storing previous weights for Neterov's momentum
            delta_weights_prev = self.delta_weights[i][:]
            delta_biases_prev = self.delta_biases[i][:]

#              Neterov's momentum
            self.delta_weights[i] =  mu * self.delta_weights[i] -  1/n * lr * grad_weights[i]
            self.delta_biases[i] =  mu * self.delta_biases[i] -  1/n * lr * np.sum(grad_biases[i], axis=0)

            self.weights[i] = (1 - lr*self.reg_param/n)*weight + (1+mu) * self.delta_weights[i] - mu * delta_weights_prev
            self.biases[i] = self.biases[i] + (1+mu) * self.delta_biases[i] - mu * delta_biases_prev

#         Regular stochastic gradient descent
#         for i, weight in enumerate(self.weights):
#             self.delta_weights[i] = lr * grad_weights[i]
#             self.delta_biases[i] = lr * np.sum(grad_biases[i], axis=0)
#             self.weights[i] -= self.delta_weights[i]
#             self.biases[i] -= self.delta_biases[i]

        return cost

    def predict(self, x):
        predicted, cache_z, cache_h = self.forward(x)
        return predicted
#         return np.argmax(predicted, axis = 1)
