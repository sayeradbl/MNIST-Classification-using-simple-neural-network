import matplotlib.pyplot as plt
import numpy as np
from prepare_mnist import MNIST
from neuralnetwork import NeuralNetwork


def calculate_accuracy(target_label, predicted_label):
        matched = np.sum(target_label == predicted_label)
        return matched / len(target_label)


#Since the labels are not probabilities as we get from the softmax
    #Convert each label as an array of zeros with '1' at the particular label value
def one_hot_encode(data):
    zeros = np.zeros((len(data), 10))
    zeros[np.arange(len(zeros)), data] += 1
    return zeros


if __name__ == "__main__":
    #Downloading and loading the MNIST dataset
    mnist = MNIST()
    mnist.init()
    x_train, Y_train, x_test, Y_test = mnist.load()

    #Shuffling the dataset before training
    dataset = list(zip(x_train, Y_train))
    np.random.shuffle(dataset)

    x_train, Y_train = zip(*dataset)
    x_train = np.array(x_train)
    Y_train = np.array(Y_train)

    x_train = x_train/255
    x_test = x_test/255


    #For training labels
    y_train = one_hot_encode(Y_train)

    #For testing labels
    y_test = one_hot_encode(Y_test)


     #Create and train the model
    topology = [x_train[0].shape[0], 100, 30, 10]
    NN = NeuralNetwork(topology, learning_rate=0.01, momentum=0.8, batch_size= 100, epochs = 20)
    print("Training Network...")
    train_costs = NN.train(x_train, y_train)
    print("Training costs: {}".format(train_costs))

    print("\nMaking predictions...\n")
    predict = NN.predict(x_test)
    predicted_labels = np.argmax(predict, axis=1)

    test_accuracy = calculate_accuracy(Y_test, predicted_labels)
    print("Test Accuracy: {}%".format(test_accuracy*100))

    plt.plot(train_costs)
    plt.xlabel("Epochs")
    plt.ylabel("Costs (NLL)")
    plt.title("Training costs")
    plt.show()

    
