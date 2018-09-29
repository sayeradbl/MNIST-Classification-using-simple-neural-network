import numpy as np
from urllib import request
import gzip
import pickle
import os

class MNIST:
    def __init__(self):
        self.filename = [
        ["X_train","train-images-idx3-ubyte.gz"],
        ["X_test","t10k-images-idx3-ubyte.gz"],
        ["y_train","train-labels-idx1-ubyte.gz"],
        ["y_test","t10k-labels-idx1-ubyte.gz"]
        ]
        self.data_path = 'data/'

    def download_mnist(self):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in self.filename:
            if  os.path.isfile(self.data_path+name[1]):
                print('Dataset already saved. Skipped downloading {}'.format(name[1]))
                continue
            print("Downloading "+name[1]+"...")
            request.urlretrieve(base_url+name[1], self.data_path+name[1])
        print("Download complete.")

    def save_mnist(self):
        mnist = {}
        for name in self.filename[:2]:
            with gzip.open(self.data_path+name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
        for name in self.filename[-2:]:
            with gzip.open(self.data_path+name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(self.data_path+"mnist.pkl", 'wb') as f:
            pickle.dump(mnist,f)
        print("Save complete.")

    def init(self):
        self.download_mnist()
        self.save_mnist()

    def load(self):
        with open(self.data_path+"mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        return mnist["X_train"], mnist["y_train"], mnist["X_test"], mnist["y_test"]

# if __name__ == '__main__':
#     MNIST().init()
