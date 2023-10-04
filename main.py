import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import sys
from PIL import Image, ImageOps
from drawer import start
import time

# np.set_printoptions(threshold=sys.maxsize)

data = pd.read_csv('letters.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_, m_train = X_train.shape


def init_params():
    W1 = np.random.rand(26, 784) - 0.5
    b1 = np.random.rand(26, 1) - 0.5
    W2 = np.random.rand(26, 26) - 0.5
    b2 = np.random.rand(26, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    print(np.sum(A),1)
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    t0=time.perf_counter()
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        tx = time.perf_counter()
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
            ti=round(time.perf_counter() - tx)
            print(f"Expected time left: {ti*iterations - ti*i} seconds({round((ti*iterations - ti*i)/60)} minutes)")
    print("Iteration: ", i+1)
    predictions = get_predictions(A2)
    print(get_accuracy(predictions, Y))
    print(f"Time taken: {round(time.perf_counter()-t0)} seconds")
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 10)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U","V", "W", "X", "Y", "Z"]
    predictions = []
    for i in range(3):
        max = get_predictions(A2)
        predictions.append((A2[max], alphabet[int(max)]))
        A2[max] = 0
    print(predictions)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    current_image = X_train[:, index, None]
    make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Label: ", alphabet[label])

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def resize():
    img = Image.open("canvas.png")
    img = ImageOps.grayscale(img)
    newsize = (28, 28)
    img = img.resize(newsize)
    img.save("canvas.png")
    numpydata = np.asarray(img)
    numpydata=np.resize(numpydata, (784, 1))
    numpydata = numpydata / 255.
    return numpydata

def test_prediction_pic(X, W1, b1, W2, b2):
    make_predictions(X, W1, b1, W2, b2)

    current_image = X.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

#Use from data set
# while True:
#     test_prediction(random.randint(0,371449), W1, b1, W2, b2)

# Use own drawing
while True:
    start()
    test_prediction_pic(resize(), W1, b1, W2, b2)


