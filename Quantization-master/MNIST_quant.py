import keras
import numpy as np
import copy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.utils import print_summary, to_categorical

from keras.datasets import mnist

# the data, split between train and test sets
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255
y_test = keras.utils.to_categorical(y_test, num_classes)
print(x_test.shape[0], 'test samples')

# num_classes = 10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255.0
# x_test /= 255.0

total_influence = 0
num_weight = 0
max_weight = 0
min_weight = 0

def cal_influence(weights, num_neighbors):
    global total_influence
    n = len(weights)
    influence = copy.deepcopy(weights)
    newlist = []
    for i in range(n):
        newlist = np.concatenate((newlist, influence[i].flatten()))
    sortIdx = np.argsort(newlist)
    newlist = sorted(newlist)
    sizeList = []
    sizeList.append(0)
    for i in range(1, n + 1):
        sizeList.append(sizeList[i - 1] + np.size(influence[i - 1], 0) * np.size(influence[i - 1], 1))

    m = len(newlist)
    for j in range(m):
        neighbor_sum = 0
        neg_num = 0
        pos_num = 0
        neighbor_list = []

        for k in range(1, num_neighbors + 1):
            if j - k >= 0:
                neighbor_list.append(abs(newlist[j - k] - newlist[j]))
            if j + k < m:
                neighbor_list.append(abs(newlist[j + k] - newlist[j]))

        neighbor_list = sorted(neighbor_list)
        neighbor_list = neighbor_list[0:num_neighbors]
        neighbor_sum = sum(neighbor_list)
        neighbor_sum /= num_neighbors
        this_influence = newlist[j] ** 2 / neighbor_sum
        total_influence += this_influence
        layer = 0
        for i in range(1, len(sizeList)):
            if sortIdx[j] < sizeList[i]:
                layer = i - 1
                break
        index_on_layer = sortIdx[j] - sizeList[layer]
        influence[layer][index_on_layer // np.size(influence[layer][0], 0)][index_on_layer % np.size(influence[layer][0], 0)] = this_influence
    return influence

def allocate_bits(influence, total_bits):
    global total_influence
    for i in range(len(influence)):
        for x in range(len(influence[i])):
            for y in range(len(influence[i][0])):
                influence[i][x][y] = total_bits * influence[i][x][y] // total_influence + 2
    return influence


def quant(weights, allocation):
    global max_abs
    n = len(weights)
    new_weights = np.copy(weights)
    for i in range(n):
        for x in range(len(weights[i])):
            for y in range(len(weights[i][0])):
                scale = (2 ** (allocation[i][x][y] - 1) - 1) / max_abs
                new_weights[i][x][y] = weights[i][x][y] * scale
                new_weights[i][x][y] = np.around(new_weights[i][x][y])
                new_weights[i][x][y] = new_weights[i][x][y] / scale
    return new_weights

model = load_model('my_model_nobias.h5')
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

weights = model.get_weights()
print(np.shape(weights[0]))
for i in range(len(weights)):
    num_weight += len(weights[i]) * len(weights[i][0])
    max_weight = np.max(weights[i])
    min_weight = np.min(weights[i])
    max_abs = max(abs(max_weight), abs(min_weight))
influence = cal_influence(weights, 10)
allocation = allocate_bits(influence, 5 * num_weight)
quantilized = quant(weights, allocation)
model.set_weights(quantilized)

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])