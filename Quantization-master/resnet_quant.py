import tensorflow.keras as keras
import numpy as np
import copy
# from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.utils import print_summary, to_categorical
import sys
import math

from keras.datasets import cifar10

# the data, split between train and test sets
num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32')
y_test = keras.utils.to_categorical(y_test, num_classes)

# total_influence = 0
num_weight = 0
max_weight = 0
min_weight = 0
total_influence_list = []
layer_size_list = []
quant_list = []

def cal_influence(weights, num_neighbors):
    global total_influence_list
    global layer_size_list
    global quant_list
    # global max_abs_list
    # global max_weight_list
    # global min_weight_list

    n = len(weights)
    weights = copy.deepcopy(weights)
    influence_list = []

    for i in range(len(weights)):  # for each layer
        if len(np.shape(weights[i])) == 1:  # skip layers with 1-dimesnion weights
            continue
        
        # to 1-dimension
        sort_idx = np.argsort(weights[i].flatten())
        sorted_layer_weight = np.sort(weights[i].flatten())
        print(i)

        layer_total_influence = 0
        layer_influence = np.zeros(weights[i].flatten().shape)

        for j in range(sorted_layer_weight.size):  # for each weight in the layer
            # calcuate sum of neighbors
            neighbor_sum = 0
            neighbor_list = []

            for k in range(1, num_neighbors + 1):
                if j - k >= 0:
                    neighbor_list.append(abs(sorted_layer_weight[j - k] - sorted_layer_weight[j]))
                if j + k < sorted_layer_weight.size:
                    neighbor_list.append(abs(sorted_layer_weight[j + k] - sorted_layer_weight[j]))

            neighbor_list = sorted(neighbor_list)
            neighbor_list = neighbor_list[0:num_neighbors]
            neighbor_sum = sum(neighbor_list)
            neighbor_sum /= num_neighbors
            
            # calcuate influence
            this_influence = (abs(sorted_layer_weight[j]) / neighbor_sum) ** 0.5

            layer_total_influence += this_influence
            layer_influence[sort_idx[j]] = this_influence

            quant_list.append(i)
        
        total_influence_list.append(layer_total_influence)
        influence_list.append(layer_influence)

    # rehshape
    influence = []
    quant_idx = 0
    # count_size = 0
    for i in range(len(weights)):
        layer_shape = np.shape(weights[i])
        if len(layer_shape) == 1:
            continue
        else:
            layer_size = 1
            for size in layer_shape:
                layer_size *= size

            # layer_list = influence_list[count_size : count_size + layer_size]
            # layer_list = influence_list[round(i/2)]
            layer_list = influence_list[quant_idx]
            # count_size += layer_size
            layer_weights = np.reshape(np.array(layer_list), layer_shape)
            influence.append(layer_weights)
            layer_size_list.append(layer_size)
            quant_idx += 1

    return influence

def allocate_bits(influence, bits_per_weight):
    global total_influence_list
    for i in range(len(influence)):  # for each non-1-dimension layer
        layer_total_influence = total_influence_list[i]
        layer_total_bits = bits_per_weight * layer_size_list[i]
        layer_bits_allo = []
        for influ in influence[i].flatten():  # for each weight influence in the layer
            bit = round(layer_total_bits * influ / layer_total_influence) + 2
            if (bit > 32):
                print(bit)
                bit = 32
            layer_bits_allo.append(bit)
        influence[i] = np.reshape(np.array(layer_bits_allo), np.shape(influence[i]))
    return influence


def uniform_allocate_bits(influence, bits_per_weight):
    global total_influence_list
    for i in range(len(influence)):  # for each non-1-dimension layer
        layer_bits_allo = []
        # for each weight influence in the layer
        for j in range(influence[i].flatten().size): 
            layer_bits_allo.append(bits_per_weight + 2)
        influence[i] = np.reshape(np.array(layer_bits_allo), np.shape(influence[i]))
    return influence

def clamp(realnum, a, b):
    if realnum < a:
        return a
    elif realnum > b:
        return b
    else:
        return realnum

def quant(weights, allocation, range_list):
    # global max_weight_all
    # global min_weight_all
    # global max_abs
    # global max_abs_all
    # global min_abs_all

    n = len(weights)
    new_weights = []
    quant_idx = 0
    for i in range(n):
        if len(np.shape(weights[i])) == 1: 
            new_weights.append(weights[i])
        else:
            # layer_range = range_list[round(i/2)]
            # layer_allo = allocation[round(i/2)].flatten()
            layer_range = range_list[quant_idx]
            layer_allo = allocation[quant_idx].flatten()
            layer_weights = weights[i].flatten()
            for j in range(layer_weights.size):
                scale = (layer_range) / (2 ** (layer_allo[j] - 1))
                layer_weights[j] = clamp(layer_weights[j], -layer_range, layer_range)
                layer_weights[j] = np.around((layer_weights[j] - layer_range) / scale)
                layer_weights[j] = layer_weights[j] * scale + layer_range

                # scale = layer_range / (2 ** (layer_allo[j] - 1))
                # layer_weights[j] = clamp(layer_weights[j], -layer_range, layer_range)
                # layer_weights[j] = np.around(layer_weights[j] / scale)
                # layer_weights[j] = layer_weights[j] * scale

                # scale = (2 ** (layer_allo[j] - 1) - 1) / max_abs
                # layer_weights[j] = layer_weights[j] * scale
                # layer_weights[j] = np.around(layer_weights[j])
                # layer_weights[j] = layer_weights[j] / scale
            layer_weights = np.reshape(layer_weights, np.shape(weights[i]))
            new_weights.append(layer_weights)
            quant_idx += 1
            print(i)

    return new_weights

# AVE_BITS_PER_WEIGHT = 8

# model = load_model('models/resnet-cifar10/maxmodel.h5')

# test_loss, test_score = model.evaluate(x_test, y_test)
# print("Test Loss:", test_loss)
# print("Test F1 Score:", test_score)

# weights = model.get_weights()

# print("get weights")
# # min_abs_list = []
# max_abs_list = []
# # max_weight_list = []
# # min_weight_list = []
# for i in range(len(weights)):
#     if len(np.shape(weights[i])) == 1:  # skip layers with 1-dimesnion weights
#             continue
#     num_weight += weights[i].size
#     max_weight = np.max(weights[i])
#     min_weight = np.min(weights[i])
#     # max_abs = max(abs(max_weight), abs(min_weight))
#     # min_abs_list.append(min(abs(max_weight), abs(min_weight)))
#     max_abs_list.append(max(abs(max_weight), abs(min_weight)))
#     # max_weight_list.append(max_weight)
#     # min_weight_list.append(min_weight)
    
# print(type(weights))
# print(type(weights[0]))

# influence = cal_influence(weights, 10)
# print("get influence")
# allocation = allocate_bits(influence, AVE_BITS_PER_WEIGHT - 2)
# print("get allocation")
# quantilized = quant(weights, allocation, max_abs_list)
# print("get quantilized weights")
# model.set_weights(quantilized)
# print("set quantilized weights")

# # Our quantization evaluation
# test_loss, test_score = model.evaluate(x_test, y_test)
# print('Our quantization test loss:', test_loss)
# print("Our Quantization test F1 Score:", test_score)

# # Uniform Quantization
# allocation = uniform_allocate_bits(influence, AVE_BITS_PER_WEIGHT - 2)
# uniform_quantilized = quant(weights, allocation, max_abs_list)
# model.set_weights(uniform_quantilized)

# # resnet
# test_loss, test_score = model.evaluate(x_test, y_test)
# print('Uniform quantization test loss:', test_loss)
# print("Uniform Quantization test F1 Score:", test_score)