import os
import math
import numpy as np
import cv2 as cv
import keras
import tensorflow as tf

from keras.applications.inception_v3 import InceptionV3
from keras.applications import inception_v3
from keras import optimizers
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.models import load_model
from resnet_quant import cal_influence, allocate_bits, quant, uniform_allocate_bits

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

num_classes = 1000

dataset_path = 'F:\\ILSVRC2020_img_validation\\'
image_list = os.listdir(dataset_path)
index = 0
acc_sum = 0

##########################################################

resnet_model = InceptionV3(weights='imagenet')
print("get model")

AVE_BITS_PER_WEIGHT = 8
num_weight = 0
max_weight = 0
min_weight = 0


weights = resnet_model.get_weights()
print(len(weights))
for i in range(len(weights)):
  print(weights[i].shape)

max_abs_list = []
for i in range(len(weights)):
    if len(np.shape(weights[i])) == 1:  # skip layers with 1-dimesnion weights
            continue
    num_weight += weights[i].size
    max_weight = np.max(weights[i])
    min_weight = np.min(weights[i])
    max_abs_list.append(max(abs(max_weight), abs(min_weight)))

influence = cal_influence(weights, 10)
print("get influence")
allocation = allocate_bits(influence, AVE_BITS_PER_WEIGHT - 2)
print("get allocation")
quantilized = quant(weights, allocation, max_abs_list)
print("get quantilized weights")
resnet_model.set_weights(quantilized)
print("set quantilized weights")
opt = optimizers.Adam(lr=0.001)
resnet_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

uniformed_resnet_model = InceptionV3(weights='imagenet')
allocation = uniform_allocate_bits(influence, AVE_BITS_PER_WEIGHT - 2)
uniform_quantilized = quant(weights, allocation, max_abs_list)
uniformed_resnet_model.set_weights(uniform_quantilized)

opt = optimizers.Adam(lr=0.001)
uniformed_resnet_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

resnet_model.save_model("my_inception_quantilized.h5")
uniformed_resnet_model.save_model("uniform_inception_quantilized.h5")

##########################################################

# resnet_model = load_model('my_inception_quantilized.h5')
# uniformed_resnet_model = load_model('uniform_inception_quantilized.h5')

##########################################################
F1 = open('F:\ILSVRC2012_devkit_t12\\data\\val.txt', "r")
List_row = F1.readlines()
list_source = []
for i in range(len(List_row)):
    column_list = List_row[i].strip().split(" ")  
    list_source.append(column_list)            
 

a = np.array(list_source)
y_test = a[:, 1].astype(np.int32)

correct = 0
uniform_correct = 0
for index in range(1000):
  first_img = image.load_img(dataset_path + image_list[index], target_size = (299, 299))
  temp_num_img = image.img_to_array(first_img)
  predictions = resnet_model.predict(inception_v3.preprocess_input(np.expand_dims(cv.resize(temp_num_img, (299, 299)), 0)))
  classes = np.argsort(predictions)
  if y_test[index] in classes[0][995:1000]:
    correct += 1
  predictions = uniformed_resnet_model.predict(inception_v3.preprocess_input(np.expand_dims(cv.resize(temp_num_img, (299, 299)), 0)))
  classes = np.argsort(predictions)
  if y_test[index] in classes[0][995:1000]:
    uniform_correct += 1

print(correct / 10)
print(uniform_correct / 10)