#!/usr/bin/env python3
import numpy as np
import pandas as pd
# import tensorflow as tf

# b = tf.reshape(tf.constant([i for i in range(50)]),(10,5))
# r = tf.reshape(tf.constant([i for i in range(1,51)]),(10,5))
# print(b*r)
#
# x = tf.constant([1,2,3], name = "x")
# y = tf.constant([1,2,3], name = "y")
# z = tf.constant([1,2,3], name = "z")
# l = [lambda x: x, name = "y"([x,y,z])]

# s ="0.2365097 0.23614341 0.2599357 0.26741126"
# l = list(map(float, s.split(" ")))
# print("sum = ",sum(l))
# print(sum(tf.nn.softmax(l)))

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x))
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# x = .7
# print(softmax([x,0]))
# print(f"{sigmoid(x):3d}, {1 - sigmoid(x)}")

#
# s = "[1] [1 0 0 0 0] [0.2604 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [2] [0 0 0 1 0] [0.179799989 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [3] [0 0 1 0 0] [0.1762 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [4] [0 1 0 0 0] [0.186999992 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [5] [0 0 1 0 0] [0.1762 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [6] [1 0 0 0 0] [0.191199988 0.265699983 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [7] [0 0 1 0 0] [0.0738 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [8] [0 1 0 0 0] [0.186999992 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [9] [0 0 1 0 0] [0.1762 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [10] [0 0 0 1 0] [0.179799989 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [11] [1 0 0 0 0] [0.191199988 0.265699983 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [12] [1 0 0 0 0] [0.08 0.1112 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [13] [0 0 1 0 0] [0.0738 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [14] [0 0 0 0 1] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],\
# [15] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [16] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [17] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [18] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [19] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [20] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [21] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [22] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [23] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [24] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [25] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [26] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [27] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [28] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [29] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan],\
# [30] [0 0 0 0 1] [-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan]"
#
# for line in s.split(","):
#     x = line.split("] [")
#     index = x[0][1:]
#     emission = ["ACGT$"[i] for i in range(5) if x[1].split(" ")[i] == "1"][0]
#     alpha = "\t".join([a[:5] for a in x[2][:-1].split(" ")])
#     print(index,emission,"\t",alpha)
#     if int(index) % 5 == 0:
#         print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
#         print("\t","ig a t g c1 c2 c3 c1 c2 c3 t a g a a g a ig i1 i2 i3 i1 i2 i3 i1 i2 i3 ter".replace(" ","\t"), sep = "")
#         print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

################################################################################
################################################################################
################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
#
#
# q = 3
# alphabet_size = 4
# def n(x):
#     return list(range(1,x+1))
# b = np.array([[[[i*j*k*l for i in n(alphabet_size)] for j in n(alphabet_size)] for k in n(alphabet_size)] for l in n(q)], dtype=np.int32)
# b = np.array([[[[i+j*10+k*100+l*1000 for i in n(alphabet_size)] for j in n(alphabet_size)] for k in n(alphabet_size)] for l in n(q)], dtype=np.int32)
# print("b=")
# print(b)
# old_inputs_2 = [[0,0,0,1],[0,1,0,0]]
# old_inputs_1 = [[0,0,1,0],[0,1,0,0]]
# inputs       = [[0,0,0,1],[0,0,1,0]]
# old_inputs_2_t = tf.transpose(old_inputs_2)
# old_inputs_1_t = tf.transpose(old_inputs_1)
# inputs_t       = tf.transpose(inputs)
# bt = tf.transpose(b)
# print("bt=", bt)
# x1 = tf.tensordot(inputs, tf.transpose(b), axes=(1,0))
# print("x1")
# print(x1)
# # x2 = tf.transpose(tf.transpose(old_inputs_1) * tf.transpose(x1))
# old_inputs_1 = tf.expand_dims(old_inputs_1, axis = -1)
# old_inputs_1 = tf.expand_dims(old_inputs_1, axis = -1)
# # now: old_inputs_1 has shape[2 4 1 1]
# # now it can be broadcasted to [2 4 4 3]
# print("shapes of x1 and old_inputs_1 =", tf.shape(x1), tf.shape(old_inputs_1))
# x2 = tf.multiply(old_inputs_1, x1)
# print("x2 before reduce sum")
# print(x2)
# # reduce sum is along axis that is as large as emission alphabet_size
# x2 = tf.reduce_sum(x2, axis = 1) # axis 0 is batch, so this has to be 1
# print("x2")
# print(x2)
# old_inputs_2 = tf.expand_dims(old_inputs_2, axis = -1)
# x3 = tf.multiply(old_inputs_2, x2)
# x3 = tf.reduce_sum(x3, axis = 1)
# print("x3")
# print(x3)
#
# b = np.random.rand(3,4,4,4)
# b = tf.convert_to_tensor(b, dtype=tf.float32)
# x = tf.keras.activations.softmax(b, axis = [1,2,3])
# print("after softmax")
# print(x)
# x = tf.reduce_sum(x, axis = -1)
# print(x)
# x = tf.reduce_sum(x, axis = -1)
# print(x)
# x = tf.reduce_sum(x, axis = -1)
# print(x)

# inputs = tf.concat([old_inputs_2, old_inputs_1, inputs], axis=1)
# print(inputs)
#
# y = tf.tensordot(old_inputs_2, old_inputs_1, axes = (1))
# print(y)

################################################################################
################################################################################
################################################################################
# old = np.array([[1,0,0,0],[0,0,0,1],[0,1,0,0]])
# inputs = np.array([[0,1,0,0],[0,0,1,0],[0,1,0,0]])
# first_tensor = inputs
# first_tensor = tf.expand_dims(tf.transpose(inputs), axis = -1)
# print("first_tensor")
# print(first_tensor)
# # second_tensor = tf.reshape(old, (12,))
# second_tensor = tf.expand_dims(old, axis = -1)
# print("second_tensor")
# print(second_tensor)
# from itertools import product
# for i,j in product([0,1,2], repeat=2):
#     try:
#         x = tf.tensordot(first_tensor, second_tensor, axes=(i,j))
#         print(f"x for {i}, {j}")
#         print(x)
#     except:
#         print(f"x for {i}, {j} didnt work")
################################################################################
################################################################################
################################################################################
indices = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,0,1]]
values = [1, 2, 1.5, 2, 1.2]
m = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = (2,2,2))
m = tf.sparse.reorder(m)
m = tf.sparse.reshape(m, (2,4))
m = tf.sparse.softmax(m)
m = tf.sparse.reshape(m, (2,2,2))

m = tf.sparse.to_dense(m)
print(m)
