#!/usr/bin/env python3.8
import numpy as np
import pandas as pd
import re
import os


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
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

#kernprof -l -v playground.py

# import argparse
#
# parser = argparse.ArgumentParser(
#     description='description')
# parser.add_argument('-x', '--xx',
#                     help='xx')
# args = parser.parse_args()
#
# if not args.xx:
#     args.xx = 7
#
# import time
# start = time.perf_counter()
# import tracemalloc
# tracemalloc.start()
#
# import random
#
# from resource import getrusage
#
# memory = True
# if memory:
#     from memory_profiler import profile
#
# # @profile
# def f(cap = float("inf")):
#     l = list(range(10**min(int(args.xx), cap))) + [random.randint(10,100)]
#     print("inf f getrusage(RUSAGE_SELF) =", getrusage(RUSAGE_SELF).ru_maxrss)
#     return l
#
# l1 = f()
# print("l1 getrusage(RUSAGE_SELF) =", getrusage(RUSAGE_SELF).ru_maxrss)
# del(l1)
# print("l1 del getrusage(RUSAGE_SELF) =", getrusage(RUSAGE_SELF).ru_maxrss)
#
# l2 = f(1)
# print("l2 getrusage(RUSAGE_SELF) =", getrusage(RUSAGE_SELF).ru_maxrss)
#
# print("sleeping")
# time.sleep(10)
# print("awakening")
#
# snapshot = True
# if snapshot:
#     snapshot = tracemalloc.take_snapshot()
#
#     top_stats = snapshot.statistics('lineno')
#
#     print("[ Top 10 ]")
#     for stat in top_stats[:10]:
#         print(stat)
#
# trace = tracemalloc.get_traced_memory()
# print("trace =", trace)
# with open("tracemalloc.log","a") as file:
#     file.write(f"{args.xx},{trace}\n")
# tracemalloc.stop()
#
# print("l2 getrusage(RUSAGE_SELF) =", getrusage(RUSAGE_SELF).ru_maxrss)

################################################################################
################################################################################
################################################################################
# import tensorflow as tf
# parameter1 = tf.Variable(4.0)
# parameter2 = tf.Variable(4.0)
#
# with tf.GradientTape() as tape:
#     tape.watch([parameter1, parameter2])
#     y = parameter1**2 * tf.math.sqrt(parameter2)
# dy_dx = tape.gradient(y, [parameter1, parameter2])
# print("dy_dx =", dy_dx)
#
# @tf.function
# def example():
#   a = tf.constant(0.)
#   b = 2 * a
#   return tf.gradients(a**2 + b, [a, b], stop_gradients=[a, b])
# e = example()
# print(e)
################################################################################
################################################################################
################################################################################

# n =
# batchsize = 32
# low = 0
# high = min(batchsize, n)
# for epoch in range(3):
#     for step in range(5):
#         print(f"batch {low}, {high}")
#         low = low + batchsize if low + batchsize < n else 0
#         # TODO: does batchsize instead of min also work? if batch_size + high == n
#         if high == n:
#             high = min(batchsize, n)
#         elif high + batchsize > n:
#             high = n
#         else:
#             high += batchsize

################################################################################
################################################################################
################################################################################
# import numpy as np
# import math
# def LSE(x): # x is vectory
#     s = sum(np.exp(x))
#     print(f"s in LSE = {s}")
#     try:
#         result = math.log(s)
#         return result
#     except:
#         print("log(0)")
# x = np.array([-1001,-1000])
# print("LSE =",LSE(x))
#
# def logLSE(x):
#     m = max(x)
#     return m + LSE(x-m)
#
# print("logLSE =", logLSE(x))

################################################################################
################################################################################
################################################################################
#
# x = np.random.choice([0,1], size = 10)
# print(x)
# a = 0
# b = 0
# best = 0
# current = 0
# potential_start = 0
# for i in range(len(x)):
#     if x[i] == 0:
#         current -=1
#     else:
#         current += 1
#     if current < 0:
#         potential_start = i+1
#         current = 0
#     if current > best:
#         a = potential_start
#         b = i
#         best = current
# print(x)
# print(x[a:b+1])
# print(f"a = {a}, b = {b}, best = {best}")

################################################################################
################################################################################
################################################################################

# class A():
#     @classmethod
#     def foo(cls):
#         print("foo")
#
#     def bar(self):
#         self.foo()
#
# a = A()
# a.bar()
# a.foo()

################################################################################
################################################################################
################################################################################
# import tensorflow as tf
# state_size = 3
# emissions_state_size = 2
# alphabet_size = 4
# m = tf.cast(tf.constant(np.arange(state_size * emissions_state_size * alphabet_size)), dtype=tf.float32)
# mask = tf.constant([0,0] * int((state_size * emissions_state_size * alphabet_size)/2))
# m = tf.reshape(m, (emissions_state_size * alphabet_size,-1))
# mask = tf.reshape(mask, (emissions_state_size * alphabet_size,-1))
# print(m, mask)
# m = tf.reshape(m, (-1, alphabet_size, state_size))
# mask = tf.reshape(mask, (-1, alphabet_size, state_size))
# print(m, mask)
# # m = tf.nn.softmax(tf.cast(m, dtype = tf.float32)/10, axis = 1)
# layer = tf.keras.layers.Softmax(axis = 1)
# m = layer(m, mask)
# print(m, mask)
# m = tf.reshape(m, (emissions_state_size * alphabet_size,-1))
# print(m, mask)

################################################################################
################################################################################
################################################################################

# find epsilons for logspace
# import tensorflow as tf
# import numpy as np
# import datetime
# emissions_state_size = 8
# state_size = 5
# batch_size = 4
# seqlen = 20
# batch = np.random.randint(emissions_state_size, size = (batch_size, seqlen))
# batch = tf.one_hot(batch, emissions_state_size)
# print("batch =", batch)
#
# optimizer = tf.keras.optimizers.SGD()
#
# np.set_printoptions(linewidth=100000)
#
#
# I_ker = tf.cast([1] + [0] * (state_size -1), dtype = tf.float32)
# A_ker = tf.cast(tf.constant(np.random.rand(state_size, state_size)), dtype = tf.float32)
# B_ker = tf.cast(tf.constant(np.random.rand(emissions_state_size, state_size)), dtype = tf.float32)
#
# epsilon = 1e-2
# for step in range(400):
#     with tf.GradientTape() as tape:
#         tape.watch([A_ker,B_ker])
#
#         A = tf.nn.softmax(A_ker) # + epsilon didnt help
#         B = tf.nn.softmax(B_ker)
#         # print(f"\n\nA =\n{A},\nB =\n{B}")
#
#         alpha = tf.math.log(tf.matmul(batch[:,0,:], B)) + tf.math.log(I_ker)
#         for i in range(1, seqlen):
#             m_alpha = tf.reduce_max(alpha, axis = 1, keepdims = True)
#             # print("alpha =", alpha)
#             # print("m_alpha =", m_alpha)
#             E = tf.math.log(tf.matmul(batch[:,i,:], B))
#             R =  tf.math.log(tf.matmul(tf.math.exp(alpha - m_alpha), A)) + m_alpha
#             alpha = E + R
#
#         # loglike = tf.math.reduce_logsumexp(alpha + epsilon, axis=1)
#         m_alpha = tf.reduce_max(alpha, axis = 1, keepdims = True)
#         loglike = tf.math.log(tf.reduce_sum(tf.math.exp(alpha - m_alpha) + epsilon, axis=1, keepdims = True))
#         mean_loglike = tf.reduce_mean(loglike)
#
#         true_loglike = tf.math.log(tf.reduce_sum(tf.math.exp(alpha - m_alpha), axis=1, keepdims = True))
#         true_mean_loglik =  tf.reduce_mean(true_loglike)
#
#     print(f"mean_loglike = {mean_loglike}, error = {true_mean_loglik - mean_loglike}")
#
#     log_dir = "logs/playground/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     grad = tape.gradient(mean_loglike, [A,B])
#     assert tf.reduce_all(tf.math.is_finite(grad[0])), f"grad {grad[0]} for A is not finite"
#     assert tf.reduce_all(tf.math.is_finite(grad[1])), f"grad {grad[1]} for B is not finite"
#
#     # optimizer.apply_gradients(zip(grad, [A_ker,B_ker]))
#
#     A_ker += grad[0] * 0.01
#     B_ker += grad[1] * 0.01
#

################################################################################
################################################################################
################################################################################

# class Klasse():
#     def foo(self):
#         a = 1
#         l = []
#         def bar():
#             print(a,l)
#         bar()
#     def call_foo(self):
#         self.foo()
# c = Klasse()
# c.call_foo()

################################################################################
################################################################################
################################################################################

# what is the max of the derichlet density

import tensorflow as tf
import tensorflow_probability as tfp

w = tf.Variable([0.5,0.5,0.5])
alpha = tf.constant([0.5,0.2,0.3])
dirichlet = tfp.distributions.Dirichlet(alpha)
iterations = 100
for i in range(iterations):
    with tf.GradientTape() as tape:
        tape.watch(w)
        p = tf.nn.softmax(w)
        print("p", p.numpy())
        print("p sum", tf.math.reduce_sum(p).numpy())

        y = -tf.reduce_prod(p**(alpha-1))
        derichlet_prob = dirichlet.prob(p).numpy()
        frac = y/derichlet_prob
        print("frac", frac.numpy())
        print("y", y.numpy())
    grad = tape.gradient(y,p)
    print("dy_dx =", [d.numpy() for d in grad])
    grad = grad / tf.linalg.norm(grad)
    w.assign_sub(0.1 * grad)
    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(w)), [w])
    print()

max_value = 1 / (tf.exp(tf.math.lbeta(alpha)) * tf.reduce_prod(alpha - 1))
print("max_value", max_value)