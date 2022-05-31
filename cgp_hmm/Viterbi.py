#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from ReadData import read_data_one_hot

# def viterbi(A, B, seq):# seq is one hot
#     n = len(seq)
#     g = np.log(np.zeros((len(A), n)))
#     # todo this needs to be adjusted if states other than the first are allowed in the first slot
#     g[0,0] = np.log(B[0, np.argmax(seq[0])]) # argmax, bc seq is one hot encoded
#     # todo only save last col of gamma, for backtracking recompute
#     for i in range(1, n):
#         for q in range(len(A)):
#             m = max(np.log(A[:,q]) + g[:,i-1])
#             g[q,i] = np.log(B[q, np.argmax(seq[i])]) + m
#     # back tracking
#     x = np.zeros(n, dtype = np.int32)
#     x[n-1] = np.argmax(g[:,n-1])
#     for i in range(n-2, 0, -1):
#         x[i] = np.argmax(np.log(A[:,x[i+1]]) + g[:,i])
#     return(x)
#
# def brute_force_viterbi(a,b,seq):
#     from itertools import product
#     max = float("-inf")
#     arg_max = 0
#     for guess in product(list(range(len(a))), repeat=len(seq)-1):# one less length, since seqs must start in state 0
#         # todo this needs to be adjusted if states other than the first are allowed in the first slot
#         guess = [0] + list(guess)
#         p = np.log(b[0, np.argmax(seq[0])])
#         for i in range(1,len(seq)):
#             p += np.log(a[guess[i-1], guess[i]])
#             p += np.log(b[guess[i], np.argmax(seq[i])])
#         if p > max:
#             max = p
#             arg_max = guess
#     return np.array(arg_max)
