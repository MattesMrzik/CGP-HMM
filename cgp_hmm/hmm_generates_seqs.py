#!/usr/bin/env python3

import numpy as np


# with open("identical_seqs_with_repeates.out","w") as file:
#     for i in range(10):
#         file.write(">id" + str(i) + "\n")
#         file.write("ACGT"*3 + "\n")
# def generate_state_emission_seqs(a,b,n,l, init_dist = None):
#
#     state_space_size = len(a)
#     emission_space_size = len(b[0])
#
#     list_to_state_space_size = list(range(state_space_size))
#     list_to_emission_space_size = list(range(emission_space_size))
#
#
#     states = np.zeros((n,l,state_space_size))
#     emissions = np.zeros((n,l,emission_space_size))
#
#     def rand_with_p(list, dist):
#         return np.random.choice(list, size = 1, p = dist)[0]
#
#     for i in range(n):
#         states[i,0,0 if init_dist == None else rand_with_p(list_to_state_space_size, init_dist)] = 1
#         emissions[i,0, rand_with_p(list_to_emission_space_size, b[np.argmax(states[0,0,:])])] = 1
#         for j in range(l):
#             states[i,j, rand_with_p(list_to_state_space_size, a[np.argmax(states[i,j-1])])] = 1
#             emissions[i,j, rand_with_p(emission_space_size, b[np.argmax(states[i,j-1])])] = 1
#         # emssions.write(seq + "\n")
#         # states.write(">id" + str(i) + "\n")
#         # states.write(state_seq + "\n")
#     return states, emissions
#
# a = np.array([[.85,.15], [.1,.9]])
# b = np.array([[.1,.35,.35,.2], [.4,.1,.1,.4]])

# print(generate_state_emission_seqs(a,b, 3, 10))
