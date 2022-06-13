#!/usr/bin/env python3

# import cgp_hmm
import matplotlib.pyplot as plt
import tensorflow as tf
from Training import fit_model
import Utility
import numpy as np
from Bio import SeqIO


a = np.array([[.9,.1,0],[0,.8,.2],[0,.15,.85]])
b = np.array([0.6,  0.2,   0.1,  0.1, \
              0.2,  0.5,   0.1, 0.2, \
              0.1,  0.1,   0.6,  0.2], \
              dtype = np.float32).reshape((3,4))
n = 1000
l = 50
states, emissions = Utility.generate_state_emission_seqs(a,b,n,l)
with open("../rest/sparse_toy_emissions.out","w") as out_handle:
    for id, emission in enumerate(emissions):
        out_handle.write(">id" + str(id) + "\n")
        out_handle.write("".join([["A","C","G","T"][i] for i in emission]) + "\n")

model, history = fit_model("../rest/sparse_toy_emissions.out")
# model.save("my_saved_model")

plt.plot(history.history['loss'])
plt.show()


# print("A =", tf.math.round(tf.nn.softmax(model.get_weights()[0])*100)/100)
w = model.get_weights()[0]
a_hat = np.zeros((3,3))
a_hat[0,0:2] = tf.nn.softmax([1-w[0], w[0]])
a_hat[1,1:3] = tf.nn.softmax([1-w[1], w[1]])
a_hat[2,1:3] = tf.nn.softmax([w[2], 1-w[2]])

print("A =", a_hat)
print("B =", tf.math.round(tf.nn.softmax(model.get_weights()[1])*100)/100)
