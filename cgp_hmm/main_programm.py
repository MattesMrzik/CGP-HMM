#!/usr/bin/env python3

# import cgp_hmm
import matplotlib.pyplot as plt
import tensorflow as tf
from Training import fit_model

model, history = fit_model()
# model.save("my_saved_model")

plt.plot(history.history['loss'])
plt.show()


print("A =", tf.math.round(tf.nn.softmax(model.get_weights()[0])*100)/100)
print("B =", tf.math.round(tf.nn.softmax(model.get_weights()[1])*100)/100)
