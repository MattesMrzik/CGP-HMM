#!/usr/bin/env python3
import numpy as np
import pandas as pd

np.set_printoptions(linewidth=100)

py = pd.read_csv("forward_py.txt")
tf = pd.read_csv("forward_tf.txt")

print(py)
print(tf)
py = np.array(py)
tf = np.array(tf)

print(py)
print(tf)
print("minus")
print(py-tf)
