#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from CgpHmmLayer import CgpHmmLayer
from ReadData import read_data_one_hot

def prRed(skk): print("Training\033[92m {}\033[00m" .format(skk))

np.set_printoptions(linewidth=400)

def make_model():
    alphabet_size = 4

    sequences = tf.keras.Input(shape = (50, alphabet_size), name = "sequences")
    # another None added automatically for yet unkown batch_size
    # todo what if sequences have differing lenghts, then the 24 cant stay, padding?

    cgp_hmm_layer = CgpHmmLayer()

    loglik = cgp_hmm_layer(sequences)

    print([tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik)])

    model = tf.keras.Model(inputs = sequences, outputs = [tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik)]) #  the output of the model is the value that is computed by a final layer that picks the loglike of the [alpha, loglik, count]

    return model, cgp_hmm_layer


def make_dataset():
    seqs = []
    one_hot = read_data_one_hot("../rest/intron_AT_exon_GC.out")
    ds = tf.data.Dataset.from_tensors(one_hot).repeat()
    return ds, one_hot

def fit_model():
    model, cgp_hmm_layer = make_model()#
    learning_rate = .1
    optimizer = tf.optimizers.Adam(learning_rate)
    model.compile(optimizer = optimizer)

    # manual call to forward algo

    # _, seqs = make_dataset()# first return value is data_set
    # model(seqs)

    data_set = make_dataset()[0] # [1] is data tensor
    class my_callback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs = None):
            print("model.weights")
            print("A =", tf.nn.softmax(model.get_weights()[0]))
    checkpoint_path = "training_1/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)


    # callbacks = [tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: print("A =", tf.nn.softmax(model.get_weights()[0])))]
    callbacks = [cp_callback]
    callbacks = []

    history = model.fit(data_set, epochs=5, steps_per_epoch=10, callbacks = callbacks) # with callbacks it is way slower
    return model, history
