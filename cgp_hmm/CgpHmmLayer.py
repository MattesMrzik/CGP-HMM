#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

from CgpHmmCell import CgpHmmCell

def prRed(skk): print("Layer\033[91m {}\033[00m" .format(skk))
# def prRed(s): pass

class CgpHmmLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CgpHmmLayer, self).__init__()

    def build(self, input_shape):
        self.C = CgpHmmCell()
        self.C.build(input_shape)
        self.F = tf.keras.layers.RNN(self.C, return_state = True, return_sequences = True) # F = forward ie the chain of cells C

    def call(self, inputs, training = False):

        # todo do i need to reset statse?

        alpha_seq, inputs_seq, count_seq,  alpha_state, loglik_state, count_state = self.F(inputs) #  build and call of CgpHmmCell are called

        # # print return sequences
        # prRed("alpha_seq, inputs_seq, count_seq")
        # return_seq = tf.concat((count_seq, inputs_seq,alpha_seq), axis = 1)
        # tf.print(return_seq, summarize = 100)
        # for i in range(tf.shape(alpha_seq)[0]): # = batch_size
        #     prRed(i)
        #     for j in range(tf.shape(alpha_seq)[1]): # = seq length
        #         tf.print(count_seq[i,j], inputs_seq[i,j], alpha_seq[i,j])
        #     break

        loglik_mean = tf.reduce_mean(loglik_state)
        # squeeze removes dimensions of size 1, ie shape (1,3,2,1) -> (3,2)
        self.add_loss(tf.squeeze(-loglik_mean))

        if training:
            prRed("training is true")
            self.add_metric(loglik_mean, "loglik")
        else:
            prRed("training is false")

        return loglik_state
