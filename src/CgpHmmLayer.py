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
        prRed("build of CgpHmmLayer")
        prRed("self.C = CgpHmmCell()")
        self.C = CgpHmmCell()
        # print(11)
        # self.C.build(input_shape)
        prRed("self.F = tf.keras.layers.RNN(self.C, return_state = True, return_sequences = True) # F = forward ie the chain of cells C")
        self.F = tf.keras.layers.RNN(self.C, return_state = True, return_sequences = True) # F = forward ie the chain of cells C
        prRed("build of CgpHmmLayer done")

    def call(self, inputs, training = False):

        # todo do i need to reset statse?


        prRed("call of CgpHmmLayer:")
        print("inputs = ")
        # tf.print(inputs, summarize=100)
        prRed("_, _, loglik = self.F(inputs)")
        # _, _, loglik = self.F(inputs, initial_state = self.C.get_initial_state()) #  build and call of CgpHmmCell are called

        # todo need initial states: for init of alpha, bc alpha_0 oder alpha_-1 is initialized with zeros

        # _, _, loglik = self.F(inputs) #  build and call of CgpHmmCell are called
        alpha_seq, inputs_seq, count_seq,  alpha_state, loglik_state, count_state = self.F(inputs) #  build and call of CgpHmmCell are called
        # todo was ist a? shape is 50,24,1 the likelihood bc return seq we get 24, is this the first
        # variable that is returned in call of cell
        # b has shape 50,5 -> return state? of last cell state, can be different to hidden state in lstm
        # isnt the loglik just sum over states of b?
        # loglike has shape 50,1

        # prRed("alpha_seq, inputs_seq, count_seq,  alpha_state, loglik_state, count_state")
        # print(alpha_seq, inputs_seq, count_seq,  alpha_state, loglik_state, count_state)

        prRed("alpha_seq, inputs_seq, count_seq")
        # return_seq = tf.concat((count_seq, inputs_seq,alpha_seq), axis = 1)
        # tf.print(return_seq, summarize = 100)
        for i in range(tf.shape(alpha_seq)[0]): # = batch_size
            prRed(i)
            for j in range(tf.shape(alpha_seq)[1]): # = seq length
                tf.print(count_seq[i,j], inputs_seq[i,j], alpha_seq[i,j])
            break


        likelihood_mean = tf.reduce_mean(loglik_state)
        # squeeze removes dimensions of size 1, ie shape (1,3,2,1) -> (3,2)
        self.add_loss(tf.squeeze(-likelihood_mean))

        if training:
            prRed("training is true")
            self.add_metric(likelihood_mean, "likelihood")
        else:
            prRed("training is false")

        prRed("return loglik")
        return loglik_state
