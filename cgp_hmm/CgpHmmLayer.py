#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

from Utility import description_to_state_id

from CgpHmmCell import CgpHmmCell

def prRed(skk): print("Layer\033[91m {}\033[00m" .format(skk))
# def prRed(s): pass

class CgpHmmLayer(tf.keras.layers.Layer):
    def __init__(self, nCodons, order_transformed_input):
        super(CgpHmmLayer, self).__init__()
        self.nCodons = nCodons
        self.order_transformed_input = order_transformed_input

    def build(self, input_shape):
        # print("in build of layer")
        self.C = CgpHmmCell(self.nCodons, self.order_transformed_input) # init
        # self.C.build(input_shape) # build
        # this isnt needed for training but when calling the layer, then i need to build C manually, but it is then called
        # a second time when calling F
        # tf.print("before RNN")
        self.F = tf.keras.layers.RNN(self.C, return_state = True, return_sequences = True) # F = forward ie the chain of cells C
        # tf.print("after RNN")

    def call(self, inputs, training = False):
        # todo do i need to reset statse?
        # cell is build again

        # alpha_seq, \
        # inputs_seq, \
        # count_seq, \
        # alpha_state, \
        # loglik_state, \
        # count_state, \
        # old_state_2, \
        # old_state_1 = self.F(inputs) #  build and call of CgpHmmCell are called

        # tf.print("in call of layer")

        # todo: felix macht auch nochmal a und b
        self.C.init_cell()
        # tf.print("in call of layer: self.C.init =", self.C.init)

        # tf.print("before result = self.F(inputs)")
        result = self.F(inputs) #  build and call of CgpHmmCell are called
        # tf.print("after result = self.F(inputs)")

        alpha_seq = result[0]
        inputs_seq = result[1]
        count_seq = result[2]
        alpha_state = result[3]
        loglik_state = result[4]
        count_state = result[5]
        if self.C.order > 0 and not self.C.order_transformed_input : # or True to checksquare
            old_state = result[6]


        # print return sequences
        # prRed("alpha_seq, inputs_seq, count_seq")
        # print(tf.shape(count_seq),tf.shape(inputs_seq),tf.shape(alpha_seq))
        # for i in range(tf.shape(alpha_seq)[0]): # = batch_size
        #     prRed(i)
        #     if i != 3:
        #         continue
        #     for j in range(tf.shape(alpha_seq)[1]): # = seq length
        #         tf.print(count_seq[i,j], inputs_seq[i,j], tf.math.round(alpha_seq[i,j]*10000)/10000, summarize = -1)


        loglik_mean = tf.reduce_mean(loglik_state)
        # squeeze removes dimensions of size 1, ie shape (1,3,2,1) -> (3,2)

        # regularization
        # siehe Treffen_04
        alpha = 4 # todo: scale punishment for inserts with different factor?
        probs_to_be_punished = []

        def add_reg(f, to):
            probs_to_be_punished.append(tf.math.log(1 - \
                                        self.C.A_dense()[description_to_state_id(f, self.C.nCodons), \
                                                 description_to_state_id(to, self.C.nCodons)]))

        # deletes to be punished
        for i in range(1, self.C.nCodons):
            add_reg("stG", f"c_{i},0")
        add_reg("stG", "stop1")
        for i in range(self.C.nCodons - 1):
            for j in range(i + 2, self.C.nCodons):
                add_reg(f"c_{i},2", f"c_{j},0")
            add_reg(f"c_{i},2", "stop1")
        # inserts to be punished
        add_reg("stG", "i_0,0")
        for i in range(self.C.nCodons):
            add_reg(f"c_{i},2", f"i_{i+1},0")

        reg_mean = sum(probs_to_be_punished) / len(probs_to_be_punished)


        if loglik_mean < 0 and reg_mean >0:
            tf.print("not same sign")
        if loglik_mean > 0 and reg_mean <0:
            tf.print("not same sign")


        self.add_loss(tf.squeeze(-loglik_mean - alpha * reg_mean))

        # if training:
            # tf.print("loglik_mean = ", loglik_mean)
            # tf.print("reg_mean = ", reg_mean)


        if training:
            prRed("training is true")
            self.add_metric(loglik_mean, "loglik")
            self.add_metric(reg_mean, f"reg_mean, not yet multiplied by alpha({alpha})")
        else:
            prRed("training is false")

        return loglik_state
