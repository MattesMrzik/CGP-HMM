#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import time
import tracemalloc
from random import randint

# from memory_profiler import profile
# WARNING:tensorflow:AutoGraph could not transform <bound method LineProfiler.wrap_function of <memory_profiler.LineProfiler object at 0x7fd8c4032af0>> and will run it as-is.
# Cause: generators are not supported


from Utility import description_to_state_id
from Utility import append_time_stamp_to_file
from Utility import append_time_ram_stamp_to_file

from CgpHmmCell import CgpHmmCell

def prRed(skk): print("Layer\033[96m {}\033[00m" .format(skk))
# def prRed(s): pass

class CgpHmmLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(start, f"Layer.init() start {run_id}", config["bench_path"])
        super(CgpHmmLayer, self).__init__()
        self.nCodons = config['nCodons']
        self.config = config
        self.order_transformed_input = config['order_transformed_input']

        append_time_ram_stamp_to_file(start, f"Layer.init() end  {run_id}", self.config["bench_path"])

    def build(self, input_shape):
        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(start, f"Layer.build() start {run_id}", self.config["bench_path"])
        # print("in build of layer")
        self.C = CgpHmmCell(self.config) # init
        # self.C.build(input_shape) # build
        # this isnt needed for training but when calling the layer, then i need to build C manually, but it is then called
        # a second time when calling F
        # tf.print("before RNN")
        self.F = tf.keras.layers.RNN(self.C, return_state = True, return_sequences = True) # F = forward ie the chain of cells C
        # tf.print("after RNN")

        append_time_ram_stamp_to_file(start, f"Layer.build() end   {run_id}", self.config["bench_path"])

    def call(self, inputs, training = False):
        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(start, f"Layer.call() start {run_id}", self.config["bench_path"])
        # todo do i need to reset statse?
        # cell is build again


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
        # if self.C.order > 0 and not self.C.order_transformed_input : # or True to checksquare
        #     old_state = result[6]


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
            if not self.config["call_type"] == 4:
                probs_to_be_punished.append(tf.math.log(1 - \
                                            self.C.A_dense[description_to_state_id(f, self.nCodons), \
                                                     description_to_state_id(to, self.nCodons)]))
            else:
                 probs_to_be_punished.append(tf.math.log(1 - \
                                             self.C.A_full_model[description_to_state_id(f, self.nCodons), \
                                                      description_to_state_id(to, self.nCodons)]))

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

        append_time_ram_stamp_to_file(start, f"Layer.call() end   {run_id}", self.config["bench_path"])
        return loglik_state
