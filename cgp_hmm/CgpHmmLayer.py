#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import time
import tracemalloc
from random import randint
import traceback
import json
import os
# from memory_profiler import profile
# WARNING:tensorflow:AutoGraph could not transform <bound method LineProfiler.wrap_function of <memory_profiler.LineProfiler object at 0x7fd8c4032af0>> and will run it as-is.
# Cause: generators are not supported


from Utility import description_to_state_id
from Utility import append_time_stamp_to_file
from Utility import append_time_ram_stamp_to_file

from CgpHmmCell import CgpHmmCell



class CgpHmmLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer init")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer init: tf")
        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(start, f"Layer.init() start {run_id}", config.bench_path)
        super(CgpHmmLayer, self).__init__()
        self.nCodons = config.nCodons
        self.config = config

        append_time_ram_stamp_to_file(start, f"Layer.init() end  {run_id}", self.config.bench_path)

    def build(self, inputs):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer build")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer build: tf")

        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(start, f"Layer.build() start {run_id}", self.config.bench_path)
        self.C = CgpHmmCell(self.config) # init

        # self.C.build(input_shape) # build
        # this isnt needed for training but when calling the layer, then i need to build C manually, but it is then called
        # a second time when calling F
        self.F = tf.keras.layers.RNN(self.C, return_state = True, return_sequences = True) # F = forward ie the chain of cells C

        append_time_ram_stamp_to_file(start, f"Layer.build() end   {run_id}", self.config.bench_path)

    def call(self, inputs, training = False): # shape of inputs is None = batch, None = seqlen, 126 = emissions_size
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer call")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer call: tf")

        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(start, f"Layer.call() start {run_id}", self.config.bench_path)

        # todo: felix macht auch nochmal a und b
        self.C.init_cell()

        result = self.F(inputs) #  build and call of CgpHmmCell are called

        alpha_seq = result[0]
        inputs_seq = result[1]
        count_seq = result[2]
        alpha_state = result[3]
        loglik_state = result[4]
        count_state = result[5]

        if self.config.batch_begin_write_weights__layer_call_write_inputs:
            # os.system(f"rm {self.config['src_path']}/output/{self.config['nCodons']}codons/batch_begin_write_weights__layer_call_write_inputs/current_inputs.txt")
            # also remove the file at beginning of batch
            # out_inputs = tf.argmax(inputs, axis = 2)
            # out_inputs = [[int(base) for base in seq]for seq in out_inputs]
            # tf.print(json.dumps(out_inputs, outstream))
            os.system(f"mv       {self.config.src_path}/output/{self.config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/current_inputs.txt.temp {self.config.src_path}/output/{self.config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/current_inputs.txt")
            outstream = f"file://{self.config.src_path}/output/{self.config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/current_inputs.txt.temp"
            tf.print(inputs, summarize = -1, output_stream = outstream)

        if self.config.write_return_sequnces:
            outstream = f"file://./output/for_unit_tests/return_sequnces.txt"

            # tf.print("alpha_seq, inputs_seq, count_seq", output_stream = outstream)
            # tf.print(f"{tf.shape(count_seq)} {tf.shape(inputs_seq)} {tf.shape(alpha_seq)}", output_stream = outstream)
            for i in range(tf.shape(alpha_seq)[0]): # = batch_size
                if i != 0:
                    continue
                for j in range(tf.shape(alpha_seq)[1]): # = seq length
                    tf.print(count_seq[i,j], tf.argmax(inputs_seq[i,j]), alpha_seq[i,j], sep = ";", summarize = -1, output_stream = outstream)

        # squeeze removes dimensions of size 1, ie shape (1,3,2,1) -> (3,2)

        def my_loss(loglik_state):
            probs_to_be_punished = []
            loglik_mean = tf.reduce_mean(loglik_state)
            if self.config.check_assert:
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik_state)), [loglik_state],              name = "loglik_state_is_finite", summarize = self.config.assert_summarize)
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik_mean)),  [loglik_mean, loglik_state], name = "loglik_mean_is_finite",  summarize = self.config.assert_summarize)


            if training:
                # self.add_metric(loglik_mean, "loglik")
                self.add_metric(tf.math.reduce_max(self.C.init_kernel),"init_kernel_max")
                self.add_metric(tf.math.reduce_min(self.C.init_kernel),"init_kernel_min")

                self.add_metric(tf.math.reduce_max(self.C.transition_kernel),"transition_kernel_max")
                self.add_metric(tf.math.reduce_min(self.C.transition_kernel),"transition_kernel_min")

                self.add_metric(tf.math.reduce_max(self.C.emission_kernel),"emission_kernel_max")
                self.add_metric(tf.math.reduce_min(self.C.emission_kernel),"emission_kernel_min")

            use_reg = False
            if use_reg:
                # regularization
                # siehe Treffen_04
                alpha = 4 # todo: scale punishment for inserts with different factor?
                def add_reg(f, to):
                    if not self.config.call_type == 4:
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
                return tf.squeeze(-loglik_mean - alpha * reg_mean)
            else:
                return tf.squeeze(-loglik_mean)
        # end myloss()
        self.add_loss(my_loss(loglik_state))

        append_time_ram_stamp_to_file(start, f"Layer.call() end   {run_id}", self.config.bench_path)
        return loglik_state
