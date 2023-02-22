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


from Utility import append_time_stamp_to_file
from Utility import append_time_ram_stamp_to_file

from CgpHmmCell import CgpHmmCell



class CgpHmmLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer init")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer init: tf")
        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(f"Layer.init() start {run_id}", config.bench_path, start)
        super(CgpHmmLayer, self).__init__()
        self.nCodons = config.nCodons
        self.config = config

        append_time_ram_stamp_to_file(f"Layer.init() end  {run_id}", self.config.bench_path, start)

    def build(self, inputs):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer build")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer build: tf")

        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(f"Layer.build() start {run_id}", self.config.bench_path, start)
        self.C = CgpHmmCell(self.config) # init

        # self.C.build(input_shape) # build
        # this isnt needed for training but when calling the layer, then i need to build C manually, but it is then called
        # a second time when calling F
        self.F = tf.keras.layers.RNN(self.C, return_state = True, return_sequences = self.config.return_seqs) # F = forward ie the chain of cells C

        append_time_ram_stamp_to_file(f"Layer.build() end   {run_id}", self.config.bench_path, start)

    def call(self, inputs, training = False): # shape of inputs is None = batch, None = seqlen, 126 = emissions_size
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer call")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer call: tf")

        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(f"Layer.call() start {run_id}", self.config.bench_path, start)

        # todo: felix macht auch nochmal a und b
        self.C.init_cell()

        result = self.F(inputs) #  build and call of CgpHmmCell are called

        # i think this is an artefact from a previous version, where i would sometimes return an additional value. I think this can be unwrapped right away on the preceeding line
        scale_count_state = 0
        if self.config.return_seqs:
            alpha_seq = result[0]
            inputs_seq = result[1]
            count_seq = result[2]
            alpha_state = result[3]
            loglik_state = result[4]
            count_state = result[5]
            if self.config.scale_with_conditional_const:
                # print("scale_count_state")
                scale_count_state = result[6]
        else:
            alpha_state = result[0]
            loglik_state = result[1]
            count_state = result[2]
            if self.config.scale_with_conditional_const:
                # print("scale_count_state")
                scale_count_state = result[3]
        # alpha_seq = result[0]
        # inputs_seq = result[1]
        # count_seq = result[2]
        # alpha_state = result[3]
        # loglik_state = result[4]
        # count_state = result[5]
        # if self.config.scale_with_conditional_const:
        #     tf.print("scale_count_state =", scale_count_state, summarize = -1)
        # if training:
        #
        #     # if a mask is used this has to be adjusted
        #     if self.config.scale_with_conditional_const:
        #         pass
        #     elif self.config.scale_with_const:
        #         tf.print("<asdfwesbgfdd")
        #         # loglik_state = tf.math.log(self.config.scale_with_const) - tf.math.log(tf.shape(inputs)[1]) - tf.math.log(self.config.scale_with_const)


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

        #=========> getting loglik_mean <======================================#
        if self.config.scale_with_const:
            length = tf.cast(tf.shape(inputs)[1], dtype=tf.float32)
            # loglik_mean = tf.reduce_mean(tf.math.log(loglik_state) - length * tf.math.log(self.config.scale_with_const))
            loglik_mean = tf.reduce_mean(loglik_state)

        elif self.config.scale_with_conditional_const:
            # print("scale_count_state =", scale_count_state)
            scale_count_state = tf.cast(scale_count_state, dtype=tf.float32) #  das sind ja eigentlich ints. kann da überhaupt eine ableitung gebildet werden?
            # loglik_mean = tf.reduce_mean(tf.math.log(tf.reduce_sum(alpha_state, axis = 1, keepdims = True)) - scale_count_state * tf.math.log(10.0))
            loglik_mean = tf.reduce_mean(loglik_state)

        elif self.config.felix:
            loglik_mean = tf.reduce_mean(loglik_state)

        elif self.config.logsumexp:
            loglik_mean = tf.reduce_mean(loglik_state)

        else:
            loglik_mean = tf.reduce_mean(loglik_state + tf.math.log(tf.reduce_sum(alpha_state + self.config.epsilon_my_scale_log, axis = -1)))
        #=========> getting loglik_mean done <=================================#



        if self.config.check_assert:
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik_state)), [loglik_state],              name = "loglik_state_is_finite", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik_mean)),  [loglik_mean, loglik_state], name = "loglik_mean_is_finite",  summarize = self.config.assert_summarize)



        #     # regularization
        #     # siehe Treffen_04
        #     alpha = 4 # todo: scale punishment for inserts with different factor?
        #     def add_reg(f, to):
        #         probs_to_be_punished.append(tf.math.log(1 - \
        #                                     self.C.A_dense[self.config.model.str_to_state_id(f, self.nCodons), \
        #                                                    self.config.model.str_to_state_id(to, self.nCodons)]))
        #
        #     # deletes to be punished
        #     for i in range(1, self.C.nCodons):
        #         add_reg("stG", f"c_{i},0")
        #     add_reg("stG", "stop1")
        #     for i in range(self.C.nCodons - 1):
        #         for j in range(i + 2, self.C.nCodons):
        #             add_reg(f"c_{i},2", f"c_{j},0")
        #         add_reg(f"c_{i},2", "stop1")

        #     # inserts to be punished
        #     add_reg("stG", "i_0,0")
        #     for i in range(self.C.nCodons):
        #         add_reg(f"c_{i},2", f"i_{i+1},0")
        #
        #     reg_mean = sum(probs_to_be_punished) / len(probs_to_be_punished)
        #
        #     if loglik_mean < 0 and reg_mean >0:
        #         tf.print("not same sign")
        #     if loglik_mean > 0 and reg_mean <0:
        #         tf.print("not same sign")
        #     self.add_loss(tf.squeeze(-loglik_mean - alpha * reg_mean))
        # else:

        reg = 0
        if self.config.regularize:
            punish_beginning_inserts = True
            if punish_beginning_inserts:
                for index_tuple in self.config.model.A_indices_begin_inserts:
                    reg += self.config.inserts_punish_factor * tf.math.log(1 - self.C.A_dense[index_tuple]) # this shouldnt be log(0) since parameters are punished if near 1
                for index_tuple in self.config.model.A_indices_continue_inserts:
                    reg += self.config.inserts_punish_factor * tf.math.log(1 - self.C.A_dense[index_tuple])
            punish_deletes = True
            if punish_deletes:
                for index_tuple in self.config.model.A_indices_deletes:
                    # TODO: longer deletes should be punihsed more
                    reg += self.config.deletes_punish_factor * tf.math.log(1 - self.C.A_dense[index_tuple])


        # TODO: sollte der reg term normalisert werden auf die anzahl der regularisierten terme?
        if self.config.regularize:
            self.add_metric(reg, "reg")
            alpha = 1
            self.add_loss(tf.squeeze(-loglik_mean - alpha * reg))
        else:
            self.add_loss(tf.squeeze(-loglik_mean))

        if training:
            # self.add_metric(loglik_mean, "loglik")

            # self.add_metric(tf.math.reduce_max(self.C.I_kernel),"ik_max")
            # self.add_metric(tf.math.reduce_min(self.C.I_kernel),"ik_min")

            # self.add_metric(tf.math.reduce_max(self.C.I_dense),"I_max")
            # self.add_metric(tf.math.reduce_min(self.C.I_dense),"I_min")

            self.add_metric(tf.math.reduce_max(self.C.A_kernel),"tk_max")
            self.add_metric(tf.math.reduce_min(self.C.A_kernel),"tk_min")

            # self.add_metric(tf.math.reduce_max(self.C.A_dense),"A_max")
            # self.add_metric(tf.math.reduce_min(self.C.A_dense),"A_min")

            self.add_metric(tf.math.reduce_max(self.C.B_kernel),"ek_max")
            self.add_metric(tf.math.reduce_min(self.C.B_kernel),"ek_min")

            # self.add_metric(tf.math.reduce_max(self.C.B_dense),"B_max")
            # self.add_metric(tf.math.reduce_min(self.C.B_dense),"B_min")


        append_time_ram_stamp_to_file(f"Layer.call() end   {run_id}", self.config.bench_path, start)
        if self.config.return_seqs:
            return loglik_state, alpha_seq
        else:
            return loglik_state
