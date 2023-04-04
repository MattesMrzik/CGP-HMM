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

        # @tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
        # def l1_reg(weight_matrix):
        #    return 0.01 * tf.math.reduce_sum(tf.math.abs(weight_matrix))

        self.C = CgpHmmCell(self.config) # init

        # self.C.build(input_shape) # build
        # this isnt needed for training but when calling the layer, then i need to build C manually, but it is then called
        # a second time when calling F
        self.F = tf.keras.layers.RNN(self.C, return_state = True) # F = forward ie the chain of cells C
        append_time_ram_stamp_to_file(f"Layer.build() end   {run_id}", self.config.bench_path, start)

    def call(self, inputs, training = False): # shape of inputs is None = batch, None = seqlen, 126 = emissions_size
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer call")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer call: tf")

        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(f"Layer.call() start {run_id}", self.config.bench_path, start)

        # todo: felix macht auch nochmal a und b

        # result = self.F(inputs, initial_state = self.C.get_initial_state(batch_size=tf.shape(inputs)[0])) #  build and call of CgpHmmCell are called
        initial_state = self.C.get_initial_state(batch_size=tf.shape(inputs)[0])
        # print("initial_state =", initial_state)
        _, result_first_init_call = self.C(inputs[:,0], initial_state, init = True, training = training)
        result = self.F(inputs[:,1:], initial_state = result_first_init_call, training = training)
        # result = self.F(inputs[:,1:,:])
        # result = self.F(inputs)
        # i think this is an artefact from a previous version, where i would sometimes return an additional value. I think this can be unwrapped right away on the preceeding line
        scale_count_state = 0
        alpha_state = result[0]
        loglik_state = result[1]
        if self.config.scale_with_conditional_const:
            # print("scale_count_state")
            scale_count_state = result[2]

        if self.config.batch_begin_write_weights__layer_call_write_inputs:
            # os.system(f"rm {self.config['out_path']}/output/{self.config['nCodons']}codons/batch_begin_write_weights__layer_call_write_inputs/current_inputs.txt")
            # also remove the file at beginning of batch
            # out_inputs = tf.argmax(inputs, axis = 2)
            # out_inputs = [[int(base) for base in seq]for seq in out_inputs]
            # tf.print(json.dumps(out_inputs, outstream))
            current_inputs = f"{self.config.out_path}/output/{self.config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/current_inputs.txt"
            current_inputs_temp = f"{current_inputs}.temp"

            os.system(f"mv {current_inputs_txt_temp} {current_inputs}")
            outstream = f"file://{current_inputs_temp}"
            tf.print(inputs, summarize = -1, output_stream = outstream)

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
            loglik_mean = tf.reduce_mean(loglik_state + tf.math.log(tf.reduce_sum(alpha_state + self.config.my_scale_log_epsilon, axis = -1)))
        #=========> getting loglik_mean done <=================================#


        if self.config.check_assert:
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik_state)), [loglik_state],              name = "loglik_state_is_finite", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik_mean)),  [loglik_mean, loglik_state], name = "loglik_mean_is_finite",  summarize = self.config.assert_summarize)


        # or do it like this
        # @tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
        # def l1_reg(weight_matrix):
        #    return 0.01 * tf.math.reduce_sum(tf.math.abs(weight_matrix))


        # TODO: move the reg also to the model


        if self.config.regularize:
            start = time.perf_counter()
            run_id = randint(0,100)
            append_time_ram_stamp_to_file(f"Layer.call() REG get indices and A_dense start {run_id}", self.config.bench_path, start)
            A_indices_begin_inserts = self.config.model.A_indices_begin_inserts
            A_indices_continue_inserts = self.config.model.A_indices_continue_inserts
            A_indices_deletes = self.config.model.A_indices_deletes
            A_dense = self.C.A_dense
            append_time_ram_stamp_to_file(f"Layer.call() REG get indices end {run_id}", self.config.bench_path, start)

            start = time.perf_counter()
            run_id = randint(0,100)
            append_time_ram_stamp_to_file(f"Layer.call() REG start {run_id}", self.config.bench_path, start)

            reg = 0
            reg += tf.reduce_sum(self.config.inserts_punish_factor * tf.math.log(1 - tf.gather_nd(A_dense, A_indices_begin_inserts)))
            reg += tf.reduce_sum(self.config.inserts_punish_factor * tf.math.log(1 - tf.gather_nd(A_dense, A_indices_continue_inserts)))
            reg += tf.reduce_sum(self.config.deletes_punish_factor * tf.math.log(1 - tf.gather_nd(A_dense, A_indices_deletes)))
            append_time_ram_stamp_to_file(f"Layer.call() REG end {run_id}", self.config.bench_path, start)

        # TODO: sollte der reg term normalisert werden auf die anzahl der regularisierten terme?
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
        return loglik_state
