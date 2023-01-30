#!/usr/bin/env python3

from Utility import append_time_ram_stamp_to_file

from Utility import tfprint


import time
from random import randint
from itertools import product
import tensorflow as tf
import numpy as np
import json


class CgpHmmCell(tf.keras.layers.Layer):
# class CgpHmmCell(tf.keras.layers.Layer):
    def __init__(self, config):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell init")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell init: tf")

        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(start, f"Cell.__init__() start {run_id}", config.bench_path)

        # super(CgpHmmCell, self).__init__()
        super(CgpHmmCell, self).__init__()

        self.nCodons = config.nCodons

        self.alphabet_size = config.alphabet_size # without terminal symbol and without "papped left flank" symbol
    # order = 0 -> emission prob depends only on current emission
        self.order = config.order

        self.config = config

        self.state_size = [config.model.number_of_states, 1, 1]

        # self.indices_for_weights_A = self.get_indices_for_weights_for_A()
        # # vielleich einfach den consts auch ein weigt geben, welches durch softmax eh dann 1 wird
        # # dann hat der gradient zwar mehr einträge, aber es muss ein concat der values und indices gemacht werden,
        #
        # self.indices_for_weights_B = self.get_indices_for_weights_for_B()
        # self.indices_for_constants_B = self.get_indices_for_constants_for_B()


        append_time_ram_stamp_to_file(start, f"Cell.__init__() end   {run_id}", self.config.bench_path)


        # this is for shared parameter vesion which ran slow
        # s = 1 # ig5'
        # s += 1 # delete
        # s += (self.nCodons + 1) * 2 # enter/exit insert
        # s += self.nCodons # enter codon
        # s += 1 # exit last codon
        #
        # return(s)

    def build(self, s):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell build")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell build: tf")

        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(start, f"Cell.build() start {run_id}", self.config.bench_path)

        # setting the initilizers
        if self.config.get_gradient_for_current_txt or self.config.init_weights_from_txt:
            with open(f"{self.config.src_path}/output/{self.config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/current_I.json") as file:
                weights_I = np.array(json.load(file))
                I_initializer = tf.constant_initializer(weights_I)
            with open(f"{self.config.src_path}/output/{self.config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/current_A.json") as file:
                weights_A = np.array(json.load(file))
                A_initializer = tf.constant_initializer(weights_A)
            with open(f"{self.config.src_path}/output/{self.config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/current_B.json") as file:
                weights_B = np.array(json.load(file))
                B_initializer = tf.constant_initializer(weights_B)
        # elif self.config["get_gradient_from_saved_model_weights"] and "model" in self.config:
        elif self.config.get_gradient_from_saved_model_weights and "weights" in self.config.__dict__:
            # weights = self.config["model"].get_weights()
            # this causes error,
            # try if txt is sufficient to get nan as gradient

            # they seem to get the same results as current.txt
            weights = self.config.weights
            I_initializer = tf.constant_initializer(weights[0])
            A_initializer = tf.constant_initializer(weights[1])
            B_initializer = tf.constant_initializer(weights[2])
        elif self.config.use_constant_initializer:
            I_initializer = tf.constant_initializer(1)
            A_initializer = tf.constant_initializer(1)
            B_initializer = tf.constant_initializer(1)
        else:
            I_initializer="random_normal"
            A_initializer="random_normal"
            B_initializer="random_normal"

        # setting the initilizers done:
        self.I_kernel = self.add_weight(shape = (self.config.model.I_kernel_size(),),
                                        initializer = I_initializer,
                                        dtype = self.config.dtype,
                                        trainable = True, name = "I_kernel")

        self.A_kernel = self.add_weight(shape = (self.config.model.A_kernel_size(),),
                                        initializer = A_initializer,
                                        dtype = self.config.dtype,
                                        trainable = True, name = "A_kernel")

        self.B_kernel = self.add_weight(shape = (self.config.model.B_kernel_size(),),
                                        initializer = B_initializer,
                                        dtype = self.config.dtype,
                                        trainable = True, name = "B_kernel")

        visualize_after_build = False
        if visualize_after_build:
            import WriteData
            import os
            WriteData.write_to_file(self.A_dense, f"{self.config.src_path}/output/{self.nCodons}codons/A.{self.nCodons}codons.txt")
            WriteData.write_to_file(tf.transpose(self.B_dense), f"{self.config.src_path}/output/{self.nCodons}codons/B.{self.nCodons}codons.txt")
            WriteData.write_to_file(self.I_dense, f"{self.config.src_path}/output/{self.nCodons}codons/I.{self.nCodons}codons.txt")

            os.system(f"./Visualize.py -c {self.config.nCodons} -o {self.config.order} -t")
            exit(1)
        append_time_ram_stamp_to_file(start, f"Cell.build() end   {run_id}", self.config.bench_path)

################################################################################
    # this might be a possible alternative for the count variable in cell.call()
    # currently, count == 0 is checked to determine whether the first symbol
    # of the seq is processed
    # but using this bool didnt work, bc it was always set to False
    # in the first call, before the actual graph is executed
    def init_cell(self):
        self.inita = True

################################################################################
    @property
    def I(self):
        return self.config.model.I(self.I_kernel)

    @property
    def A(self):
        return self.config.model.A(self.A_kernel)

    @property
    def B(self):
        return self.config.model.B(self.B_kernel)

    # these transformation to the dense form are used in the NaN asserts
    @property
    def I_dense(self):
        return self.I

    @property
    def A_dense(self):
        if self.config.A_is_sparse:
            return tf.sparse.to_dense(self.A)
        return self.A

    @property
    def B_dense(self):
        if self.config.B_is_sparse:
            return tf.sparse.to_dense(self.B)
        return self.B
################################################################################
################################################################################
    def get_E(self, inputs):
        if self.config.B_is_dense:
            return tf.matmul(inputs, self.B)
        return tf.sparse.sparse_dense_matmul(inputs, self.B)

    def get_R(self, old_forward, init = False):
        if init:
            return self.I, tf.cast(1.0, dtype = self.config.dtype) # bc return must be same in main and off branch, must be != 0 bc assert check z != 0

        Z_i_minus_1 = tf.reduce_sum(old_forward, axis = 1, keepdims = True)
        scaled_forward = old_forward / Z_i_minus_1
        # if add_epsilon_to_z:
        #     Z_i_minus_1 = tf.math.add(Z_i_minus_1, add_epsilon_to_z)
        if self.config.A_is_sparse:
            R = tf.sparse.sparse_dense_matmul(scaled_forward, self.A)
        else:
            R = tf.matmul(scaled_forward, self.A)

        return R, Z_i_minus_1

################################################################################
    def call(self, inputs, states, training = None): # call_sparse
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell call_sparse")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell call_sparse: tf")

        old_forward, old_loglik, count = states

        # TODO: make this a bool
        # print("optype", self.A_dense.op.type)
        count = tf.math.add(count, 1)

        if self.config.check_assert:

            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.I_kernel)), [self.I_kernel, self.A_kernel, self.B_kernel, count[0,0]], name = "I_kernel_beginning_of_cell", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.A_kernel)), [self.A_kernel], name = "A_kernel_beginning_of_cell", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.B_kernel)), [self.B_kernel], name = "B_kernel_beginning_of_cell", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.I_dense)),  [self.I_dense, count[0,0]], name = "I_dense_beginning_of_call", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.A_dense)),  [self.A_dense, old_loglik, old_forward, count[0,0]], name = "A_dense_beginning_of_call", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.B_dense)),  [self.B_dense, count[0,0]], name = "B_dense_beginning_of_call", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(old_forward)),  [old_forward,   count[0,0]], name = "old_forward",               summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(old_loglik)),   [old_loglik,    count[0,0]], name = "old_loglik",                summarize = self.config.assert_summarize)

        run_id = randint(0,100)

        verbose = self.config.verbose

        if verbose:
            if self.config.print_to_file:
                outstream = f"file://{self.config.src_path}/verbose/{self.nCodons}codons.txt"
            else:
                import sys
                outstream = sys.stdout
        def verbose_print(string, data):
            if verbose:
                tf.print(count[0,0], run_id, string, tf.shape(data), output_stream = outstream, sep = ";")
                if verbose == 2:
                    tf.print(count[0,0], run_id, ">" + string, data, output_stream = outstream, sep = ";", summarize=-1)

        E = self.get_E(inputs)

        if verbose:
            verbose_print("count", count[0,0])
            if count[0,0] == 1:
                verbose_print("A", self.A)
                verbose_print("B", self.B)
            verbose_print("inputs", inputs)
            verbose_print("old_forward", old_forward)
            verbose_print("old_loglik", old_loglik)
            verbose_print("E", E)

        R, Z_i_minus_1 = self.get_R(old_forward, init = count[0,0] == 1)

        if self.config.check_assert:
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(Z_i_minus_1)),  [Z_i_minus_1, count[0,0]],  name = "z_finite",  summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(Z_i_minus_1 != 0),                [Z_i_minus_1, count[0,0]],  name = "z_nonzero", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(R)),            [R, count[0,0]],            name = "R_finite",  summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(E)),            [E, count[0,0]],            name = "E_finite",  summarize = self.config.assert_summarize)
        alpha = E * R # batch_size * state_size

        # keepsdims is true such that shape of result is (32,1) rather than (32,)
        # loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis = -1, keepdims = True, name = "loglik"))
        loglik = tf.math.add(old_loglik, tf.math.log(tf.reduce_sum(alpha, axis = -1, keepdims = True)), name = "loglik")

        if self.config.check_assert:
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(alpha)),  [alpha, count[0,0], alpha],  name = "alpha",         summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik)), [loglik, count[0,0],[123456789], alpha, [123456789], E,[123456789],  R, [123456789], inputs], name = "loglik_finite", summarize = self.config.assert_summarize)
            # i think this should be allowed since sum across alpha can be 1, then log is 0, which is fine
            # tf.debugging.Assert(tf.math.reduce_all(loglik != 0),                     [loglik, count[0,0]],       name = "loglik_nonzero",            summarize = -1)

            #todo also check if loglik is zero, bc then a seq should be impossible to be emitted, which shouldnt be the case

        if verbose:
            verbose_print("R", R)
            verbose_print("forward", alpha)
            verbose_print("loglik", loglik)

        return [alpha, inputs, count], [alpha, loglik, count]
