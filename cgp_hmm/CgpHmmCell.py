#!/usr/bin/env python3

from Utility import append_time_ram_stamp_to_file

from Utility import tfprint

import os
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
            path = f"{self.config.src_path}/output/{self.config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/"
            weights = self.read_weights_from_file(path)
            I_initializer = tf.constant_initializer(weights[0])
            A_initializer = tf.constant_initializer(weights[1])
            B_initializer = tf.constant_initializer(weights[2])
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
            config.model.export_to_dot_and_png(A_kernel, B_kernel)
            exit()
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
            return self.I

        # if add_epsilon_to_z:
        #     Z_i = tf.math.add(Z_i, add_epsilon_to_z)
        if self.config.A_is_sparse:
            R = tf.sparse.sparse_dense_matmul(old_forward, self.A)
        else:
            R = tf.matmul(old_forward, self.A)

        return R

################################################################################

    def call(self, inputs, states, training = None): # how often is the graph for this build?
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
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.A_dense)),  [self.A_dense, old_forward, count[0,0]], name = "A_dense_beginning_of_call", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.B_dense)),  [self.B_dense,  count[0,0]], name = "B_dense_beginning_of_call", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(old_forward)),   [old_forward,   count[0,0]], name = "old_forward",              summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(old_loglik)),[old_loglik,        count[0,0]], name = "old_loglik",               summarize = self.config.assert_summarize)

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

        R = self.get_R(old_forward, init = count[0,0] == 1)

        if self.config.check_assert:
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(R)),            [R, count[0,0]],            name = "R_finite",  summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(E)),            [E, count[0,0]],            name = "E_finite",  summarize = self.config.assert_summarize)

        unscaled_alpha = E * R # batch_size * state_size

        if self.config.scale_with_const:
            scaled_alpha = unscaled_alpha * self.config.scale_with_const
            loglik = tf.reduce_sum(scaled_alpha, axis = 1, keepdims = True)
            Z_i = loglik # just to have a value to be returned
        elif self.config.scale_with_conditional_const:
            Z_i = tf.reduce_sum(unscaled_alpha, axis = 1, keepdims = True)
            Z_i = tf.cast(Z_i < 0.1, dtype = self.config.dtype)
            Z_i *= 9
            Z_i += 1
            scaled_alpha = unscaled_alpha * Z_i
            loglik = tf.reduce_sum(scaled_alpha, axis = -1, keepdims = True, name = "likelihood")
        elif self.config.felix:
            if count[0,0] != 1:
                Z_i = tf.reduce_sum(old_forward, axis = 1, keepdims = True, name = "felix_z")
                scaled_alpha = tf.math.divide(unscaled_alpha, Z_i)
            else:
                scaled_alpha = unscaled_alpha
                Z_i = unscaled_alpha
            # scaled_alpha = unscaled_alpha
            loglik = tf.math.add(old_loglik, tf.math.log(tf.reduce_sum(scaled_alpha, axis = -1)), name = "loglik")
        else:
            Z_i = tf.reduce_sum(unscaled_alpha, axis = 1, keepdims = True, name = "my_z")
            loglik = tf.math.add(old_loglik, tf.math.log(Z_i), name = "loglik")
            scaled_alpha = unscaled_alpha / Z_i

        # keepsdims is true such that shape of result is (32,1) rather than (32,)
        # loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis = -1, keepdims = True, name = "loglik"))

        if self.config.check_assert:
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(scaled_alpha)),  [scaled_alpha, count[0,0]],  name = "alpha",         summarize = self.config.assert_summarize)

            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik)), [loglik, count[0,0],[123456789], scaled_alpha, [123456789], E,[123456789],  R, [123456789], inputs], name = "loglik_finite", summarize = self.config.assert_summarize)
            # i think this should be allowed since sum across alpha can be 1, then log is 0, which is fine
            # tf.debugging.Assert(tf.math.reduce_all(loglik != 0),                     [loglik, count[0,0]],       name = "loglik_nonzero",            summarize = -1)

            #todo also check if loglik is zero, bc then a seq should be impossible to be emitted, which shouldnt be the case

        if verbose:
            verbose_print("R", R)
            verbose_print("forward", scaled_alpha)
            verbose_print("loglik", loglik)

        return [scaled_alpha, inputs, count, Z_i], [scaled_alpha, loglik, count]

################################################################################

    def write_weights_to_file(self, path): # is this sufficient to get reproducable behaviour?
        ik = [float(x) for x in self.I_kernel.numpy()]
        ak = [float(x) for x in self.A_kernel.numpy()]
        bk = [float(x) for x in self.B_kernel.numpy()]

        if not os.path.exists(path):
            os.system(f"mkdir -p {path}")

        with open(f"{path}/I_kernel.json", "w") as file:
            json.dump(ik, file)
        with open(f"{path}/A_kernel.json", "w") as file:
            json.dump(ak, file)
        with open(f"{path}/B_kernel.json", "w") as file:
            json.dump(bk, file)

    def read_weights_from_file(self, path):
        with open(f"{path}/I_kernel.json") as file:
            weights_I = np.array(json.load(file))
        with open(f"{path}/A_kernel.json") as file:
            weights_A = np.array(json.load(file))
        with open(f"{path}/B_kernel.json") as file:
            weights_B = np.array(json.load(file))
        return weights_I, weights_A, weights_B
