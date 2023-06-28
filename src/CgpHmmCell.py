#!/usr/bin/env python3
from Utility import append_time_ram_stamp_to_file

import os
import time
import tensorflow as tf
import numpy as np
import json


class CgpHmmCell(tf.keras.layers.Layer):
    def __init__(self, config):

        start = time.perf_counter()
        append_time_ram_stamp_to_file(f"Cell.__init__() start ", config.bench_path, start)

        # super(CgpHmmCell, self).__init__()
        super(CgpHmmCell, self).__init__()

        self.nCodons = config.nCodons

        self.alphabet_size = config.alphabet_size # without terminal symbol and without "left flank" symbol I
        self.order = config.order

        self.config = config
        self.state_size = [config.model.number_of_states, 1]


        append_time_ram_stamp_to_file(f"Cell.__init__() end ", self.config.bench_path, start)

    def build(self, shape):

        start = time.perf_counter()
        append_time_ram_stamp_to_file(f"Cell.build() start ", self.config.bench_path, start)

        # setting the initilizers
        if self.config.init_weights_from:
            print("init cell weights from ", self.config.init_weights_from)
            weights = self.read_weights_from_file(self.config.init_weights_from)
            A_initializer = tf.constant_initializer(weights[1])
            B_initializer = tf.constant_initializer(weights[2])

        elif self.config.use_constant_initializer:
            A_initializer = tf.constant_initializer(1)
            B_initializer = tf.constant_initializer(1)
        elif self.config.use_thesis_weights:
            # initial_weights_for_trainable_parameters = np.array(self.config.model.A_initial_weights_for_trainable_parameters, dtype = np.float32)
            A_initializer = tf.constant_initializer(self.config.model.A_initial_weights_for_trainable_parameters)
            B_initializer = tf.constant_initializer(self.config.model.B_initial_weights_for_trainable_parameters)
        else:
            A_initializer="random_normal"
            B_initializer="random_normal"


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
            self.config.model.export_to_dot_and_png(self.A_kernel, self.B_kernel)
            exit()
        append_time_ram_stamp_to_file(f"Cell.build() end ", self.config.bench_path, start)

################################################################################
    @property
    def I(self):
        return self.config.model.I()

    @property
    def A(self):
        return self.config.model.A(self.A_kernel)

    @property
    def B(self):
        return self.config.model.B(self.B_kernel)

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
        # returns inputs * B
        if self.config.trace_verbose:
            print("cell.get_E()")

        if self.config.B_is_dense:
            B = tf.matmul(inputs, self.B)
        else:
            B = tf.sparse.sparse_dense_matmul(inputs, self.B)
        if self.config.dtype == np.float64:
            B = tf.cast(B, np.float64)
        return B
################################################################################
    def get_R(self, old_forward, init = False):
        # returns old_forward * A
        # init = True means that this is the first call of the cell
        # use initial case of forward variable
        if self.config.trace_verbose:
            print("cell.get_R()")
        if init:
            return tf.cast(tf.math.log(old_forward), self.config.dtype)

        # if add_epsilon_to_z:
        #     Z_i = tf.math.add(Z_i, add_epsilon_to_z)
        def mul(a, b):
            if self.config.A_is_sparse:
                return tf.sparse.sparse_dense_matmul(a, b)
            else:
                return tf.matmul(a, b)

        m_alpha = tf.reduce_max(old_forward, axis = 1, keepdims = True)
        R = tf.math.log(mul(tf.math.exp(old_forward - m_alpha) + self.config.R_epsilon, self.A)) + m_alpha

        if self.config.dtype == np.float64:
            R = tf.cast(R, np.float64)
        return R
################################################################################
    def calc_new_cell_state(self, E, R):
        # calculates alpha and loglik
        if self.config.trace_verbose:
            print("cell.calc_new_cell_state()")
        E_epsilon = tf.cast(self.config.E_epsilon, self.config.dtype)
        l_epsilon = tf.cast(self.config.l_epsilon, self.config.dtype)
        alpha = tf.math.log(E + E_epsilon) + R
        m_alpha = tf.math.reduce_max(alpha, axis = 1, keepdims = True)
        loglik = tf.math.log(tf.reduce_sum(tf.math.exp(alpha - m_alpha) + l_epsilon, axis = 1, keepdims = True)) + m_alpha

        return alpha, loglik
#################################################################################
    def call(self, inputs, states, init = False, training = None):
        if self.config.trace_verbose:
            print("cell.call()")
        old_forward, old_loglik = states

        self.assert_check_beginning_of_call(old_forward, old_loglik)
        self.verbose_beginning_of_call(inputs, old_forward, old_loglik)

        E = self.get_E(inputs)

        R = self.get_R(old_forward, init = init)

        alpha, loglik = self.calc_new_cell_state(E, R)

        self.assert_check_end_of_call(alpha, loglik)
        self.verbose_end_of_call(E, R, alpha, loglik)

        states = [alpha, loglik]
        # nothing returned for the return sequnces
        return [], states
################################################################################
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # calculated the initial state of the cell
        # that is used for the first call of the cell
        # first means for the first emission of the sequence
        if self.config.trace_verbose:
            print("cell.get_initial_state()")
        old_forward = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, self.config.model.number_of_states-1))], axis = 1)
        loglik = tf.zeros((batch_size, 1), dtype=self.config.dtype)
        S = [old_forward, loglik]
        return S
################################################################################
    def assert_check_beginning_of_call(self, old_forward, old_loglik):
        if self.config.check_assert:
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.A_kernel)), [self.A_kernel], name = "A_kernel_beginning_of_cell", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.B_kernel)), [self.B_kernel], name = "B_kernel_beginning_of_cell", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.A_dense)),  [self.A_dense],  name = "A_dense_beginning_of_call",  summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.B_dense)),  [self.B_dense],  name = "B_dense_beginning_of_call",  summarize = self.config.assert_summarize)
            tf.debugging.Assert(not tf.math.reduce_any(tf.math.is_nan(old_forward)),   [old_forward],   name = "old_forward_nan_beginning_of_call",              summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(old_loglik)),        [old_loglik], name = "old_loglik_finite_beginning_of_call",  summarize = self.config.assert_summarize)
################################################################################
    def assert_check_end_of_call(self, alpha, loglik):
        if self.config.check_assert:
            tf.debugging.Assert(not tf.math.reduce_any(tf.math.is_nan(alpha)), [alpha], name = "alpha_nan_end_of_call", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik)), [loglik, [123456789], alpha], name = "loglik_finite_end_of_call", summarize = self.config.assert_summarize)

################################################################################
    def verbose_beginning_of_call(self, inputs, old_forward, old_loglik):
        if self.config.verbose:
            if self.config.print_to_file:
                outstream = f"file://{self.config.out_path}/verbose/{self.nCodons}codons.txt"
                if not os.path.exists(f"{self.config.out_path}/verbose/"):
                    os.system(f"mkdir -p {self.config.out_path}/verbose/")
            else:
                import sys
                outstream = sys.stdout
        def verbose_print(string, data):
            if self.config.verbose:
                tf.print(string, tf.shape(data), output_stream = outstream, sep = ";")
                if self.config.verbose == 2:
                    tf.print(">" + string, data, output_stream = outstream, sep = ";", summarize=-1)

        if self.config.verbose:
            verbose_print("A", self.A)
            verbose_print("B", self.B)
            verbose_print("inputs", inputs)
            verbose_print("old_forward", old_forward)
            verbose_print("old_loglik", old_loglik)
################################################################################
    def verbose_end_of_call(self, E, R, alpha, loglik):
        if self.config.verbose:
            if self.config.print_to_file:
                outstream = f"file://{self.config.out_path}/verbose/{self.nCodons}codons.txt"
            else:
                import sys
                outstream = sys.stdout
        def verbose_print(string, data):
            if self.config.verbose:
                tf.print(string, tf.shape(data), output_stream = outstream, sep = ";")
                if self.config.verbose == 2:
                    tf.print(">" + string, data, output_stream = outstream, sep = ";", summarize=-1)
        if self.config.verbose:
            verbose_print("E", E)
            verbose_print("R", R)
            verbose_print("alpha", alpha)
            verbose_print("loglik", loglik)
################################################################################
    def assert_E_R_alpha(self, E, R, alpha):
        if self.config.check_assert:
            tf.debugging.Assert(not tf.math.reduce_any(tf.math.is_nan(R)),     [R], name = "R_finite", summarize = self.config.assert_summarize)
            tf.debugging.Assert(not tf.math.reduce_any(tf.math.is_nan(E)),     [E], name = "E_finite", summarize = self.config.assert_summarize)
            tf.debugging.Assert(not tf.math.reduce_any(tf.math.is_nan(alpha)), [alpha], name = "E_finite", summarize = self.config.assert_summarize)
################################################################################
    def write_weights_to_file(self, path): # is this sufficient to get reproducable behaviour?
        ik = []
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
