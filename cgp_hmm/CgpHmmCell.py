#!/usr/bin/env python3

from Utility import append_time_ram_stamp_to_file
from Utility import description_to_state_id
from Utility import state_id_to_description

from Utility import tfprint

from Utility import get_indices_for_weights_for_A
from Utility import get_indices_for_constants_for_A
from Utility import get_indices_for_weights_for_B
from Utility import get_indices_for_constants_for_B
from Utility import emissions_state_size
from Utility import n_emission_columns_in_B

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

        self.state_size = [self.number_of_states, 1, 1]

        self.indices_for_weights_A = config.indices_for_weights_A# if "indices_for_weights_A" in config else get_indices_for_weights_for_A(config)
        # vielleich einfach den consts auch ein weigt geben, welches durch softmax eh dann 1 wird
        # dann hat der gradient zwar mehr einträge, aber es muss ein concat der values und indices gemacht werden,
        self.indices_for_constants_A = config.indices_for_constants_A #if "indices_for_constants_A" in config else get_indices_for_constants_for_A(config)
        self.indices_for_A = config.indices_for_A


        self.indices_for_weights_B = config.indices_for_weights_B#if "indices_for_weights_B" in config else get_indices_for_weights_for_B(config)
        self.indices_for_constants_B = config.indices_for_constants_B# if "indices_for_constants_B" in config else get_indices_for_constants_for_B(config)
        self.indices_for_B = config.indices_for_B

        self.indices_for_I = config.indices_for_I

        # self.indices_for_weights_A = self.get_indices_for_weights_for_A()
        # # vielleich einfach den consts auch ein weigt geben, welches durch softmax eh dann 1 wird
        # # dann hat der gradient zwar mehr einträge, aber es muss ein concat der values und indices gemacht werden,
        # self.indices_for_constants_A = self.get_indices_for_constants_for_A()
        #
        # self.indices_for_weights_B = self.get_indices_for_weights_for_B()
        # self.indices_for_constants_B = self.get_indices_for_constants_for_B()


        append_time_ram_stamp_to_file(start, f"Cell.__init__() end   {run_id}", self.config.bench_path)

    @property
    def number_of_states(self):
        # ig 5'
        number_of_states = 1
        # start
        number_of_states += 3
        # codons
        number_of_states += 3 * self.nCodons
        # codon inserts
        number_of_states += 3 * (self.nCodons + 1)
        # stop
        number_of_states += 3
        # ig 3'
        number_of_states += 1
        # terminal
        number_of_states += 1

        return number_of_states


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
        else:
            I_initializer="random_normal"
            A_initializer="random_normal"
            B_initializer="random_normal"

        # setting the initilizers done

        # ---> full model <---
        if self.config.call_type == 4:
            self.init_kernel = self.add_weight(shape = (self.number_of_states,),
                                               initializer = I_initializer,
                                               dtype = self.config.dtype,
                                               trainable = True, name = "init_kernel")
            self.transition_kernel = self.add_weight(shape = (self.number_of_states,self.number_of_states),
                                                     initializer = A_initializer,
                                                     dtype = self.config.dtype,
                                                     trainable = True, name = "transition_kernel")

            self.emission_kernel = self.add_weight(shape = (n_emission_columns_in_B(self.config.alphabet_size, self.config.order), self.number_of_states),
                                                  initializer = B_initializer,
                                                  dtype = self.config.dtype,
                                                  trainable = True, name = "emission_kernel")
        # ---> full model done <---
        # ---> standard model <---
        else:
            self.init_kernel = self.add_weight(shape = (len(self.indices_for_I),),
                                               initializer = I_initializer,
                                               dtype = self.config.dtype,
                                               trainable = True, name = "init_kernel")

            if self.config.use_weights_for_consts:
                transition_kernel_shape = (len(self.config.indices_for_A),)
                emission_kernel_shape = (len(self.config.indices_for_B), )
            else:
                transition_kernel_shape = (len(self.config.indices_for_weights_A),)
                emission_kernel_shape = (len(self.config.indices_for_weights_B), )

            self.transition_kernel = self.add_weight(shape = transition_kernel_shape,
                                                     initializer = A_initializer,
                                                     dtype = self.config.dtype,
                                                     trainable = True, name = "transition_kernel")
            self.emission_kernel = self.add_weight(shape = emission_kernel_shape,
                                                  initializer = B_initializer,
                                                  dtype = self.config.dtype,
                                                  trainable = True, name = "emission_kernel")
        # ---> standard model done <---

        visualize_after_build = False
        if visualize_after_build:
            import WriteData
            import os
            WriteData.write_to_file(self.A_dense, f"{self.config.src_path}/output/{self.nCodons}codons/A.{self.nCodons}codons.txt")
            WriteData.write_to_file(tf.transpose(self.B_dense), f"{self.config.src_path}/output/{self.nCodons}codons/B.{self.nCodons}codons.txt")
            WriteData.write_to_file(self.I_dense, f"{self.config.src_path}/output/{self.nCodons}codons/I.{self.nCodons}codons.txt")
            WriteData.write_order_transformed_B_to_csv(self.B_dense, f"{self.config.src_path}/output/{self.nCodons}codons/B.{self.nCodons}codons.csv", self.config.order, self.nCodons)

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
    def A_sparse(self):
        # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.transition_kernel)), [self.transition_kernel], name = "self.transition_kernel_when_at_property_A_sparse", summarize = -1)
        if self.config.use_weights_for_consts:
            indices = self.indices_for_A
            values =  self.transition_kernel
        else:
            consts = tf.cast([1.0] * len(self.indices_for_constants_A), dtype = self.config.dtype)
            values = tf.concat([self.transition_kernel, consts], axis = 0)
            indices = self.indices_for_weights_A + self.indices_for_constants_A

        transition_matrix = tf.sparse.SparseTensor(indices = indices, \
                                                   values = values, \
                                                   dense_shape = [self.number_of_states] * 2)

        transition_matrix = tf.sparse.reorder(transition_matrix)
        transition_matrix = tf.sparse.softmax(transition_matrix, name = "A_sparse")

        return transition_matrix

    @property
    def A_dense(self): # ca 7% != 0
        return tf.sparse.to_dense(self.A_sparse, name = "A_dense")

    @property
    def A_full_model(self):
        transition_matrix = self.transition_kernel
        transition_matrix = tf.nn.softmax(transition_matrix, name = "A_full_model")
        return transition_matrix
############################################################################
############################################################################
############################################################################

    @property
    def B_sparse(self):
        # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.emission_kernel)), [self.emission_kernel], name = "self.emission_kernel_when_at_property_B_sparse", summarize = -1)

        # TODO: how often is this called?
        if self.config.use_weights_for_consts:
            indices = self.indices_for_B
            values = self.emission_kernel
        else:
            indices = self.indices_for_weights_B + self.indices_for_constants_B
            consts = tf.cast([1.0] * len(self.indices_for_constants_B), dtype = self.config.dtype)
            values = tf.concat([self.emission_kernel, consts], axis = 0)

        dense_shape = [self.number_of_states, \
                       n_emission_columns_in_B(self.config.alphabet_size, self.config.order)]

        emission_matrix = tf.sparse.SparseTensor(indices = indices, \
                                                 values = values, \
                                                 dense_shape = dense_shape)

        emission_matrix = tf.sparse.reorder(emission_matrix)
        emission_matrix = tf.sparse.reshape(emission_matrix, shape = (self.number_of_states, -1, self.config.alphabet_size))
        emission_matrix = tf.sparse.softmax(emission_matrix)
        emission_matrix = tf.sparse.reshape(emission_matrix, shape = (self.number_of_states, -1))

        emission_matrix = tf.sparse.transpose(emission_matrix, name = "B_sparse")
        return emission_matrix

    @property # ca 17% != 0
    def B_dense(self): #  this is order transformed if sparse is
        return tf.sparse.to_dense(self.B_sparse, name = "B_dense")

    @property
    def B_full_model(self):
        emission_matrix = self.emission_kernel
        emission_matrix = tf.nn.softmax(emission_matrix, name = "B_full_model")
        return emission_matrix
############################################################################
    @property
    def I_sparse(self): # i think dense is always used
        # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.init_kernel)), [self.init_kernel], name = "self.init_kernel_when_at_property_I_sparse", summarize = -1)

        # indices, values = self.get_indices_and_values_from_initial_kernel(self.init_kernel, self.nCodons)
        initial_matrix = tf.sparse.SparseTensor(indices = self.indices_for_I, values = self.init_kernel, dense_shape = [self.number_of_states,1])
        initial_matrix = tf.sparse.reorder(initial_matrix)
        initial_matrix = tf.sparse.reshape(initial_matrix, (1,self.number_of_states), name = "I_sparse")
        initial_matrix = tf.sparse.softmax(initial_matrix, name = "I_sparse")

        return initial_matrix

    @property
    def I_dense(self):
        return tf.sparse.to_dense(self.I_sparse, name = "I_dense")

    @property
    def I_full_model(self):
        initial_matrix = self.init_kernel
        initial_matrix = tf.nn.softmax(initial_matrix)
        initial_matrix = tf.reshape(initial_matrix, (1,self.number_of_states), name = "I_full_model")
        return initial_matrix

################################################################################
################################################################################
################################################################################
    def get_E(self, inputs):
        if self.config.call_type in [2, 3]: # B is sparse
            return tf.matmul(inputs, self.B_dense)
        if self.config.call_type == 4: # full_model
            return tf.matmul(inputs, self.B_full_model)
        return tf.sparse.sparse_dense_matmul(inputs, self.B_sparse)

    def get_R(self, old_forward, init = False):
        if init:
            if self.config.call_type == 4:
                return self.I_full_model, tf.cast(1.0, dtype = self.config.dtype) # bc return must be same in main and off branch, must be != 0 bc assert check z != 0
            return self.I_dense, tf.cast(1.0, dtype = self.config.dtype) # bc return must be same in main and off branch, must be != 0 bc assert check z != 0

        Z_i_minus_1 = tf.reduce_sum(old_forward, axis = 1, keepdims = True)
        scaled_forward = old_forward / Z_i_minus_1
        # if add_epsilon_to_z:
        #     Z_i_minus_1 = tf.math.add(Z_i_minus_1, add_epsilon_to_z)
        if self.config.call_type in [0,2]: # A is sparse
            R = tf.sparse.sparse_dense_matmul(scaled_forward, self.A_sparse)
        elif self.config.call_type == 4:
            R = tf.matmul(scaled_forward, self.A_full_model)
        else:
            R = tf.matmul(scaled_forward, self.A_dense)

        return R, Z_i_minus_1

################################################################################
    def call(self, inputs, states, training = None): # call_sparse
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell call_sparse")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell call_sparse: tf")

        old_forward, old_loglik, count = states
        # print("optype", self.A_dense.op.type)
        count = tf.math.add(count, 1)

        if self.config.check_assert:
            A = self.A_dense if self.config.call_type != 4 else self.A_full_model
            B = self.B_dense if self.config.call_type != 4 else self.B_full_model
            I = self.I_dense if self.config.call_type != 4 else self.I_full_model

            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.init_kernel)),       [self.init_kernel, self.transition_kernel, self.emission_kernel, count[0,0]], name = "init_kernel_beginning_of_cell", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.transition_kernel)), [self.transition_kernel], name = "transition_kernel_beginning_of_cell", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.emission_kernel)),   [self.emission_kernel],   name = "emission_kernel_beginning_of_cell", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(A)),            [A, old_loglik, old_forward, count[0,0]], name = "A_dense_beginning_of_call", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(B)),            [B, count[0,0]],            name = "B_dense_beginning_of_call", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(I)),            [I, count[0,0]], name = "I_dense_beginning_of_call", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(old_forward)),  [old_forward, count[0,0]],  name = "old_forward",               summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(old_loglik)),   [old_loglik, count[0,0]],   name = "old_loglik",                summarize = self.config.assert_summarize)

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
                verbose_print("A", self.A_dense)
                verbose_print("B", self.B_dense)
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

################################################################################

if __name__ == '__main__':
    import numpy as np
    config = {}

    config["nCodons"] = 1
    config["order"] = 2
    config["order_transformed_input"] = True
    config["call_type"] = 4 # 0:A;B sparse, 1:A dense, 2:B dense, 3:A;B dense, 4:fullmodel
    config["bench_path"] = f"./bench/{config['nCodons']}codons/{config['call_type']}_{config['order_transformed_input']}orderTransformedInput.log"

    config["alphabet_size"] = 4
    c = CgpHmmCell(config)
    print(len(c.get_indices_and_values_for_transition_kernel_higher_order(np.arange(10000))[0]))
    print(len(c.get_indices_for_weights_for_A()), len(c.get_indices_for_constants_for_A()))
