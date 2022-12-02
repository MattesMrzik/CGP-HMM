#!/usr/bin/env python3

from Utility import higher_order_emission_to_id
from Utility import append_time_ram_stamp_to_file
from Utility import description_to_state_id
from Utility import state_id_to_description

from Utility import tfprint

from Utility import get_indices_for_weights_from_transition_kernel_higher_order
from Utility import get_indices_for_constants_from_transition_kernel_higher_order
from Utility import get_indices_for_weights_from_emission_kernel_higher_order
from Utility import get_indices_for_constants_from_emission_kernel_higher_order

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
        append_time_ram_stamp_to_file(start, f"Cell.__init__() start {run_id}", config["bench_path"])

        # super(CgpHmmCell, self).__init__()
        super(CgpHmmCell, self).__init__()

        self.nCodons = config["nCodons"]

        self.alphabet_size = config["alphabet_size"] # without terminal symbol and without "papped left flank" symbol

        self.order = config["order"] # order = 0 -> emission prob depends only on current emission

        self.config = config

        self.state_size = [self.number_of_states, 1,      1]

        if not config["order_transformed_input"]:
            self.state_size = [self.number_of_states, 1,         1] + ([tf.TensorShape([self.order, self.alphabet_size + 2])] if self.order > 0 else [])

        self.indices_for_weights_A = config["indices_for_weights_A"]# if "indices_for_weights_A" in config else get_indices_for_weights_from_transition_kernel_higher_order(config)
        # vielleich einfach den consts auch ein weigt geben, welches durch softmax eh dann 1 wird
        # dann hat der gradient zwar mehr einträge, aber es muss ein concat der values und indices gemacht werden,
        self.indices_for_constants_A = config["indices_for_constants_A"] #if "indices_for_constants_A" in config else get_indices_for_constants_from_transition_kernel_higher_order(config)
        self.indices_for_A = config["indices_for_A"]


        self.indices_for_weights_B = config["indices_for_weights_B"]#if "indices_for_weights_B" in config else get_indices_for_weights_from_emission_kernel_higher_order(config)
        self.indices_for_constants_B = config["indices_for_constants_B"]# if "indices_for_constants_B" in config else get_indices_for_constants_from_emission_kernel_higher_order(config)
        self.indices_for_B = config["indices_for_B"]

        self.indices_for_I = config["indices_for_I"]

        # self.indices_for_weights_A = self.get_indices_for_weights_from_transition_kernel_higher_order()
        # # vielleich einfach den consts auch ein weigt geben, welches durch softmax eh dann 1 wird
        # # dann hat der gradient zwar mehr einträge, aber es muss ein concat der values und indices gemacht werden,
        # self.indices_for_constants_A = self.get_indices_for_constants_from_transition_kernel_higher_order()
        #
        # self.indices_for_weights_B = self.get_indices_for_weights_from_emission_kernel_higher_order()
        # self.indices_for_constants_B = self.get_indices_for_constants_from_emission_kernel_higher_order()


        append_time_ram_stamp_to_file(start, f"Cell.__init__() end   {run_id}", self.config["bench_path"])

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

        if not self.config["order_transformed_input"]:
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

    def build(self, input_shape):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell build")
        tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell build: tf")

        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(start, f"Cell.build() start {run_id}", self.config["bench_path"])


        if self.config["get_gradient_for_current_txt"]:
            with open(f"{self.config['src_path']}/output/{self.config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_I.json") as file:
                weights_I = np.array(json.load(file))
                I_initializer = tf.constant_initializer(weights_I)
            with open(f"{self.config['src_path']}/output/{self.config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_A.json") as file:
                weights_A = np.array(json.load(file))
                A_initializer = tf.constant_initializer(weights_A)
            with open(f"{self.config['src_path']}/output/{self.config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_B.json") as file:
                weights_B = np.array(json.load(file))
                B_initializer = tf.constant_initializer(weights_B)
        elif self.config["get_gradient_from_saved_model_weights"] and "model" in self.config:
            weights = self.config["model"].get_weights()
            # this causes error,
            # try if txt is sufficient to get nan as gradient
            I_initializer = tf.constant_initializer(weights[0])
            A_initializer = tf.constant_initializer(weights[1])
            B_initializer = tf.constant_initializer(weights[2])
        else:
            I_initializer="random_normal"
            A_initializer="random_normal"
            B_initializer="random_normal"

        self.init_kernel = self.add_weight(shape = (len(self.indices_for_I),),
                                           initializer = I_initializer,
                                           dtype = self.config["dtype"],
                                           trainable = True, name = "init_kernel")

        # full model
        if self.config["call_type"] == 4 and self.config["order_transformed_input"]:
            self.transition_kernel = self.add_weight(shape = (self.number_of_states,self.number_of_states),
                                                     initializer = A_initializer,
                                                     dtype = self.config["dtype"],
                                                     trainable = True, name = "transition_kernel")
            how_many_emissions = (self.config["alphabet_size"] + 1) ** (self.config["order"] + 1 ) + 1
            self.emission_kernel = self.add_weight(shape = (how_many_emissions, self.number_of_states),
                                                  initializer = B_initializer,
                                                  dtype = self.config["dtype"],
                                                  trainable = True, name = "emission_kernel")
        else:
            if self.config["use_weights_for_consts"]:
                self.transition_kernel = self.add_weight(shape = (len(self.config["indices_for_A"]),),
                                                         initializer = A_initializer,
                                                         dtype = self.config["dtype"],
                                                         trainable = True, name = "transition_kernel")
                how_many_emissions = len(self.config["indices_for_B"])
                if not self.config["order_transformed_input"]:
                    how_many_emissions = self.number_of_states * (self.alphabet_size + 2)**(self.order + 1)

                self.emission_kernel = self.add_weight(shape = (how_many_emissions, ),
                                                      initializer = B_initializer,
                                                      dtype = self.config["dtype"],
                                                      trainable = True, name = "emission_kernel")

                # i need more weights for the indices that where for consts before
            else:
                self.transition_kernel = self.add_weight(shape = (len(self.indices_for_weights_A),),
                                                         initializer = A_initializer,
                                                         dtype = self.config["dtype"],
                                                         trainable = True, name = "transition_kernel")

                how_many_emissions = len(self.indices_for_weights_B)
                if not self.config["order_transformed_input"]:
                    how_many_emissions = self.number_of_states * (self.alphabet_size + 2)**(self.order + 1)

                self.emission_kernel = self.add_weight(shape = (how_many_emissions, ),
                                                      initializer = B_initializer,
                                                      dtype = self.config["dtype"],
                                                      trainable = True, name = "emission_kernel")

        visualize_after_build = False
        if visualize_after_build:
            import WriteData
            import os
            WriteData.write_to_file(self.A_dense, f"{self.config['src_path']}/output/{self.nCodons}codons/A.{self.nCodons}codons.txt")
            WriteData.write_to_file(tf.transpose(self.B_dense), f"{self.config['src_path']}/output/{self.nCodons}codons/B.{self.nCodons}codons.txt")
            WriteData.write_to_file(self.I_dense, f"{self.config['src_path']}/output/{self.nCodons}codons/I.{self.nCodons}codons.txt")
            WriteData.write_order_transformed_B_to_csv(self.B_dense, f"{self.config['src_path']}/output/{self.nCodons}codons/B.{self.nCodons}codons.csv", self.config["order"], self.nCodons)

            os.system(f"./Visualize.py -c {self.config['nCodons']} -o {self.config['order']} -t")
            exit(1)
        append_time_ram_stamp_to_file(start, f"Cell.build() end   {run_id}", self.config["bench_path"])


    def init_cell(self):
        self.inita = True
    # order transformed input, sparse

############################################################################
############################################################################
############################################################################
    def get_indices_for_constants_from_transition_kernel_higher_order(self):
        nCodons = self.nCodons
        # from start a
        indices = [[1,2]]
        # from start t
        indices += [[2,3]]

        # first to second codon position
        indices += [[4 + i*3, 5 + i*3] for i in range(nCodons)]
        # second to third codon position
        indices += [[5 + i*3, 6 + i*3] for i in range(nCodons)]

        # inserts
        offset = 8 + 3*nCodons
        # begin inserts


        # first to second position in insert
        indices += [[offset + i*3, offset + 1 + i*3] for i in range(nCodons + 1)]
        # second to third position in insert
        indices += [[offset + 1 + i*3, offset + 2 + i*3] for i in range(nCodons + 1)]
        # ending an insert

        # stop T
        indices += [[4 + nCodons*3, 5 + nCodons*3]]

        # second to third position in stop
        indices += [[5 + nCodons*3, 6 + nCodons*3]]

        # stop -> ig 3'
        indices += [[6 + nCodons*3, 7 + nCodons*3]]


        index_of_terminal_1 = 8 + nCodons*3 + (nCodons + 1) *3
        indices += [[index_of_terminal_1, index_of_terminal_1]]

        return indices

    def get_indices_for_weights_from_transition_kernel_higher_order(self): # no shared parameters
        nCodons = self.nCodons
        # from ig 5'
        indices = [[0,0], [0,1]]

        # enter codon
        indices += [[3 + i*3, 4 + i*3] for i in range(nCodons)]

        # begin inserts
        offset = 8 + 3*nCodons
        indices += [[3 + i*3, offset + i*3] for i in range(nCodons + 1)]
        # ending an insert
        indices += [[offset + 2 + i*3, 4 + i*3] for i in range(nCodons + 1)]
        # continuing an insert
        indices += [[offset + 2 + i*3, offset + i*3] for i in range(nCodons +1)]

        # exit last codon
        indices += [[3 + nCodons*3, 4 + nCodons*3]]

        # deletes
        i_delete = [3 + i*3 for i in range(nCodons) for j in range(nCodons-i)]
        j_delete = [4 + j*3 for i in range(1,nCodons+1) for j in range(i,nCodons+1)]
        indices += [[i,j] for i,j in zip(i_delete, j_delete)]

        # ig -> ig, terminal_1
        index_of_terminal_1 = 8 + nCodons*3 + (nCodons + 1) *3
        indices += [[7 + nCodons*3, 7 + nCodons*3], [7 + nCodons*3, index_of_terminal_1]]


        return indices

    def get_indices_and_values_from_transition_kernel_higher_order(self, w):
        nCodons = self.nCodons
        k = 0
        # ig 5'
        indices = [[0,0], [0,1]]
        values = [1 - w[k], w[k]] # lieber sigmoid
        k += 1
        # start a
        indices += [[1,2]]
        values += [1]
        # start t
        indices += [[2,3]]
        values += [1]

        # enter codon
        indices += [[3 + i*3, 4 + i*3] for i in range(nCodons)]
        # print("values =", values)
        # print("w[k: k + nCodons] =", w[k: k + nCodons])
        values = tf.concat([values, w[k: k + nCodons]], axis = 0)
        k += nCodons
        # first to second codon position
        indices += [[4 + i*3, 5 + i*3] for i in range(nCodons)]
        values = tf.concat([values, [1] * nCodons], axis = 0)
        # second to third codon position
        indices += [[5 + i*3, 6 + i*3] for i in range(nCodons)]
        values = tf.concat([values, [1] * nCodons], axis = 0)

        # inserts
        offset = 8 + 3*nCodons
        # begin inserts
        use_inserts = True
        if use_inserts:
            indices += [[3 + i*3, offset + i*3] for i in range(nCodons + 1)]
            values = tf.concat([values, w[k: k + nCodons + 1]], axis = 0)
            k += nCodons + 1

        # exit last codon
        indices += [[3 + nCodons*3, 4 + nCodons*3]]
        values = tf.concat([values, [w[k]]], axis = 0)
        k += 1

        # first to second position in insert
        indices += [[offset + i*3, offset + 1 + i*3] for i in range(nCodons + 1)]
        values = tf.concat([values, [1] * (nCodons + 1)], axis = 0)
        # second to third position in insert
        indices += [[offset + 1 + i*3, offset + 2 + i*3] for i in range(nCodons + 1)]
        values = tf.concat([values, [1] * (nCodons + 1)], axis = 0)
        # ending an insert
        indices += [[offset + 2 + i*3, 4 + i*3] for i in range(nCodons + 1)]
        values = tf.concat([values, w[k: k + nCodons + 1]], axis = 0)

        # continuing an insert
        indices += [[offset + 2 + i*3, offset + i*3] for i in range(nCodons +1)]
        values = tf.concat([values, 1-w[k: k + nCodons +1]], axis = 0)
        k += nCodons + 1

        # deletes
        i_delete = [3 + i*3 for i in range(nCodons) for j in range(nCodons-i)]
        j_delete = [4 + j*3 for i in range(1,nCodons+1) for j in range(i,nCodons+1)]
        indices += [[i,j] for i,j in zip(i_delete, j_delete)]
        # print("deletes =", [1-w[k] * w[k]**((j-i)/3) for i,j in zip(i_delete, j_delete)])
        values = tf.concat([values, [1-w[k] * w[k]**int((j-i)/3) for i,j in zip(i_delete, j_delete)]], axis = 0)
        k += 1

        # stop T
        indices += [[4 + nCodons*3, 5 + nCodons*3]]
        values = tf.concat([values, [1]], axis = 0)

        # second to third position in stop
        indices += [[5 + nCodons*3, 6 + nCodons*3]]
        values = tf.concat([values, [1]], axis = 0)

        # stop -> ig 3'
        indices += [[6 + nCodons*3, 7 + nCodons*3]]
        values = tf.concat([values, [1]], axis = 0)

        # ig -> ig, terminal_1
        index_of_terminal_1 = 8 + nCodons*3 + (nCodons + 1) *3
        indices += [[7 + nCodons*3, 7 + nCodons*3], [7 + nCodons*3, index_of_terminal_1]]
        # values = tf.concat([values, [.5] * 2], axis = 0) # this parameter doesnt have to be learned (i think)
        # .5 can be any other number, since softmax(x,x) = [.5, .5]
        # but: TypeError: Cannot convert [0.5, 0.5] to EagerTensor of dtype int32   (todo)
        values = tf.concat([values, [1] * 2], axis = 0) # this parameter doesnt have to be learned (i think)


        # if self.order_transformed_input:
            # terminal -> terminal
        indices += [[index_of_terminal_1, index_of_terminal_1]]
        values = tf.concat([values, [1]], axis = 0)

        # not order transformed input
        # else:
        #     # terminal_1 -> terminal_1, a mix of true bases and X are emitted
        #     # terminal_1 -> terminal_2, only X are emitted
        #     indices += [[index_of_terminal_1, index_of_terminal_1], [index_of_terminal_1, index_of_terminal_1 +1]]
        #     values = tf.concat([values, [1] * 2], axis = 0)
        #
        #     # terminal_2 -> terminal_2
        #     indices += [[index_of_terminal_1 +1, index_of_terminal_1 +1]]
        #     values = tf.concat([values, [1]], axis = 0)



        return indices, values


    @property
    def A_sparse(self):
        # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.transition_kernel)), [self.transition_kernel], name = "self.transition_kernel_when_at_property_A_sparse", summarize = -1)
        if self.config["use_weights_for_consts"]:
            transition_matrix = tf.sparse.SparseTensor(indices = self.indices_for_A, \
                                                       values = self.transition_kernel, dense_shape = [self.number_of_states] * 2)
        else:
            consts = tf.cast([1.0] * len(self.indices_for_constants_A), dtype = self.config["dtype"])
            values = tf.concat([self.transition_kernel, consts], axis = 0)
            transition_matrix = tf.sparse.SparseTensor(indices = self.indices_for_weights_A + self.indices_for_constants_A, \
                                                       values = values, dense_shape = [self.number_of_states] * 2)
        transition_matrix = tf.sparse.reorder(transition_matrix)
        transition_matrix = tf.sparse.softmax(transition_matrix, name = "A_sparse")

        if self.config["weaken_softmax"]:
            transition_matrix = tf.sparse.map_values(tf.add, transition_matrix, 0.0001)
            s = tf.sparse.reduce_sum(transition_matrix, axis = 0, name = "A_sparse_weakend_softmax")
            # tfprint(s)
            # tfprint(1/s)
            inverse_s = 1/s
            transition_matrix = transition_matrix * inverse_s
        return transition_matrix

    # @property
    # def A_sparse(self):
    #     indices, values = self.get_indices_and_values_from_transition_kernel_higher_order(self.transition_kernel, self.nCodons)
    #     transition_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [self.number_of_states] * 2)
    #     transition_matrix = tf.sparse.reorder(transition_matrix)
    #     transition_matrix = tf.sparse.softmax(transition_matrix)
    #     return transition_matrix

    @property
    def A_dense(self): # ca 7% != 0
        return tf.sparse.to_dense(self.A_sparse, name = "A_dense")

    # @property
    # def A(self):
    #     return self.A_sparse
    #     # return self.A_dense
    @property

    def A_full_model(self):
        transition_matrix = self.transition_kernel
        transition_matrix = tf.nn.softmax(transition_matrix, name = "A_full_model")
        return transition_matrix
############################################################################
############################################################################
############################################################################

    def nucleotide_ambiguity_code_to_array(self, emission):
        # todo: somehow having this dict as self.code made it slower, why???
        code = {
            "A" : [0],
            "C" : [1],
            "G" : [2],
            "T" : [3],
            "Y" : [1,3],
            "R" : [0,2],
            "W" : [0,3],
            "S" : [1,2],
            "K" : [2,3],
            "M" : [0,1],
            "D" : [0,2,3],
            "V" : [0,1,2],
            "H" : [0,1,3],
            "B" : [1,2,3],
            "N" : [0,1,2,3],
            "X" : [5]
        }
        return code[emission]

    def strip_or_pad_emission_with_n(self, ho_emission):
        return ["N"] * (self.order - len(ho_emission) + 1) + list(ho_emission)[-self.order-1:]

    def has_I_emission_after_base(self, emission):
        found_emission = False
        invalid_emission = False
        for i in range(self.order +1):
            if found_emission and emission[i] == self.config["alphabet_size"]:
                # print("not adding ", x)
                invalid_emission = True
                break
            if emission[i] != self.config["alphabet_size"]:
                found_emission = True
        return invalid_emission

    def emission_is_stop_codon(self, ho_emission):
        stops = [[3,0,0],[3,0,2],[3,2,0]]
        if len(ho_emission) < 3:
            return False

        def same(a,b):
            for i in range(3):
                if a[i] != b[len(b) - 3 + i]:
                    return False
            return True
        for stop in stops:
            if same(ho_emission, stop):
                return True
        return False

    def state_is_third_pos_in_frame(self, state):
        des = state_id_to_description(state, self.nCodons, self.config["state_id_description_list"])
        if des [-1] == "2" and des != "stop2" and des != "ter2":
            return True
        return False

    def get_emissions_that_fit_ambiguity_mask(self, ho_mask, x_bases_must_preceed, state):

        # getting the allowd base emissions in each slot
        # ie "NNA" and x_bases_must_preceed = 2 -> [][0,1,2,3], [0,1,2,3], [0]]
        allowed_bases = [0] * (self.order + 1)
        for i, emission in enumerate(self.strip_or_pad_emission_with_n(ho_mask)):
            allowed_bases[i] = self.nucleotide_ambiguity_code_to_array(emission)
            if i < self.order - x_bases_must_preceed:
                allowed_bases[i] += [4] # initial emission symbol

        allowed_ho_emissions = []
        state_is_third_pos_in_frame_bool = self.state_is_third_pos_in_frame(state)
        for ho_emission in product(*allowed_bases):
            if not self.has_I_emission_after_base(ho_emission) \
            and not (state_is_third_pos_in_frame_bool and self.emission_is_stop_codon(ho_emission)):
                allowed_ho_emissions += [ho_emission]

        return allowed_ho_emissions


    def get_indices_and_values_for_emission_higher_order_for_a_state(self, weights, \
                                                                     k, indices, \
                                                                     values, state, \
                                                                     mask, \
                                                                     x_bases_must_preceed, \
                                                                     trainable = True):
        # if self.order_transformed_input and emissions[-1] == "X":
        if mask[-1] == "X":
            indices += [[state, (self.alphabet_size + 1) ** (self.order +1)]]
            values[0] = tf.concat([values[0], [1]], axis = 0)
            return

        count_weights = 0
        for ho_emission in self. get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state):

            indices += [[state, higher_order_emission_to_id(ho_emission, self.alphabet_size, self.order)]]
            count_weights += 1

        if trainable:
            values[0] = tf.concat([values[0], weights[k[0]:k[0] + count_weights]], axis = 0)
            k[0] += count_weights
        else:
            values[0] = tf.concat([values[0], [1] * count_weights], axis = 0)

    def get_indices_and_values_for_emission_higher_order_for_a_state_old_inputs(self, w, nCodons, alphabet_size):
        pass

    def get_indices_and_values_from_emission_kernel_higher_order(self, w, nCodons, alphabet_size):
        indices = []
        values = [[]] # will contain one tensor at index 0, wrapped it in a list such that it can be passed by reference, ie such that it is mutable
        weights = w
        k = [0]

        # ig 5'
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,0,"N",0)
        # start a
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,1,"A",0)
        # start t
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,2,"AT",0)
        # start g
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,3,"ATG",2, trainable = False)
        # codon_11
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,4,"ATGN",2)
        # codon_12
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,5,"ATGNN",2)
        # all other codons
        for state in range(6, 6 + nCodons*3 -2):
            self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,state,"N",2)
        # stop
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,4 + nCodons*3,"T",self.order)
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,5 + nCodons*3,"TA",self.order)
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,5 + nCodons*3,"TG",self.order)
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,6 + nCodons*3,"TAA",self.order, trainable = False)
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,6 + nCodons*3,"TAG",self.order, trainable = False)
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,6 + nCodons*3,"TGA",self.order, trainable = False)
        # ig 3'
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,7 + nCodons*3,"N",self.order)
        # inserts
        for state in range(8 + nCodons*3, 8 + nCodons*3 + (nCodons + 1)*3):
            self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,state,"N",self.order)

        self.get_indices_and_values_for_emission_higher_order_for_a_state(\
                     weights,k,indices,values,8 + nCodons*3 + (nCodons+1)*3,"X",self.order)

        return indices, values[0]

    def get_indices_for_emission_higher_order_for_a_state(self, \
                                                          indices, \
                                                          state, \
                                                          mask, \
                                                          x_bases_must_preceed):
        # if self.order_transformed_input and emissions[-1] == "X":
        if mask[-1] == "X":
            indices += [[state, (self.alphabet_size + 1) ** (self.order +1)]]
            return

        count_weights = 0
        for ho_emission in self. get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state):
            indices += [[state, higher_order_emission_to_id(ho_emission, self.alphabet_size, self.order)]]

    def get_indices_for_weights_from_emission_kernel_higher_order(self):
        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(start, f"Cell.get_indices_for_weights_from_emission_kernel_higher_order() start   {run_id}", self.config["bench_path"])
        nCodons = self.nCodons
        indices = []

        # ig 5'
        self.get_indices_for_emission_higher_order_for_a_state(indices,0,"N",0)
        # start a
        self.get_indices_for_emission_higher_order_for_a_state(indices,1,"A",1)
        # start t
        self.get_indices_for_emission_higher_order_for_a_state(indices,2,"AT",2)

        # codon_11
        self.get_indices_for_emission_higher_order_for_a_state(indices,4,"ATGN",2)
        # codon_12
        self.get_indices_for_emission_higher_order_for_a_state(indices,5,"ATGNN",2)
        # all other codons
        for state in range(6, 6 + nCodons*3 -2):
            self.get_indices_for_emission_higher_order_for_a_state(indices,state,"N",2)
        # stop
        self.get_indices_for_emission_higher_order_for_a_state(indices,4 + nCodons*3,"T",self.order)
        self.get_indices_for_emission_higher_order_for_a_state(indices,5 + nCodons*3,"TA",self.order)
        self.get_indices_for_emission_higher_order_for_a_state(indices,5 + nCodons*3,"TG",self.order)
        # ig 3'
        self.get_indices_for_emission_higher_order_for_a_state(indices,7 + nCodons*3,"N",self.order)
        # inserts
        for state in range(8 + nCodons*3, 8 + nCodons*3 + (nCodons + 1)*3):
            self.get_indices_for_emission_higher_order_for_a_state(indices,state,"N",self.order)

        self.get_indices_for_emission_higher_order_for_a_state(\
                              indices,8 + nCodons*3 + (nCodons+1)*3,"X",self.order)

        append_time_ram_stamp_to_file(start, f"Cell.get_indices_for_weights_from_emission_kernel_higher_order() end   {run_id}", self.config["bench_path"])

        return indices

    def get_indices_for_constants_from_emission_kernel_higher_order(self):
        nCodons = self.nCodons
        indices = []

        self.get_indices_for_emission_higher_order_for_a_state(indices,3,"ATG",2)
        self.get_indices_for_emission_higher_order_for_a_state(indices,6 + nCodons*3,"TAA",self.order)
        self.get_indices_for_emission_higher_order_for_a_state(indices,6 + nCodons*3,"TAG",self.order)
        self.get_indices_for_emission_higher_order_for_a_state(indices,6 + nCodons*3,"TGA",self.order)

        return indices

    @property
    def B_sparse(self):
        # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.emission_kernel)), [self.emission_kernel], name = "self.emission_kernel_when_at_property_B_sparse", summarize = -1)
#
        if self.config["use_weights_for_consts"]:
            emission_matrix = tf.sparse.SparseTensor(indices = self.indices_for_B, \
                                                     values = self.emission_kernel, \
                                                     dense_shape = [self.number_of_states, \
                                                                (self.alphabet_size + 1) ** (self.order + 1) + 1])
        else:
            consts = tf.cast([1.0] * len(self.indices_for_constants_B), dtype = self.config["dtype"])
            values = tf.concat([self.emission_kernel, consts], axis = 0)
            # indices, values = self.get_indices_and_values_from_emission_kernel_higher_order(self.emission_kernel, self.nCodons, self.alphabet_size)
            emission_matrix = tf.sparse.SparseTensor(indices = self.indices_for_weights_B + self.indices_for_constants_B, \
                                                     values = values, \
                                                     dense_shape = [self.number_of_states, \
                                                                (self.alphabet_size + 1) ** (self.order + 1) + 1])
        emission_matrix = tf.sparse.reorder(emission_matrix)
        emission_matrix = tf.sparse.softmax(emission_matrix)

        if self.config["weaken_softmax"]:
            emission_matrix = tf.sparse.map_values(tf.add, emission_matrix, 0.0001)
            s = tf.sparse.reduce_sum(emission_matrix, axis = 0, name = "B_sparse_weakend_softmax")
            # tfprint(s)
            # tfprint(1/s)
            inverse_s = 1/s
            emission_matrix = emission_matrix * inverse_s

        emission_matrix = tf.sparse.transpose(emission_matrix, name = "B_sparse")
        return emission_matrix

    @property # ca 17% != 0
    def B_dense(self): #  this is order transformed if sparse is
        return tf.sparse.to_dense(self.B_sparse, name = "B_dense")

    # @property
    # def B(self):
    #     return self.B_sparse
        # return self.B_dense()
    @property
    def B_full_model(self):
        if self.config["call_type"] == 4:
            emission_matrix = self.emission_kernel
            emission_matrix = tf.nn.softmax(emission_matrix, name = "B_full_model")
            return emission_matrix
############################################################################
############################################################################
############################################################################
    @property
    def I_sparse(self): # todo this is not yet used in call()
        # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.init_kernel)), [self.init_kernel], name = "self.init_kernel_when_at_property_I_sparse", summarize = -1)

        # indices, values = self.get_indices_and_values_from_initial_kernel(self.init_kernel, self.nCodons)
        initial_matrix = tf.sparse.SparseTensor(indices = self.indices_for_I, values = self.init_kernel, dense_shape = [self.number_of_states,1])
        initial_matrix = tf.sparse.reorder(initial_matrix)
        initial_matrix = tf.sparse.reshape(initial_matrix, (1,self.number_of_states))
        initial_matrix = tf.sparse.softmax(initial_matrix, name = "I_sparse")
        if self.config["weaken_softmax"]:
            initial_matrix = tf.sparse.map_values(tf.add, initial_matrix, 0.0001)
            s = tf.sparse.reduce_sum(initial_matrix, axis = 0, name = "B_sparse_weakend_softmax")
            # tfprint(s)
            # tfprint(1/s)
            inverse_s = 1/s
            initial_matrix = initial_matrix * inverse_s
        # initial_matrix = tf.sparse.reshape(initial_matrix, (self.number_of_states,1))
        return initial_matrix

    @property
    def I_dense(self):
        return tf.sparse.to_dense(self.I_sparse, name = "I_dense")

    @property
    def I(self):
        # return self.I_sparse()
        return self.I_dense
################################################################################
################################################################################
################################################################################
    def call_old_inputs(self, inputs, states, training = None):
        pass
        # for i in range(self.order):
        #     old_inputs_i_expanded = tf.expand_dims(old_inputs[:,i,:], axis = -1)
        #     for j in range(i + 1, self.order):
        #         old_inputs_i_expanded = tf.expand_dims(old_inputs_i_expanded, axis = -1)
        #     E = tf.multiply(old_inputs_i_expanded, E)
        #     E = tf.reduce_sum(E, axis = 1) # axis 0 is batch, so this has to be 1

################################################################################
    def call_full_model(self, inputs, states, training = None):
        old_forward, old_loglik, count = states
        count = count + 1

        run_id = randint(0,100)

        verbose = self.config["verbose"]

        if verbose:
            if self.config["print_to_file"]:
                outstream = f"file://{self.config['src_path']}/verbose/{self.nCodons}codons.txt"
            else:
                import sys
                outstream = sys.stdout

        E = tf.matmul(inputs, self.B_full_model)

        def verbose_print(string, data):
            if verbose:
                tf.print(count[0,0], run_id, string, tf.shape(data), output_stream = outstream, sep = ";")
                if verbose == 2:
                    tf.print(count[0,0], run_id, ">" + string, data, output_stream = outstream, sep = ";", summarize=-1)

        if verbose:
            verbose_print("count", count[0,0])
            verbose_print("inputs", inputs)
            verbose_print("old_forward", old_forward)
            verbose_print("old_loglik", old_loglik)
            verbose_print("E", E)

        if count[0,0] == 1:
            R = self.I_dense
            verbose_print("A", self.A_full_model)
            verbose_print("B", self.B_full_model)
        else:
            R = tf.matmul(old_forward, self.A_full_model)
            Z_i_minus_1 = tf.reduce_sum(old_forward, axis = 1, keepdims = True)
            R /= Z_i_minus_1

        alpha = E * R

        loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis = -1, keepdims = True, name = "loglik")) # todo keepdims = True?
        if verbose:
            verbose_print("R", R)
            verbose_print("forward", alpha)
            verbose_print("loglik", loglik)

        return [alpha, inputs, count], [alpha, loglik, count]
################################################################################
    def call_sparse(self, inputs, states, training = None): # call_sparse
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell call_sparse")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ cell call_sparse: tf")

        # print values of A B loglik alpha to see whether nan appear bc of nan loglik
        # bc some underflow in model (=A or B) and therefor the model ie liklihood breaks
        # or the gradient gets infinite or sth like this

        old_forward, old_loglik, count = states
        # print("optype", self.A_dense.op.type)

        check_assert = True

        if check_assert:
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.A_dense)), [self.A_dense, old_loglik, old_forward, count[0,0]], name = "A_dense_beginning_of_call", summarize = -1)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.B_dense)), [self.B_dense, count[0,0]], name = "B_dense_beginning_of_call", summarize = -1)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.I_dense)), [self.I_dense, count[0,0]], name = "I_dense_beginning_of_call", summarize = -1)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(old_forward)),  [old_forward, count[0,0]],  name = "old_forward",               summarize = -1)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(old_loglik)),   [old_loglik, count[0,0]],   name = "old_loglik",                summarize = -1)

        count = tf.math.add(count, 1)

        run_id = randint(0,100)

        verbose = self.config["verbose"]

        if verbose:
            if self.config["print_to_file"]:
                outstream = f"file://{self.config['src_path']}/verbose/{self.nCodons}codons.txt"
            else:
                import sys
                outstream = sys.stdout

        # inputs is shape batch * 126 (= (4+1)^3+1)

        E = tf.sparse.sparse_dense_matmul(inputs, self.B_sparse) # returns dense

        def verbose_print(string, data):
            if verbose:
                tf.print(count[0,0], run_id, string, tf.shape(data), output_stream = outstream, sep = ";")
                if verbose == 2:
                    tf.print(count[0,0], run_id, ">" + string, data, output_stream = outstream, sep = ";", summarize=-1)

        if verbose:
            verbose_print("count", count[0,0])
            verbose_print("inputs", inputs)
            verbose_print("old_forward", old_forward)
            verbose_print("old_loglik", old_loglik)
            verbose_print("E", E)

        if count[0,0] == 1:
            R = self.I # this might have to be dense, bc TypeError: 'R' must have the same nested structure in the main and else branches: and in the else branch it is dense
            verbose_print("A", self.A_dense)
            verbose_print("B", self.B_dense)
        else:
            R = tf.sparse.sparse_dense_matmul(old_forward, self.A_sparse)
            Z_i_minus_1 = tf.reduce_sum(old_forward, axis = 1, keepdims = True)
            R /= Z_i_minus_1
            if check_assert:
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(Z_i_minus_1)),  [Z_i_minus_1, count[0,0]],  name = "z_finite",      summarize = -1)
                tf.debugging.Assert(tf.math.reduce_all(Z_i_minus_1 != 0),                [Z_i_minus_1, count[0,0]],  name = "z_nonzero",      summarize = -1)
        alpha = E * R # batch * state_size

        # keepsdims is true such that shape of result is (32,1) rather than (32,)
        # loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis = -1, keepdims = True, name = "loglik"))
        loglik = tf.math.add(old_loglik, tf.math.log(tf.reduce_sum(alpha, axis = -1, keepdims = True)), name = "loglik")

        if check_assert:
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.A_dense)), [self.A_dense, count[0,0]], name = "A_dense_beginning_of_call", summarize = -1)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(self.B_dense)), [self.B_dense, count[0,0]], name = "B_dense_beginning_of_call", summarize = -1)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(alpha)),        [alpha, count[0,0]],        name = "alpha",                     summarize = -1)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik)),       [loglik, count[0,0]],       name = "loglik_finite",             summarize = -1)
            # i think this should be allowed since sum across alpha can be 1, then log is 0, which is fine
            # tf.debugging.Assert(tf.math.reduce_all(loglik != 0),                     [loglik, count[0,0]],       name = "loglik_nonzero",            summarize = -1)

            #todo also check if loglik is zero, bc then a seq should be impossible to be emitted, which shouldnt be the case

        if verbose:
            verbose_print("R", R)
            verbose_print("forward", alpha)
            verbose_print("loglik", loglik)

        return [alpha, inputs, count], [alpha, loglik, count]
################################################################################
    def call_A_sparse(self, inputs, states, training = None): # call_A_sparse
        old_forward, old_loglik, count = states
        count = count + 1

        run_id = randint(0,100)

        verbose = self.config["verbose"]

        if verbose:
            if self.config["print_to_file"]:
                outstream = f"file://{self.config['src_path']}/verbose/{self.nCodons}codons.txt"
            else:
                import sys
                outstream = sys.stdout

        # inputs is shape batch * 126 (= (4+1)^3+1)

        E = tf.matmul(inputs, self.B_dense)

        def verbose_print(string, data):
            if verbose:
                tf.print(count[0,0], run_id, string, tf.shape(data), output_stream = outstream, sep = ";")
                if verbose == 2:
                    tf.print(count[0,0], run_id, ">" + string, data, output_stream = outstream, sep = ";", summarize=-1)

        if verbose:
            verbose_print("count", count[0,0])
            verbose_print("inputs", inputs)
            verbose_print("old_forward", old_forward)
            verbose_print("old_loglik", old_loglik)
            verbose_print("E", E)

        if self.inita:
            # tf.print(count[0,0], run_id, " ", "self.init = ", self.init)
            self.inita = False

        if count[0,0] == 1:
            R = self.I # this might have to be dense, bc TypeError: 'R' must have the same nested structure in the main and else branches: and in the else branch it is dense
            verbose_print("A", self.A_dense)
            verbose_print("B", self.B_dense)
        else:
            R = tf.sparse.sparse_dense_matmul(old_forward, self.A_sparse)
            Z_i_minus_1 = tf.reduce_sum(old_forward, axis = 1, keepdims = True)
            R /= Z_i_minus_1
        alpha = E * R # batch * state_size

        # keepsdims is true such that shape of result is (32,1) rather than (32,)
        loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis = -1, keepdims = True, name = "loglik"))

        if verbose:
            verbose_print("R", R)
            verbose_print("forward", alpha)
            verbose_print("loglik", loglik)

        return [alpha, inputs, count], [alpha, loglik, count]

################################################################################
    def call_B_sparse(self, inputs, states, training = None): # call_B_sparse
        old_forward, old_loglik, count = states
        count = count + 1

        run_id = randint(0,100)

        verbose = self.config["verbose"]

        if verbose:
            if self.config["print_to_file"]:
                outstream = f"file://{self.config['src_path']}/verbose/{self.nCodons}codons.txt"
            else:
                import sys
                outstream = sys.stdout

        # inputs is shape batch * 126 (= (4+1)^3+1)

        E = tf.sparse.sparse_dense_matmul(inputs, self.B_sparse) # returns dense

        def verbose_print(string, data):
            if verbose:
                tf.print(count[0,0], run_id, string, tf.shape(data), output_stream = outstream, sep = ";")
                if verbose == 2:
                    tf.print(count[0,0], run_id, ">" + string, data, output_stream = outstream, sep = ";", summarize=-1)

        if verbose:
            verbose_print("count", count[0,0])
            verbose_print("inputs", inputs)
            verbose_print("old_forward", old_forward)
            verbose_print("old_loglik", old_loglik)
            verbose_print("E", E)

        if self.inita:
            # tf.print(count[0,0], run_id, " ", "self.init = ", self.init)
            self.inita = False

        if count[0,0] == 1:
            R = self.I # this might have to be dense, bc TypeError: 'R' must have the same nested structure in the main and else branches: and in the else branch it is dense
            verbose_print("A", self.A_dense)
            verbose_print("B", self.B_dense)
        else:
            R = tf.matmul(old_forward, self.A_dense)
            Z_i_minus_1 = tf.reduce_sum(old_forward, axis = 1, keepdims = True)
            R /= Z_i_minus_1
        alpha = E * R # batch * state_size

        # keepsdims is true such that shape of result is (32,1) rather than (32,)
        loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis = -1, keepdims = True, name = "loglik"))
        if verbose:
            verbose_print("R", R)
            verbose_print("forward", alpha)
            verbose_print("loglik", loglik)

        return [alpha, inputs, count], [alpha, loglik, count]
################################################################################
    def call_dense(self, inputs, states, training = None): # call_dense
        old_forward, old_loglik, count = states
        count = count + 1

        run_id = randint(0,100)

        verbose = self.config["verbose"]

        if verbose:
            if self.config["print_to_file"]:
                outstream = f"file://{self.config['src_path']}/verbose/{self.nCodons}codons.txt"
            else:
                import sys
                outstream = sys.stdout

        E = tf.matmul(inputs, self.B_dense)

        def verbose_print(string, data):
            if verbose:
                tf.print(count[0,0], run_id, string, tf.shape(data), output_stream = outstream, sep = ";")
                if verbose == 2:
                    tf.print(count[0,0], run_id, ">" + string, data, output_stream = outstream, sep = ";", summarize=-1)

        if verbose:
            verbose_print("count", count[0,0])
            verbose_print("inputs", inputs)
            verbose_print("old_forward", old_forward)
            verbose_print("old_loglik", old_loglik)
            verbose_print("E", E)

        if count[0,0] == 1:
            R = self.I_dense
            verbose_print("A", self.A_dense)
            verbose_print("B", self.B_dense)
        else:
            R = tf.matmul(old_forward, self.A_dense)
            Z_i_minus_1 = tf.reduce_sum(old_forward, axis = 1, keepdims = True)
            R /= Z_i_minus_1

        alpha = E * R

        loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis = -1, keepdims = True, name = "loglik")) # todo keepdims = True?
        if verbose:
            verbose_print("R", R)
            verbose_print("forward", alpha)
            verbose_print("loglik", loglik)

        return [alpha, inputs, count], [alpha, loglik, count]
################################################################################

    def call(self, inputs, states, training = None):
        if self.config["call_type"] == 0:
            return self.call_sparse(inputs, states, training)
        elif self.config["call_type"] == 1:
            return self.call_B_sparse(inputs, states, training)
        elif self.config["call_type"] == 2:
            return self.call_A_sparse(inputs, states, training)
        elif self.config["call_type"] == 3:
            return self.call_dense(inputs, states, training)
        elif self.config["call_type"] == 4:
            return self.call_full_model(inputs, states, training)

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
    print(len(c.get_indices_and_values_from_transition_kernel_higher_order(np.arange(10000))[0]))
    print(len(c.get_indices_for_weights_from_transition_kernel_higher_order()), len(c.get_indices_for_constants_from_transition_kernel_higher_order()))
