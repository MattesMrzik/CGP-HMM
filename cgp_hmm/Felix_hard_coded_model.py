#!/usr/bin/env python3
from Model import Model
import re
from itertools import product
import tensorflow as tf
import json
import numpy as np

class My_Model(Model):

    # this overwrites the init from Model. alternatively i can omit it
    def __init__(self, config):
        Model.__init__(self, config)

        self.insert_low = 0 if self.config.inserts_at_intron_borders else 1
        self.insert_high = self.config.nCodons + 1 if self.config.inserts_at_intron_borders else self.config.nCodons
        # =================> states <============================================
        self.id_to_state, self.state_to_id = self.get_state_id_description_dicts()
        self.number_of_states = self.get_number_of_states()


        # =================> emissions <========================================
        self.emissions_state_size = self.get_emissions_state_size()
        self.number_of_emissions = self.get_number_of_emissions()
        self.emi_to_id, self.id_to_emi = self.get_dicts_for_emission_tuple_and_id_conversion() # these are dicts


        self.A_is_dense = config.A_is_dense
        self.A_is_sparse = config.A_is_sparse
        self.B_is_dense = config.B_is_dense
        self.B_is_sparse = config.B_is_sparse

        # I
        self.I_indices = self.I_indices()

        # A
        self.A_indices_for_weights, \
        self.A_indices_for_constants, \
        self.A_initial_weights_for_trainable_parameters, \
        self.A_initial_weights_for_constants = self.A_indices_and_initial_weights()
        self.A_indices = np.concatenate([self.A_indices_for_weights, self.A_indices_for_constants])

        if config.use_weights_for_consts:

            self.A_indices_for_weights = np.concatenate([self.A_indices_for_weights, self.A_indices_for_constants])
            self.A_indices_for_constants = []
            self.A_initial_weights_for_trainable_parameters = np.concatenate([self.A_initial_weights_for_trainable_parameters, self.A_initial_weights_for_constants])
            self.A_initial_weights_for_constants = []


        # if self.config.my_initial_guess_for_parameters:
        #     self.A_my_initial_guess_for_parameters = self.get_A_my_initial_guess_for_parameters()

        # self.A_consts = self.get_A_consts()

        # B
        self.B_indices_for_weights, self.B_indices_for_constants = self.B_indices_for_weights_and_consts()
        self.B_indices = self.B_indices_for_weights + self.B_indices_for_constants

        if config.use_weights_for_consts:
            self.B_indices = sorted(self.B_indices)

        shape = (self.number_of_emissions, self.number_of_states)
        B_indices_complement = tf.where(tf.ones(shape, dtype = tf.float32) - tf.scatter_nd(self.B_indices, [1.0] * len(self.B_indices), shape = shape))
        self.B_indices_complement = tf.cast(B_indices_complement, dtype = tf.int32)

        self.A_as_dense_to_file("./A_msa.csv", self.A_initial_weights_for_trainable_parameters, with_description = 1)
        self.B_as_dense_to_file("./B_msa.csv", self.B_kernel_size() * [1], with_description = 1)

    # =================> states <===============================================
    def get_number_of_states(self):
        return len(self.id_to_state)

    def get_state_id_description_dicts(self):
        # if this is changed, also change state_is_third_pos_in_frame()

        states = ["L"]

        states += ["M1", "M2", "M3", "M4"]
        states += ["I1", "I2", "I3"]

        states += ["C", "R", "ter"]


        for i, state in enumerate(states):
            print(i, state)

        state_to_id = dict(zip(states, range(len(states))))
        id_to_state = dict(zip(range(len(states)), states))

        return id_to_state, state_to_id

    def state_id_to_str(self, id):
        return self.id_to_state[id]

    def str_to_state_id(self, s):
        return self.state_to_id[s]

    # =================> emissions <============================================
    def get_emissions_state_size(self):# with out terminal symbol
        alphabet_size = self.config.alphabet_size
        order = self.config.order
        if order == 0:
            return alphabet_size #  hier stand ne 1, weil plus 1 oder ienfach nur ausversehen

        #      [IACGT] x [ACGT]^order
        s = (alphabet_size + 1) * alphabet_size ** (order)
        # IIA IIC IIG IIT (if order == 2)
        s += sum([alphabet_size ** i for i in range(1, order)])
        return s

    # added 4, of which the last one corresponds to the terminal symbol
    # the other ones are dummy, they make the columns in B divisable by 4
    def get_number_of_emissions(self):
        return self.emissions_state_size + 4

    def get_dicts_for_emission_tuple_and_id_conversion(self):
        # if config == None:
        #     assert alphabet_size != None, "get_dicts_for_emission_tuple_and_id_conversion must be provided with config or (alphabet_size and order)"
        #     assert order != None, "get_dicts_for_emission_tuple_and_id_conversion must be provided with config or (alphabet_size and order)"
        # if alphabet_size == None:
        #     alphabet_size = config.alphabet_size
        # if order == None:
        #     order = config.order
        alphabet_size = self.config.alphabet_size
        order = self.config.order
        emi_to_id = {}
        id_to_emi = {}
        if order == 0:
            emi_to_id = dict([(tuple([base]), id) for id, base in enumerate(list(range(alphabet_size)) + ["X"])])
            id_to_emi = dict([(id, tuple([base])) for id, base in enumerate(list(range(alphabet_size)) + ["X"])])
        else:
            id = 0
            for emission_tuple in product(list(range(alphabet_size + 1)), repeat = order + 1):
                if not self.has_I_emission_after_base(emission_tuple):
                    id_to_emi[id] = emission_tuple
                    emi_to_id[emission_tuple] = id
                    id += 1
            id_to_emi[id] = tuple("X")
            emi_to_id[tuple("X")] = id

        # print("emi_to_id =", emi_to_id)
        # print("id_to_emi =", id_to_emi)

        return emi_to_id, id_to_emi

    # emission is either a tuple like [2,1,3] or "X"
    def emission_tuple_to_id(self, emission_tuple):
        return self.emi_to_id[tuple(emission_tuple)]

    def emission_id_to_tuple(self, id):
        return self.id_to_emi[id]

    def emission_tuple_to_str(self, emission_tuple):
        if emission_tuple[0] == "X":
            return "X"
        return "".join(list(map(lambda x: "ACGTI"[x], emission_tuple)))

    def emission_id_to_str(self, id):
        try:
            return self.emission_tuple_to_str(self.emission_id_to_tuple(id))
        except:
            return "-1"

    def str_to_emission_tuple(self, s):
        pass

    def str_to_emission_id(self, s):
        pass

################################################################################
################################################################################
################################################################################
    def I_kernel_size(self):
        # return len(self.I_indices)
        return 0

    def A_kernel_size(self):
        return len(self.A_indices_for_weights)

    def B_kernel_size(self):
        if self.config.use_weights_for_consts:
            return len(self.B_indices_for_weights) + len(self.B_indices_for_constants)
        return len(self.B_indices_for_weights)
################################################################################
################################################################################
################################################################################
    # TODO: maybe use small letters instead of capital ones
    def I_indices(self):
        only_start_in_ig5 = True
        if only_start_in_ig5:
            return [[0,0]]
        # start and codons
        indices = [[i,0] for i in range(3 + self.config.nCodons*3)]
        # inserts
        indices += [[i,0] for i in range(8 + self.config.nCodons*3, 8 + self.config.nCodons*3 + (self.config.nCodons + 1)*3)]

        return indices
################################################################################
################################################################################
################################################################################
    def A_indices_and_initial_weights(self):
        # für weights die trainable sind und gleich viele einer ähnlichen art sind,
        # die in eine separate methode auslagen, damit ich leichter statistiken
        # dafür ausarbeiten kann
        indicies_for_constant_parameters = []
        indices_for_trainable_parameters = []
        initial_weights_for_consts = []
        initial_weights_for_trainable_parameters = []

        # etwas zufälligkeit auf die initial parameter addieren?
        def append_transition(s1 = None, s2 = None, l = None, trainable = True, initial_weights = None):
            if l == None: # -> make l and list containing single weight
                assert s1 != None and s2 != None, "s1 and s2 must be != None if l = None"
                if initial_weights == None:
                    initial_weights = 0
                assert type(initial_weights) in [int, float], "if you append a single transition, you must pass either no initial weight or int or float"
                initial_weights = [initial_weights]
                l = [[self.str_to_state_id(s1), self.str_to_state_id(s2)]]
            if initial_weights == None:
                initial_weights = [0] * len(l)
            assert type(initial_weights) == list, "if you pass a list of transitions, you must pass either no initial weights or list of ints or floats"
            assert len(l) == len(initial_weights), "list of indices must be the same length as list of initial parameters"
            for entry, weight in zip(l, initial_weights):
                if self.config.add_noise_to_initial_weights:
                    weight += "small variance random normal"
                if trainable:
                    indices_for_trainable_parameters.append(entry)
                    initial_weights_for_trainable_parameters.append(weight)
                else:
                    indicies_for_constant_parameters.append(entry)
                    initial_weights_for_consts.append(weight)

        # append_transition("left_intron", "left_intron", trainable = not self.config.left_intron_const, initial_weights = self.config.left_intron_const)
        append_transition("L", "L", trainable = False)
        append_transition("L", "M1")
        append_transition("L", "M2")
        append_transition("L", "M3")
        append_transition("L", "M4")
        append_transition("L", "R")
        append_transition("L", "ter")
        append_transition("L", "C")

        append_transition("I1","I1")
        append_transition("I2","I2")
        append_transition("I3","I3")

        append_transition("M1","I1")

        append_transition("M2","I2")

        append_transition("M3","I3")

        append_transition("I1","M2")
        append_transition("I2","M3")
        append_transition("I3","M4")

        append_transition("M1", "R")
        append_transition("M2", "R")
        append_transition("M3", "R")
        append_transition("M4", "R")

        append_transition("M1", "ter")
        append_transition("M2", "ter")
        append_transition("M3", "ter")
        append_transition("M4", "ter")

        append_transition("M1", "C")
        append_transition("M2", "C")
        append_transition("M3", "C")
        append_transition("M4", "C")

        append_transition("M1", "M2")
        append_transition("M2", "M3")
        append_transition("M3", "M4")

        append_transition("C", "C")
        append_transition("C", "M1")
        append_transition("C", "M2")
        append_transition("C", "M3")
        append_transition("C", "M4")
        append_transition("C", "R")
        append_transition("C", "ter")

        append_transition("R", "R")
        append_transition("R", "ter")
        append_transition("ter", "ter")



        print("trainable")
        for index in indices_for_trainable_parameters:
            print(self.state_id_to_str(index[0]),"\t", self.state_id_to_str(index[1]))

        print("const")
        for index in indicies_for_constant_parameters:
            print(self.state_id_to_str(index[0]),"\t", self.state_id_to_str(index[1]))

        initial_weights_for_trainable_parameters = np.array(initial_weights_for_trainable_parameters, dtype = np.float32)
        initial_weights_for_consts = np.array(initial_weights_for_consts, dtype = np.float32)
        return indices_for_trainable_parameters, indicies_for_constant_parameters, initial_weights_for_trainable_parameters, initial_weights_for_consts

    @property
    def A_indices_enter_next_codon(self):
        indices = []
        for i in range(self.config.nCodons-1):
            indices += [[self.str_to_state_id(f"c_{i},2"), self.str_to_state_id(f"c_{i+1},0")]]
        return indices

    # deletes
    @property
    def A_indices_normal_deletes(self):
        indices = []
        # from codons
        for after_codon in range(self.config.nCodons):
            for to_codon in range(after_codon + 2, self.config.nCodons):
                indices += [[self.str_to_state_id(f"c_{after_codon},2"), self.str_to_state_id(f"c_{to_codon},0")]]
        return indices

    @property
    def A_indices_deletes_after_intron_to_codon(self):
        indices = []
        for i in range(1,self.config.nCodons):# including to the last codon
            for j in range(3):
                indices += [[self.str_to_state_id("AG"), self.str_to_state_id(f"c_{i},{j}")]]
        return indices
    @property
    def A_indices_deletes_after_codon_to_intron(self):
        indices = []
        for i in range(self.config.nCodons-1):# including to the first codon
            for j in range(3):
                indices += [[self.str_to_state_id(f"c_{i},{j}"), self.str_to_state_id("G")]]
        return indices
    @property
    def A_indices_deletes_after_insert_to_codon(self):
        indices = []
        for codon_id in range(self.config.nCodons-2):
            for insert_id in range(2, self.config.nCodons):
                indices += [[self.str_to_state_id(f"c_{codon_id},2"), self.str_to_state_id(f"i_{insert_id}, 0")]]
        return indices
    @property
    def A_indices_deletes_after_codon_to_insert(self):
        indices = []
        for insert_id in range(1, self.config.nCodons - 1):
            for codon_id in range(2, self.config.nCodons):
                indices += [[self.str_to_state_id(f"i_{insert_id},2"), self.str_to_state_id(f"c_{codon_id}, 0")]]
        return indices

    # inserts
    @property
    def A_indices_begin_inserts(self):
        indices = []
        if self.config.inserts_at_intron_borders:
            indices += [[self.str_to_state_id("AG"), self.str_to_state_id("i_0,0")]]
            indices += [[self.str_to_state_id("AG"), self.str_to_state_id("i_0,1")]]
            indices += [[self.str_to_state_id("AG"), self.str_to_state_id("i_0,2")]]
        for i in range(self.config.nCodons - 1):
            indices += [[self.str_to_state_id(f"c_{i},2"), self.str_to_state_id(f"i_{i+1},0")]]
        if self.config.inserts_at_intron_borders:
            indices += [[self.str_to_state_id(f"c_{self.config.nCodons-1},0"), self.str_to_state_id(f"i_{self.insert_high},0")]]
            indices += [[self.str_to_state_id(f"c_{self.config.nCodons-1},1"), self.str_to_state_id(f"i_{self.insert_high},0")]]
            indices += [[self.str_to_state_id(f"c_{self.config.nCodons-1},2"), self.str_to_state_id(f"i_{self.insert_high},0")]]
        return indices
    @property
    def A_indices_end_inserts(self):
        indices = []
        for i in range(self.insert_low, self.config.nCodons):
            indices += [[self.str_to_state_id(f"i_{i},2"), self.str_to_state_id(f"c_{i},0")]]
         # including last insert -> GT
        if self.config.inserts_at_intron_borders:
                indices += [[self.str_to_state_id(f"i_{self.config.nCodons},2"), self.str_to_state_id("G")]]
        return indices
    @property
    def A_indices_continue_inserts(self):
        indices = []
        for i in range(self.insert_low, self.insert_high):
            indices += [[self.str_to_state_id(f"i_{i},2"), self.str_to_state_id(f"i_{i},0")]]
        return indices
################################################################################
    def A_indices(self):
        return self.A_indices_for_weights + self.A_indices_for_constants
################################################################################
    # def get_A_consts(self):
    #     if self.config.left_intron_const:
    #         # return tf.cast(tf.concat([[5.0,1], [1.0] * (len(self.A_indices_for_constants) -2)], axis = 0),dtype = self.config.dtype)
    #         if self.config.right_intron_const:
    #             return tf.cast(tf.concat([[self.config.left_intron_const,1], [1.0] * (len(self.A_indices_for_constants) -4), [self.config.right_intron_const,1]], axis = 0),dtype = self.config.dtype)
    #         else:
    #             return tf.cast(tf.concat([[self.config.left_intron_const,1], [1.0] * (len(self.A_indices_for_constants) -2)], axis = 0),dtype = self.config.dtype)
    #     return tf.cast([1.0] * len(self.A_indices_for_constants), dtype = self.config.dtype)
################################################################################
    # def get_A_my_initial_guess_for_parameters(self):
    #     # für die ordnung die asserts vielleicht nach Config.py verschieben
    #     assert self.config.ig5_const_transition, "when using my initial guess for parameters also pass ig5_const_transition"
    #     assert self.config.ig3_const_transition, "when using my initial guess for parameters also pass ig3_const_transition"
    #     assert not self.config.no_deletes, "when using my initial guess for parameters do not pass no_deletes"
    #     assert not self.config.no_inserts, "when using my initial guess for parameters do not pass no_inserts"
    #     assert not self.config.use_weights_for_consts, "when using my initial guess for parameters do not pass use_weights_for_consts"
    #
    #     my_weights = []
    #     # enter codon
    #     my_weights += [4] * len(self.A_indices_enter_next_codon)
    #
    #     # begin_inserts
    #     my_weights += [1] * len(self.A_indices_begin_inserts)
    #
    #     # end inserts
    #     my_weights += [4] * len(self.A_indices_end_inserts)
    #
    #     # continue inserts
    #     my_weights += [1] * len(self.A_indices_continue_inserts)
    #
    #     # enter stop
    #     my_weights += [4]
    #
    #     # deletes                                  2 is just an arbitrary factor
    #     my_weights += [1 - j/2 for i in range(self.config.nCodons) for j in range(self.config.nCodons - i)]
    #
    #     # cast
    #     # my_weights = tf.cast(my_weights, dtype = self.config.dtype)
    #
    #     return my_weights

################################################################################
################################################################################
################################################################################
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
################################################################################
    def strip_or_pad_emission_with_n(self, ho_emission):
        return ["N"] * (self.config.order - len(ho_emission) + 1) + list(ho_emission)[- self.config.order - 1:]
################################################################################
    def has_I_emission_after_base(self, ho_emission): # or is only I
        alphabet_size = self.config.alphabet_size
        order = self.config.order

        found_emission = False
        invalid_emission = False
        for i in range(order +1):
            if found_emission and ho_emission[i] == alphabet_size:
                # print("not adding ", x)
                invalid_emission = True
                break
            if ho_emission[i] != alphabet_size:
                found_emission = True
        invalid_emission = invalid_emission if found_emission else True
        return invalid_emission
################################################################################
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
################################################################################
    def state_is_third_pos_in_frame(self, state):
        # s = self.state_id_to_str(state)
        # if s [-1] == "2" and s != "stop2" and s != "ter2":
        #     return True
        return False
################################################################################
    def get_emissions_that_fit_ambiguity_mask(self, ho_mask, x_bases_must_preceed, state):

        # getting the allowd base emissions in each slot
        # ie "NNA" and x_bases_must_preceed = 2 -> [][0,1,2,3], [0,1,2,3], [0]]
        allowed_bases = [0] * (self.config.order + 1)
        for i, emission in enumerate(self.strip_or_pad_emission_with_n(ho_mask)):
            allowed_bases[i] = self.nucleotide_ambiguity_code_to_array(emission)
            if i < self.config.order - x_bases_must_preceed:
                allowed_bases[i] += [4] # initial emission symbol

        allowed_ho_emissions = []
        state_is_third_pos_in_frame_bool = self.state_is_third_pos_in_frame(state)
        for ho_emission in product(*allowed_bases):
            if not self.has_I_emission_after_base(ho_emission) \
            and not (state_is_third_pos_in_frame_bool and self.emission_is_stop_codon(ho_emission)):
                allowed_ho_emissions += [ho_emission]

        return allowed_ho_emissions
################################################################################
    def get_indices_for_emission_and_state(self, indices, state, mask, x_bases_must_preceed):
        # if self.order_transformed_input and emissions[-1] == "X":
        if mask[-1] == "X":
            indices += [[self.emission_tuple_to_id("X"), state]]
            return

        for ho_emission in self.get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state):
            indices += [[self.emission_tuple_to_id(ho_emission), state]]
################################################################################
    def B_indices_for_weights_and_consts(self):
        nCodons = self.config.nCodons
        indices_for_trainable_parameters = []
        indicies_for_constant_parameters = []
        states_which_are_already_added = []
        def append_emission(state, mask = "N", x_bases_must_preceed = self.config.order, trainable = True):
            states_which_are_already_added.append(state)
            if trainable:
                self.get_indices_for_emission_and_state(indices_for_trainable_parameters, self.str_to_state_id(state), mask, x_bases_must_preceed)
            else:
                self.get_indices_for_emission_and_state(indicies_for_constant_parameters, self.str_to_state_id(state), mask, x_bases_must_preceed)

        append_emission("L", x_bases_must_preceed=0)
        append_emission("ter", mask="X")

        states_that_werent_added_yet = set(self.state_to_id.keys()).difference(states_which_are_already_added)
        states_that_werent_added_yet = sorted(states_that_werent_added_yet)
        for state in states_that_werent_added_yet:
            append_emission(state)

        return indices_for_trainable_parameters, indicies_for_constant_parameters
################################################################################
################################################################################
################################################################################
    def I(self, weights):
        # initial_matrix = tf.scatter_nd([[0,0]], [1.0], [self.number_of_states,1])
        initial_matrix = tf.scatter_nd([[0,0]], [1.0], [1, self.number_of_states])
        return initial_matrix

################################################################################
    def A(self, weights):
        if self.config.use_weights_for_consts:
            values = weights
        else:
            values = tf.concat([weights, self.A_initial_weights_for_constants], axis = 0)
        dense_shape = [self.number_of_states, self.number_of_states]

        if self.config.A_is_sparse:
            # print("a sparse")
            transition_matrix = tf.sparse.SparseTensor(indices = self.A_indices, \
                                                       values = values, \
                                                       dense_shape = dense_shape)

            transition_matrix = tf.sparse.reorder(transition_matrix)
            transition_matrix = tf.sparse.softmax(transition_matrix, name = "A_sparse")

        if self.config.A_is_dense:
            # print("a dense")
            transition_matrix = tf.scatter_nd(self.A_indices, values, dense_shape)
            softmax_layer = tf.keras.layers.Softmax()
            mask = tf.scatter_nd(self.A_indices, [1.0] * len(self.A_indices), dense_shape)
            transition_matrix = softmax_layer(transition_matrix, mask)
            # transition_matrix = tf.nn.softmax(transition_matrix, name = "A_dense")

        return transition_matrix
################################################################################
    def B(self, weights):
        if self.config.use_weights_for_consts:
            values = weights
        else:
            consts = tf.cast([1.0] * len(self.B_indices_for_constants), dtype = self.config.dtype)
            values = tf.concat([weights, consts], axis = 0)
        dense_shape = [self.number_of_emissions, \
                       self.number_of_states]

        if self.config.B_is_sparse:
            # print("b sparse")
            emission_matrix = tf.sparse.SparseTensor(indices = self.B_indices, \
                                                     values = values, \
                                                     dense_shape = dense_shape)

            emission_matrix = tf.sparse.reorder(emission_matrix)
            emission_matrix = tf.sparse.transpose(emission_matrix)
            emission_matrix = tf.sparse.reshape(emission_matrix, shape = (self.number_of_states, -1, self.config.alphabet_size))
            emission_matrix = tf.sparse.softmax(emission_matrix)# for sparse only sparse.softmax works, which has no arg "axis"
            # softmax_layer = tf.keras.layers.Softmax(axis = 1)
            # softmax_layer(emission_matrix)
            emission_matrix = tf.sparse.reshape(emission_matrix, shape = (self.number_of_states, self.number_of_emissions), name = "B_sparse")
            emission_matrix = tf.sparse.transpose(emission_matrix)

        if self.config.B_is_dense:
            shape_to_apply_softmax_to = (-1, self.config.alphabet_size, self.number_of_states)
            emission_matrix = tf.scatter_nd(self.B_indices, values, dense_shape)
            mask = tf.scatter_nd(self.B_indices, [1.0] * len(self.B_indices), dense_shape)
            # reshape
            emission_matrix = tf.reshape(emission_matrix, shape = shape_to_apply_softmax_to)
            mask            = tf.reshape(mask,            shape = shape_to_apply_softmax_to)
            # softmax
            softmax_layer = tf.keras.layers.Softmax(axis = 1) # using layer here, bc it has a mask
            emission_matrix = softmax_layer(emission_matrix, mask)# this leaves [0.25, 0.25, 0.25, 0.25] in columns where the mask has only zeros
            # reshape
            emission_matrix = tf.reshape(emission_matrix, shape = dense_shape, name = "B_dense")
            #removing the [0.25, 0.25, 0.25, 0.25] artefact
            emission_matrix = tf.tensor_scatter_nd_min(emission_matrix, \
                                                       self.B_indices_complement, \
                                                       [0.0] * len(self.B_indices_complement))
        return emission_matrix
################################################################################
################################################################################
################################################################################
    # def export_to_dot_and_png(self, A_weights, B_weights, out_path = None):
    #     # TODO: add I parameters???
    #     import numpy as np
    #     n_labels = self.number_of_emissions ** (self.config.order + 1)
    #     nCodons = self.config.nCodons
    #     if out_path == None:
    #         out_path = f"output/{nCodons}codons/graph.{nCodons}codons.gv"
    #
    #     A = self.A(A_weights) if self.A_is_dense else tf.sparse.to_dense(self.A(A_weights))
    #     B = self.B(B_weights) if self.B_is_dense else tf.sparse.to_dense(self.B(B_weights))
    #
    #     B_reshaped = tf.reshape(B, shape = (-1, self.config.alphabet_size, self.number_of_states))
    #     B_argmax = np.argmax(B_reshaped, axis = 1)
    #
    #     id_to_base = {0:"A", 1:"C",2:"G",3:"T",4:"I",5:"Ter"}
    #     with open(out_path, "w") as graph:
    #         graph.write("DiGraph G{\nrankdir=LR;\n")
    #         # graph.write("nodesep=0.5; splines=polyline;")
    #         for from_state, row in enumerate(A):
    #             from_state_str = self.state_id_to_str(from_state)
    #             write_B = False
    #             graph.write("\"" + from_state_str + "\"\n") #  this was to_state before
    #             if write_B:
    #
    #                 graph.write("[\n")
    #                 graph.write("\tshape = none\n")
    #                 graph.write("\tlabel = <<table border=\"0\" cellspacing=\"0\"> \n")
    #                 try:
    #                     color = {"c_":"teal", "i_": "crimson"}[from_state_str[0:2]]
    #                 except:
    #                     color = "white"
    #
    #                 graph.write(f"\t\t<tr><td port=\"port1\" border=\"1\" bgcolor=\"{color}\">" + from_state_str + "</td></tr>\n")
    #
    #                 for k, most_likely_index in enumerate(B_argmax[:,from_state]):
    #                     emission_id = most_likely_index + k * self.config.alphabet_size
    #                     emission_str = self.emission_id_to_str(emission_id)
    #                     emi_prob = str(np.round(B[emission_id, from_state].numpy(),4))
    #                     graph.write(f"\t\t<tr><td port=\"port{k+2}\" border=\"1\">({emission_str + ' ' +emi_prob})</td></tr>\n" )
    #                 graph.write("\t </table>>\n")
    #                 graph.write("]\n")
    #
    #             for to_state, prob in enumerate(row):
    #                 to_state_str = self.state_id_to_str(to_state)
    #                 if prob > 0:
    #                     prob = prob.numpy()
    #                     graph.write(f"\"{from_state_str}\" -> \"{to_state_str}\" [label = {str(np.round(prob, 4))[:6]} fontsize=\"{30*prob + 5}pt\"]\n")
    #
    #         graph.write("}")
    #     # run(f"cat graph.{nCodons}codons.gv")
    #     from Utility import run
    #     run(f"dot -Tpng {out_path} -o {out_path[:-2] + 'png'}")


    def I_as_dense_to_json_file(self, path, weights):
        with open(path, "w") as out_file:
            json.dump(self.I(weights).numpy().tolist(), out_file)

    # TODO: or do i want to have these functions in the cell, such that i dont have to pass the weights?
    def A_as_dense_to_str(self, weights, with_description = False, sep = " "):
        A = self.A(weights) if self.A_is_dense else tf.sparse.to_dense(self.A(weights))
        result = ""
        if with_description:
            result += " "
            for to_state in range(self.number_of_states):
                if sep in self.state_id_to_str(to_state):
                    print(f"sep '{sep}' is contained in self.state_id_to_str(to_state) '{self.state_id_to_str(to_state)}'")
                result += self.state_id_to_str(to_state)
                result += " "
            result += "\n"
        for from_state, row in enumerate(A):
            if with_description:
                result += self.state_id_to_str(from_state) + " "
            for entry in row:
                result += str(entry.numpy())
                result += " "
            result += "\n"
        return result

    def A_as_dense_to_file(self, path, weights, with_description = False):
        with open(path, "w") as out_file:
            out_file.write(self.A_as_dense_to_str(weights, with_description))

    def A_as_dense_to_json_file(self, path, weights):
        with open(path, "w") as out_file:
            A = self.A(weights) if self.A_is_dense else tf.sparse.to_dense(self.A(weights))
            json.dump(A.numpy().tolist(), out_file)

    def B_as_dense_to_str(self, weights, with_description = False, sep = " "):
        B = self.B(weights) if self.B_is_dense else tf.sparse.to_dense(self.B(weights))
        result = ""
        if with_description:
            result += sep
            for to_state in range(self.number_of_states):
                if sep in self.state_id_to_str(to_state):
                    print(f"sep '{sep}' is contained in self.state_id_to_str(to_state) '{self.state_id_to_str(to_state)}'")
                result += self.state_id_to_str(to_state)
                result += sep
            result += "\n"

        for emission_id, row in enumerate(B):
            if with_description:
                if sep in self.emission_id_to_str(emission_id)
                    print(f"sep '{sep}' is contained in self.emission_id_to_str(emission_id) '{self.emission_id_to_str(emission_id)}'")
                result += self.emission_id_to_str(emission_id) + sep
            for entry in row:
                result += str(entry.numpy())
                result += sep
            result += "\n"
        return result

    def B_as_dense_to_file(self, path, weights, with_description = False):
        with open(path, "w") as out_file:
            out_file.write(self.B_as_dense_to_str(weights, with_description))

    def B_as_dense_to_json_file(self, path, weights):
        with open(path, "w") as out_file:
            B = self.B(weights) if self.B_is_dense else tf.sparse.to_dense(self.B(weights))
            json.dump(B.numpy().tolist(), out_file)


if __name__ == '__main__':
    from Config import Config
    config = Config("main_programm")
    f = My_Model(config)
    import numpy as np
    print(f.A(np.ones(13)))
