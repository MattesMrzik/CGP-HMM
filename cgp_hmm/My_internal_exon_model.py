#!/usr/bin/env python3
from Model import Model
import re
from itertools import product
import tensorflow as tf
import json
import numpy as np
import os

class My_Model(Model):

    # this overwrites the init from Model. alternatively i can omit it
    def __init__(self, config):
        Model.__init__(self, config)

    def prepare_model(self):
        self.insert_low = 0 if self.config.inserts_at_intron_borders else 1
        self.insert_high = self.config.nCodons + 1 if self.config.inserts_at_intron_borders else self.config.nCodons

        # =================> states <============================================
        self.id_to_state, self.state_to_id = self.get_state_id_description_dicts()
        self.number_of_states = self.get_number_of_states()
        # =================> emissions <========================================
        self.emissions_state_size = self.get_emissions_state_size()
        self.number_of_emissions = self.get_number_of_emissions()
        self.emi_to_id, self.id_to_emi = self.get_dicts_for_emission_tuple_and_id_conversion() # these are dicts
        self.A_is_dense = self.config.A_is_dense
        self.A_is_sparse = self.config.A_is_sparse
        self.B_is_dense = self.config.B_is_dense
        self.B_is_sparse = self.config.B_is_sparse


    def make_model(self):

        if self.config.priorB:
            from load_priors import Prior
            self.prior = Prior(self.config)

        # I
        self.I_indices = self.I_indices()

        # A
        self.A_indices_for_weights, \
        self.A_indices_for_constants, \
        self.A_initial_weights_for_trainable_parameters, \
        self.A_initial_weights_for_constants = self.A_indices_and_initial_weights()
        self.A_indices = np.concatenate([self.A_indices_for_weights, self.A_indices_for_constants])

        if self.config.priorA:
            self.get_A_prior_matrix()

        if self.config.use_weights_for_consts:

            self.A_indices_for_weights = np.concatenate([self.A_indices_for_weights, self.A_indices_for_constants])
            self.A_indices_for_constants = []
            self.A_initial_weights_for_trainable_parameters = np.concatenate([self.A_initial_weights_for_trainable_parameters, self.A_initial_weights_for_constants])
            self.A_initial_weights_for_constants = []


        # if self.config.my_initial_guess_for_parameters:
        #     self.A_my_initial_guess_for_parameters = self.get_A_my_initial_guess_for_parameters()

        # self.A_consts = self.get_A_consts()

        # B
        self.make_preparations_for_B()
        if self.config.priorB:
            self.get_B_prior_matrix()

        shape = (self.number_of_emissions, self.number_of_states)
        B_indices_complement = tf.where(tf.ones(shape, dtype = tf.float32) - tf.scatter_nd(self.B_indices, [1.0] * len(self.B_indices), shape = shape))
        self.B_indices_complement = tf.cast(B_indices_complement, dtype = tf.int32)

        self.B_weight_index_tuple_to_id_conversion_dict = self.get_index_tuple_to_id_conversion_dict(self.B_indices_for_trainable_parameters)
        self.B_const_index_tuple_to_id_conversion_dict = self.get_index_tuple_to_id_conversion_dict(self.B_indices_for_constant_parameters)

        self.B_initial_weights_for_trainable_parameters, \
        self.B_initial_weights_for_constant_parameters = self.make_B_initial_weights()


        # write prior parameters to file
        if self.config.nCodons < 20:
            dir_name = f"{self.config.out_path}/output/{self.config.nCodons}codons/prior_calculation"
            import os
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            self.B_as_dense_to_file(f"{dir_name}/B_init_parameters_after_conversion.csv", self.B_initial_weights_for_trainable_parameters, with_description = True)

        if self.config.use_weights_for_consts:
            self.B_indices = sorted(self.B_indices)

################################################################################
    def B_weight_tuple_id(self, tuple):
        return self.B_weight_index_tuple_to_id_conversion_dict[tuple]
    def B_const_tuple_id(self, tuple):
        return self.B_const_index_tuple_to_id_conversion_dict[tuple]

################################################################################
    def get_index_tuple_to_id_conversion_dict(self, list_of_indices):
        tuple_to_id = {}
        for i, indi in enumerate(list_of_indices):
            tuple_to_id[tuple(indi)] = i
        return tuple_to_id

    # =================> states <===============================================
    def get_number_of_states(self):
        return len(self.id_to_state)

    def get_state_id_description_dicts(self):
        # if this is changed, also change state_is_third_pos_in_frame()

        states = ["left_intron"]

        states += [f"ak_{i}" for i in range(self.config.akzeptor_pattern_len)]
        states += ["A", "AG"]
        codons = ["c_" + str(i) + "," + str(j) for i in range(self.config.nCodons) for j in range(3)]
        states += codons

        inserts = ["i_" + str(i) + "," + str(j) for i in range(self.insert_low, self.insert_high) for j in range(3)]
        states += inserts

        states += ["G", "GT"]
        states += [f"do_{i}" for i in range(self.config.donor_pattern_len)]
        states += ["right_intron"]

        states += ["ter"]
        # for i, state in enumerate(states):
        #     print(i, state)

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
            return f"no_str_for_id_{id}"

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
        return len(self.B_indices_for_trainable_parameters)
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
        single_high_prob_kernel = self.config.single_high_prob_kernel
        # für weights die trainable sind und gleich viele einer ähnlichen art sind,
        # die in eine separate methode auslagen, damit ich leichter statistiken
        # dafür ausarbeiten kann
        indicies_for_constant_parameters = []
        indices_for_trainable_parameters = []
        initial_weights_for_consts = []
        initial_weights_for_trainable_parameters = []

        self.A_prior_indices =[]

        # etwas zufälligkeit auf die initial parameter addieren?
        def append_transition(s1 = None, s2 = None, l = None, trainable = True, initial_weights = None, use_as_prior = False):
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
            if use_as_prior:
                assert trainable, "if you pass use_as_prior to append_transition() you must also pass trainable = True"
            for entry, weight in zip(l, initial_weights):
                if use_as_prior:
                    self.A_prior_indices.append(entry)
                if self.config.add_noise_to_initial_weights:
                    weight += "small variance random normal"
                if trainable:
                    indices_for_trainable_parameters.append(entry)
                    initial_weights_for_trainable_parameters.append(weight)
                else:
                    indicies_for_constant_parameters.append(entry)
                    initial_weights_for_consts.append(weight)


        append_transition("left_intron", "right_intron", trainable = self.config.exon_skip_const, initial_weights = self.config.exon_skip_init_weight)

        append_transition("left_intron", "left_intron", trainable = not self.config.left_intron_const, initial_weights = self.config.left_intron_init_weight)

        if self.config.akzeptor_pattern_len == 0:
            append_transition("left_intron", "A", trainable = not self.config.left_intron_const, initial_weights = 0)
        else:
            append_transition("left_intron", "ak_0", trainable = not self.config.left_intron_const, initial_weights =0)
            for i in range(self.config.akzeptor_pattern_len - 1):
                append_transition(f"ak_{i}", f"ak_{i+1}", trainable = False)
            append_transition(f"ak_{self.config.akzeptor_pattern_len - 1}", "A", trainable = False)

        append_transition("A", "AG", trainable = False)

        # enter first codon at all phases
        append_transition("AG", "c_0,0")
        append_transition("AG", "c_0,1")
        append_transition("AG", "c_0,2")

        append_transition(l = self.A_indices_enter_next_codon, initial_weights = [single_high_prob_kernel] * len(self.A_indices_enter_next_codon), use_as_prior=True)

        for i in range(self.config.nCodons):
            append_transition(f"c_{i},0", f"c_{i},1", trainable = False) # TODO: do i want these trainable if i pass deletes after/before intron?
            append_transition(f"c_{i},1", f"c_{i},2", trainable = False)

        # inserts
        #i dont have inserts right after or before splice site
        append_transition(l = self.A_indices_begin_inserts, use_as_prior=True)
        for i in range(self.insert_low, self.insert_high):
            append_transition(f"i_{i},0", f"i_{i},1", trainable = False)
            append_transition(f"i_{i},1", f"i_{i},2", trainable = False)
        append_transition(l = self.A_indices_end_inserts, initial_weights = [single_high_prob_kernel] * len(self.A_indices_end_inserts), use_as_prior=True)
        append_transition(l = self.A_indices_continue_inserts, use_as_prior=True)

        # deletes
        A_indices_normal_deletes, A_init_weights_normal_deletes = self.A_indices_and_init_weights_normal_deletes
        # print("A_init_weights_normal_deletes", A_init_weights_normal_deletes)
        append_transition(l = A_indices_normal_deletes, initial_weights = A_init_weights_normal_deletes, use_as_prior=True)
        if self.config.deletes_after_intron_to_codon:
            append_transition(l = self.A_indices_deletes_after_intron_to_codon)
        if self.config.deletes_after_codon_to_intron:
            append_transition(l = self.A_indices_deletes_after_codon_to_intron)
        if self.config.deletes_after_insert_to_codon:
            append_transition(l = self.A_indices_deletes_after_insert_to_codon)
        if self.config.deletes_after_codon_to_insert:
            append_transition(l = self.A_indices_deletes_after_codon_to_insert)

        # exit last codon
        append_transition(f"c_{self.config.nCodons-1},0", "G")
        append_transition(f"c_{self.config.nCodons-1},1", "G")
        append_transition(f"c_{self.config.nCodons-1},2", "G")

        append_transition("G", "GT", trainable = False)

        if self.config.donor_pattern_len == 0:
            append_transition("GT", "right_intron", trainable = False)
        else:
            append_transition("GT", "do_0", trainable = False)
            for i in range(self.config.donor_pattern_len-1):
                append_transition(f"do_{i}", f"do_{i+1}", trainable = False)
            append_transition(f"do_{self.config.donor_pattern_len-1}", "right_intron", trainable = False)

        append_transition("right_intron", "right_intron", trainable = not self.config.right_intron_const, initial_weights = self.config.right_intron_init_weight)
        append_transition("right_intron", "ter", trainable = not self.config.right_intron_const, initial_weights = 0)
        append_transition("ter", "ter")

        # print("trainable")
        # for index in indices_for_trainable_parameters:
        #     print(self.state_id_to_str(index[0]),"\t", self.state_id_to_str(index[1]))
        #
        # print("const")
        # for index in indicies_for_constant_parameters:
        #     print(self.state_id_to_str(index[0]),"\t", self.state_id_to_str(index[1]))


        initial_weights_for_consts = np.array(initial_weights_for_consts, dtype = np.float32)
        initial_weights_for_trainable_parameters = np.array(initial_weights_for_trainable_parameters, dtype = np.float32)
        return indices_for_trainable_parameters, indicies_for_constant_parameters, initial_weights_for_trainable_parameters, initial_weights_for_consts

    @property
    def A_indices_enter_next_codon(self):
        indices = []
        for i in range(self.config.nCodons-1):
            indices += [[self.str_to_state_id(f"c_{i},2"), self.str_to_state_id(f"c_{i+1},0")]]
        return indices

    # deletes
    @property
    def A_indices_and_init_weights_normal_deletes(self):
        indices = []
        init_weights = []
        # from codons
        for after_codon in range(self.config.nCodons):
            for to_codon in range(after_codon + 2, self.config.nCodons):
                indices += [[self.str_to_state_id(f"c_{after_codon},2"), self.str_to_state_id(f"c_{to_codon},0")]]
                init_weights += [-(to_codon - after_codon)/self.config.diminishing_factor]
        return indices, init_weights

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
        s = self.state_id_to_str(state)
        if s[-1] == "2" and s != "stop2" and s != "ter2" and s[0] in "ic":
            return True
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
    def prior_and_initial_parameter_for_state_and_emission(self, state, emission):
        emission = emission.lower()
        # state and emission should be string

        # TODO i think i dont have to norm left_intron and codon and insert since they should already be normed in the file
        prior = -1.0
        initial_parameter = -1.0

        if not self.config.priorB:
            return prior, initial_parameter


        if state in ["left_intron", "right_intron"] and emission[0] != "i":
            norm = sum([self.prior.get_intron_prob(emission[:-1] + base) for base in "ACGT"])
            assert abs(norm - 1 ) < 1e-4, f"norm is {norm} but should be one"
            prior = self.prior.get_intron_prob(emission) / norm
            initial_parameter = prior
        if state[0] == "c": # c_0,2   c_12,1
            codon_id = int(re.search(r"c_(\d+),\d+", state).group(1))
            if codon_id not in [0, self.config.nCodons - 1]:
                norm = sum([self.prior.get_exon_prob(emission[:-1] + base, window = int(state[-1])) for base in "ACGT"])
                assert abs(norm - 1 ) < 1e-3, f"norm is {norm} but should be one"
                prior = self.prior.get_exon_prob(emission, window = int(state[-1])) / norm
                initial_parameter = prior
            if codon_id == 0:
                assert self.config.ass_end == 2, " self.config.ass_end == 2 is assumed for the following code, but it is set to a different value"
                if state[-1] == "0":
                    pattern = f".{{{self.config.ass_start}}}AG{emission[-1]}.{{{self.config.ass_end-1}}}"
                    norm_pattern = f".{{{self.config.ass_start}}}AG.{{{self.config.ass_end}}}"
                    unscaled_prob = self.prior.get_splice_site_matching_pattern_probs(description = "ASS", pattern = pattern)
                    norm = self.prior.get_splice_site_matching_pattern_probs(description = "ASS", pattern = norm_pattern)
                    assert norm != 0, f"norm is zero in state {state} and emission {emission}, with pattern {pattern}"
                    prior = unscaled_prob/norm
                    initial_parameter = prior
                if state[-1] == "1":
                    pattern = f".{{{self.config.ass_start}}}AG{emission[1:]}.{{{self.config.ass_end-2}}}"
                    norm_pattern = f".{{{self.config.ass_start}}}AG{emission[1]}.{{{self.config.ass_end-1}}}"
                    unscaled_prob = self.prior.get_splice_site_matching_pattern_probs(description = "ASS", pattern = pattern)
                    norm = self.prior.get_splice_site_matching_pattern_probs(description = "ASS", pattern = norm_pattern)
                    assert norm != 0, f"norm is zero in state {state} and emission {emission}, with pattern {pattern}"
                    prior = unscaled_prob/norm
                    initial_parameter = prior


        if state[0] == "i":
            codon_id = int(re.search(r"i_(\d+),\d+", state).group(1))
            if codon_id not in [0, self.config.nCodons]:
                norm = sum([self.prior.get_exon_prob(emission[:-1] + base, window = int(state[-1])) for base in "ACGT"])
                assert abs(norm - 1 ) < 1e-3, f"norm is {norm} but should be one"
                prior = self.prior.get_exon_prob(emission, window = int(state[-1])) / norm
                initial_parameter = prior

        if state.startswith("ak_"):
            ak_id = int(state[-1])
            how_many_states_before_A = self.config.akzeptor_pattern_len - ak_id
            # = 1 if state is right before A, = 2 if there is one more state before A
            start_in_prior_pattern = self.config.ass_start - how_many_states_before_A - self.config.order
            fits_in_prior_pattern = start_in_prior_pattern >= 0
            if fits_in_prior_pattern:
                before_emission = f".{{{self.config.ass_start - how_many_states_before_A - self.config.order}}}"
                after_emission = f".{{{how_many_states_before_A-1}}}AG.*"
                pattern = f"{before_emission}{emission}{after_emission}"
                unscaled_prob = self.prior.get_splice_site_matching_pattern_probs(description = "ASS", pattern = pattern)
                norm_pattern = f"{before_emission}{emission[:-1]}.{after_emission}"
                norm = self.prior.get_splice_site_matching_pattern_probs(description = "ASS", pattern = norm_pattern)
                assert norm != 0, "norm in ak is zero 0923htui4"
                prior = unscaled_prob/norm
                initial_parameter = prior
                # TODO if prior is zero maybe dont set initial para to zero
                # TODO are pseudo count also added to non occuring patterns?

        if state.startswith("do_"):
            assert self.config.donor_pattern_len < 10, "donor pattern len >=10 gubi2t9w0gurz8"
            if state[-1] == "0":
                pattern = f".{{{self.config.dss_start}}}GT{emission[-1]}.{{{self.config.dss_end-1}}}"
                norm_pattern = f".{{{self.config.dss_start}}}GT.{{{self.config.dss_end}}}"
                unscaled_prob = self.prior.get_splice_site_matching_pattern_probs(description = "DSS", pattern = pattern)
                norm = self.prior.get_splice_site_matching_pattern_probs(description = "DSS", pattern = norm_pattern)
                assert norm != 0, "norm in ak is zero 2ß30tu9z8wg"
                prior = unscaled_prob/norm
                initial_parameter = prior
            if state[-1] == "1":
                pattern = f".{{{self.config.dss_start}}}GT{emission[1:]}.{{{self.config.dss_end-2}}}"
                norm_pattern = f".{{{self.config.dss_start}}}GT{emission[1]}.{{{self.config.dss_end-1}}}"
                unscaled_prob = self.prior.get_splice_site_matching_pattern_probs(description = "DSS", pattern = pattern)
                norm = self.prior.get_splice_site_matching_pattern_probs(description = "DSS", pattern = norm_pattern)
                assert norm != 0, "norm in ak is zero 2ß30tu9z8wg"
                prior = unscaled_prob/norm
                initial_parameter = prior



        # from prior and initial parameter the initial weights are computed.
        # if -1 is passed then the prob is split amoung the parameters
        # assert that if prob is split there is still something left to be split
        # and if it it greater than 1 then norm it and send allert to user
        return prior, initial_parameter
################################################################################
################################################################################
################################################################################
    ''' Getting priors'''

    def get_A_prior_matrix(self):
        # TODO rather than dense calculation use sparse ones

        # A with initial parameters
        dense_shape = [self.number_of_states, self.number_of_states]
        values = tf.concat([self.A_initial_weights_for_trainable_parameters, self.A_initial_weights_for_constants], axis = 0)
        # values = tf.cast(values, tf.float32)

        A_init = tf.scatter_nd(self.A_indices, \
                                    values, \
                                    dense_shape)
        # A mask for all parameters
        A_mask = tf.scatter_nd(self.A_indices, \
                               [1] * len(self.A_indices), \
                               dense_shape)
        # softmax
        softmax_layer = tf.keras.layers.Softmax()

        # A = tf.scatter_nd([[1,0],[0,1]],[1,3], shape = [2,2])
        # mask = tf.scatter_nd([[1,0],[0,1],[1,1]],[1,1,1], shape = [2,2])
        # print(softmax_layer(A,mask))

        A_init_stochastic = softmax_layer(A_init, tf.cast(A_mask, tf.int32))

        # A mask for priors
        self.A_prior_indices = tf.cast(self.A_prior_indices, tf.int32)
        # tf.print("self.A_prior_indices", self.A_prior_indices, summarize = -1)
        # tf.print("len self.A_prior_indices", len(self.A_prior_indices), summarize = -1)
        # tf.print("[1.0] * len(self.A_prior_indices)", [1.0] * len(self.A_prior_indices), summarize = -1)

        prior_mask = tf.scatter_nd(self.A_prior_indices, \
                                   [1.0] * len(self.A_prior_indices), \
                                   dense_shape)

        # extract prior probs for init matrix
        self.A_prior_matrix = A_init_stochastic * prior_mask

        # write results to file for inspection
        if self.config.nCodons < 10:
            dir_path = f"{self.config.out_path}/output/{self.config.nCodons}codons/prior_calculation"
            self.A_as_dense_to_file(f"{dir_path}/A_init.csv", "dummy weight parameter", A = A_init, with_description = self.config.nCodons < 20)
            self.A_as_dense_to_file(f"{dir_path}/A_init_stochastic.csv", "dummy weight parameter", A = A_init_stochastic, with_description = self.config.nCodons < 20)
            self.A_as_dense_to_file(f"{dir_path}/A_prior_mask.csv", "dummy weight parameter", A = prior_mask, with_description = self.config.nCodons < 20)
            self.A_as_dense_to_file(f"{dir_path}/A_prior_matrix.csv", "dummy weight parameter", A = self.A_prior_matrix, with_description = self.config.nCodons < 20)
    ################################################################################
    def get_A_log_prior(self, A_kernel):
        self.A_prior_matrix = tf.cast(self.A_prior_matrix, dtype = self.config.dtype)
        alphas = self.A_prior_matrix * self.config.priorA - 1
        log_prior = tf.math.log(tf.sparse.to_dense(self.A(A_kernel)) + tf.cast(self.config.log_prior_epsilon, self.config.dtype))
        before_reduce_sum = (alphas * log_prior)
        return tf.math.reduce_sum(tf.gather_nd(before_reduce_sum, self.A_prior_indices))
################################################################################
    def get_B_prior_matrix(self):
        dense_shape = [self.number_of_emissions, self.number_of_states]
        self.B_prior_matrix = tf.scatter_nd(self.B_prior_indices, self.B_priors, shape = dense_shape)
        self.B_prior_matrix = tf.cast(self.B_prior_matrix, dtype = self.config.dtype)

        if self.config.nCodons < 10:
            dir_path = f"{self.config.out_path}/output/{self.config.nCodons}codons/prior_calculation"
            self.B_as_dense_to_file(f"{dir_path}/B_prior_matrix.csv", "dummy weight parameter", B = self.B_prior_matrix, with_description = self.config.nCodons < 20)
################################################################################
    def get_B_log_prior(self, B_kernel):
        alphas = self.B_prior_matrix * self.config.priorB - 1
        log_prior = tf.math.log(self.B(B_kernel) + self.config.log_prior_epsilon)
        before_reduce_sum = (alphas * log_prior)
        return tf.math.reduce_sum(tf.gather_nd(before_reduce_sum, self.B_prior_indices))
################################################################################
################################################################################
################################################################################
    def get_indices_for_emission_and_state(self, state, mask, x_bases_must_preceed, trainable = None):
        state_id = self.str_to_state_id(state)
        # if self.order_transformed_input and emissions[-1] == "X":
        indices = self.B_indices_for_trainable_parameters if trainable else self.B_indices_for_constant_parameters
        initial_parameters = self.B_initial_trainalbe_para_setting if trainable else self.B_initial_constant_para_setting
        if mask[-1] == "X":
            indices += [[self.emission_tuple_to_id("X"), state_id]]
            initial_parameters.append(1.0)
            return

        for ho_emission in self.get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state_id):
            p,i =  self.prior_and_initial_parameter_for_state_and_emission(state, self.emission_tuple_to_str(ho_emission))
            initial_parameters.append(i)
            index = [self.emission_tuple_to_id(ho_emission), state_id]
            if p != -1:
                self.B_priors.append(p)
                self.B_prior_indices.append(index)

            indices += [index]
################################################################################
    def make_preparations_for_B(self):
        self.B_indices_for_trainable_parameters = []
        self.B_indices_for_constant_parameters = []
        states_which_are_already_added = []
        self.B_priors = []
        self.B_prior_indices = []
        self.B_initial_trainalbe_para_setting = []
        self.B_initial_constant_para_setting = []
        def append_emission(state,
                            mask = "N",
                            x_bases_must_preceed = self.config.order,
                            trainable = True,
        ):
            states_which_are_already_added.append(state)
            self.get_indices_for_emission_and_state(state,
                                                    mask,
                                                    x_bases_must_preceed,
                                                    trainable = trainable)

        # here i add states + their emissions if a want to enforce a mask or want to make them not trainable, or x bases must preceed
        append_emission("left_intron","N", 0)
        # TODO these can be non trainable
        append_emission("A", "A", self.config.order, trainable = False)
        append_emission("AG", "AG", self.config.order, trainable = False)

        append_emission("c_0,0", "AGN")
        # for c_0,1 and c_0,2 i cant force AG to be emitted before, since the exon can be entered in phase 1 or 2

        # TODO these can be non trainable
        append_emission("G","G", trainable = False)
        append_emission("GT","GT", trainable = False)

        for i in range(self.config.donor_pattern_len):
            append_emission(f"do_{i}", "GT" + "N"*(i+1))

        append_emission("ter", "X",  trainable = False)

        states_that_werent_added_yet = set(self.state_to_id.keys()).difference(states_which_are_already_added)
        states_that_werent_added_yet = sorted(states_that_werent_added_yet)
        for state in states_that_werent_added_yet:
            append_emission(state)

        self.B_indices = self.B_indices_for_trainable_parameters + self.B_indices_for_constant_parameters



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
    # TODO falls dieses nicht get cann ich vielleicht tf dont convert machen
    def make_B_initial_weights(self):
        initial_parameters_for_weights = np.zeros(len(self.B_indices_for_trainable_parameters))
        initial_parameters_for_consts = np.zeros(len(self.B_indices_for_constant_parameters))

        concated_init_para_settings = tf.concat([self.B_initial_trainalbe_para_setting, self.B_initial_constant_para_setting], axis = 0)

        dense_shape = [self.number_of_emissions, self.number_of_states]
        emission_matrix = tf.scatter_nd(self.B_indices, concated_init_para_settings, dense_shape)

        if self.config.nCodons < 20:
            import os
            dir_name = f"{self.config.out_path}/output/{self.config.nCodons}codons/prior_calculation"
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            self.B_as_dense_to_file(f"{dir_name}/B_init_parameters_before_conversion.csv", self.B_initial_trainalbe_para_setting, with_description = True, B = emission_matrix)


        for state in range(tf.shape(emission_matrix)[1]):
            for emission_id_4 in range(0,tf.shape(emission_matrix)[0], self.config.alphabet_size):
                sum_with_out_zeros_and_negatives_ones = 0.0
                found_negative_ones = 0.0
                for emission_id in range(emission_id_4, emission_id_4 + self.config.alphabet_size):
                    if emission_matrix[emission_id, state] != -1:
                        sum_with_out_zeros_and_negatives_ones += tf.cast(emission_matrix[emission_id, state], tf.float32)
                    else:
                        found_negative_ones += 1
                # epsilon = 1e-5
                # if not found_negative_ones:
                #     assert abs(sum_with_out_zeros_and_negatives_ones - 1) < epsilon, f"sum of probs state {state}, emission_id_4 {emission_id_4} is not equal to one (={sum_with_out_zeros_and_negatives_ones})"

                # if found_negative_ones:
                #     assert 1 - sum_with_out_zeros_and_negatives_ones > epsilon

                for emission_id in range(emission_id_4, emission_id_4 + self.config.alphabet_size):
                    if emission_matrix[emission_id, state] == -1:
                        weight = tf.math.log((1-sum_with_out_zeros_and_negatives_ones)/found_negative_ones)
                    else:
                        weight = tf.math.log(tf.cast(emission_matrix[emission_id, state], tf.float32))
                    indx = (emission_id, state)
                    if list(indx) in self.B_indices_for_trainable_parameters:
                        initial_parameters_for_weights[self.B_weight_tuple_id(indx)] = weight
                    elif list(indx) in self.B_indices_for_constant_parameters:
                        initial_parameters_for_consts[self.B_const_tuple_id(indx)] = weight
                    # else:
                    #     print(f"index {indx} not found in make_B_initial_weights()")
                    #     exit()
        initial_parameters_for_weights = np.array(initial_parameters_for_weights, np.float32)
        initial_parameters_for_consts = np.array(initial_parameters_for_consts, np.float32)
        return initial_parameters_for_weights, initial_parameters_for_consts

################################################################################

    def B(self, weights):
        # consts = tf.cast([1.0] * len(self.B_indices_for_constants), dtype = self.config.dtype)
        try:
            consts = self.B_initial_weights_for_constant_parameters
        except:
            consts = self.B_initial_constant_para_setting
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
    # def export_to_dot_and_png(self, A_weights, B_weights, out_path = "this is still hard coded"):
    #     # TODO: add I parameters???
    #     import numpy as np
    #     n_labels = self.number_of_emissions ** (self.config.order + 1)
    #     nCodons = self.config.nCodons
    #
    #     A = self.A(A_weights) if self.A_is_dense else tf.sparse.to_dense(self.A(A_weights))
    #     B = self.B(B_weights) if self.B_is_dense else tf.sparse.to_dense(self.B(B_weights))
    #
    #     B_reshaped = tf.reshape(B, shape = (-1, self.config.alphabet_size, self.number_of_states))
    #     B_argmax = np.argmax(B_reshaped, axis = 1)
    #
    #     id_to_base = {0:"A", 1:"C",2:"G",3:"T",4:"I",5:"Ter"}
    #     with open(f"output/{nCodons}codons/graph.{nCodons}codons.gv", "w") as graph:
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
    #     run(f"dot -Tpng output/{nCodons}codons/graph.{nCodons}codons.gv -o output/{nCodons}codons/graph.{nCodons}codons.png")

    def I_as_dense_to_json_file(self, path, weights):
        with open(path, "w") as out_file:
            json.dump(self.I(weights).numpy().tolist(), out_file)

    # TODO: or do i want to have these functions in the cell, such that i dont have to pass the weights?
    def A_as_dense_to_str(self, weights, with_description = False, A = None):
        if A == None:
            A = self.A(weights) if self.A_is_dense else tf.sparse.to_dense(self.A(weights))
        result = ""
        if with_description:
            result += " "
            for to_state in range(self.number_of_states):
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

    def A_as_dense_to_file(self, path, weights, with_description = False, A = None):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        with open(path, "w") as out_file:
            out_file.write(self.A_as_dense_to_str(weights, with_description, A = A))

    def A_as_dense_to_json_file(self, path, weights):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        with open(path, "w") as out_file:
            A = self.A(weights) if self.A_is_dense else tf.sparse.to_dense(self.A(weights))
            json.dump(A.numpy().tolist(), out_file)

    def B_as_dense_to_str(self, weights, with_description = False, B = None):
        if B == None:
            B = self.B(weights) if self.B_is_dense else tf.sparse.to_dense(self.B(weights))
        result = ""
        if with_description:
            result += " "
            for to_state in range(self.number_of_states):
                result += self.state_id_to_str(to_state)
                result += " "
            result += "\n"

        for emission_id, row in enumerate(B):
            if with_description:
                result += self.emission_id_to_str(emission_id) + " "
            for entry in row:
                result += str(entry.numpy())
                result += " "
            result += "\n"
        return result

    def B_as_dense_to_file(self, path, weights, with_description = False, B = None):
        # TODO does this use the option -p??
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))

        with open(path, "w") as out_file:
            out_file.write(self.B_as_dense_to_str(weights, with_description, B = B))

    def B_as_dense_to_json_file(self, path, weights):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))

        with open(path, "w") as out_file:
            B = self.B(weights) if self.B_is_dense else tf.sparse.to_dense(self.B(weights))
            json.dump(B.numpy().tolist(), out_file)

if __name__ == '__main__':
    from Config import Config
    config = Config("main_programm_dont_interfere")
    model = My_Model(config)
    import numpy as np
