#!/usr/bin/env python3
from Model import Model
import re
from itertools import product
import tensorflow as tf
import json

class My_Model(Model):

    # this overwrites the init from Model. alternatively i can omit it
    def __init__(self, config):
        Model.__init__(self, config)
        self.number_of_states_per_intron = config.pattern_length_before_intron_loop + 1 + config.pattern_length_after_intron_loop

        # =================> states <============================================
        self.number_of_states = self.get_number_of_states()
        self.state_id_description_list = self.get_state_id_description_list()

        # =================> emissions <========================================
        self.emissions_state_size = self.get_emissions_state_size()
        self.number_of_emissions = self.get_number_of_emissions()

        self.id_to_emi = self.get_dicts_for_emission_tuple_and_id_conversion()[1] # these are dicts
        self.emi_to_id = self.get_dicts_for_emission_tuple_and_id_conversion()[0]

        self.A_is_dense = config.A_is_dense
        self.A_is_sparse = config.A_is_sparse
        self.B_is_dense = config.B_is_dense
        self.B_is_sparse = config.B_is_sparse

        # init for reg in layer
        self.A_indices_begin_inserts
        self.A_indices_continue_inserts
        self.A_indices_deletes


        self.I_indices = self.I_indices()

        self.A_indices_for_weights = self.A_indices_for_weights()
        self.A_indices_for_constants = self.A_indices_for_constants()
        self.A_indices = self.A_indices_for_weights + self.A_indices_for_constants

        if self.config.my_initial_guess_for_parameters:
            self.A_my_initial_guess_for_parameters = self.get_A_my_initial_guess_for_parameters()

        self.A_consts = self.get_A_consts()

        self.B_indices_for_weights = self.B_indices_for_weights()
        self.B_indices_for_constants = self.B_indices_for_constants()
        self.B_indices = self.B_indices_for_weights + self.B_indices_for_constants

        if config.use_weights_for_consts:
            self.B_indices = sorted(self.B_indices)

        shape = (self.number_of_emissions, self.number_of_states)
        B_indices_complement = tf.where(tf.ones(shape, dtype = tf.float32) - tf.scatter_nd(self.B_indices, [1.0] * len(self.B_indices), shape = shape))
        self.B_indices_complement = tf.cast(B_indices_complement, dtype = tf.int32)

    # =================> states <===============================================
    def get_number_of_states(self):
        number_of_states = 1
        # start
        number_of_states += 3
        # intron after ATG
        number_of_states += self.number_of_states_per_intron
        # codons
        number_of_states += 3 * self.config.nCodons
        # introns after codon state
        number_of_states += 3 * self.config.nCodons * self.number_of_states_per_intron
        # codon inserts
        number_of_states += 3 * (self.config.nCodons + 1)
        # introns after insert state
        number_of_states += 3 * (self.config.nCodons + 1) * self.number_of_states_per_intron
        # stop
        number_of_states += 3
        # ig 3'
        number_of_states += 1
        # terminal
        number_of_states += 1

        self.total_number_of_introns = (1 + 3 * self.config.nCodons  + 3 * (self.config.nCodons + 1))
        self.total_number_of_intron_states = self.number_of_states_per_intron * self.total_number_of_introns
        print("total_number_of_intron_states", self.total_number_of_intron_states)
        return number_of_states

    def get_state_id_description_list(self):
        # if this is changed, also change state_is_third_pos_in_frame()
        states = re.split(" ", "ig5' stA stT stG")
        states += ["c_" + str(i) + "," + str(j) for i in range(self.config.nCodons) for j in range(3)]
        states += re.split(" ", "stop1 stop2 stop3 ig3'")
        states += ["i_" + str(i) + "," + str(j) for i in range(self.config.nCodons+1) for j in range(3)]
        states += [f"int_{after},{j}" for after in range(self.total_number_of_introns) for j in range(self.number_of_states_per_intron)]
        states += ["ter"]
        for i, state in enumerate(states):
            print(i, state)
        return states

    def state_id_to_str(self, id):
        return self.state_id_description_list[id]

    def str_to_state_id(self, s):
        # TODO convert state_id_description_list to dict to make this faster
        try:
            return self.state_id_description_list.index(s)
        except:
            return -1

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
        if self.config.use_weights_for_consts:
            return len(self.A_indices_for_weights) + len(self.A_indices_for_constants)
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

    def A_indices_for_constants(self):
        indicies_for_constant_parameters = []
        indices_for_trainable_parameters = []
        def append_transition(s1, s2, l = None, trainable = False):
            if l == None:
                l = [[self.str_to_state_id(s1), self.str_to_state_id(s2)]]
            if trainable:
                indices_for_trainable_parameters += l
            else:
                indicies_for_constant_parameters += l

        append_transition(l = self.A_indices_ig5, trainable = not self.config.ig5_const_transition)
        append_transition("stA", "stT")
        append_transition("stT", "stG")

        append_transition(l = self.A_indices_enter_next_codon, trainable = True)

        for i in range(self.config.nCodons):
            append_transition(f"c_{i},0", f"c_{i},1")
            append_transition(f"c_{i},1", f"c_{i},2")

        append_transition(l = self.A_indices_begin_inserts, trainable = True)
        for i in range(self.config.nCodons + 1):
            append_transition(f"i_{i},0", f"i_{i},1")
            append_transition(f"i_{i},1", f"i_{i},2")
        append_transition(l = self.A_indices_end_inserts, trainable = True)
        append_transition(l = self.A_indices_continue_inserts, trainable = True)

        append_transition(l = self.A_indices_enter_stop, trainable = True)

        append_transition(l = self.A_indices_deletes, trainable = True)

        append_transition("stop1", "stop2")
        append_transition("stop1", "stop3")

        append_transition("stop3", "ig3'")

        append_transition("ig3'", "ig3'", trainable = not self.ig3_const_transition)
        append_transition("ig3'", "ter", trainable =  not self.ig3_const_transition)
        append_transition("ter", "ter")

        # intron after start
        state_before_intron = self.str_to_state_id("stG")
        intron_start_id = self.str_to_state_id("int_0,0")
        append_transition(l = [[state_before_intron, intron_start_id]], trainable = True)
        for id_in_before_loop_pattern in range(self.config.pattern_length_before_intron_loop):
            append_transition(l = [[intron_start_id + id_in_before_loop_pattern, intron_start_id + id_in_before_loop_pattern + 1]], trainable = True)
        # loop
        intron_loop_id = intron_start_id + self.config.pattern_length_before_intron_loop
        append_transition(l = [[intron_loop_id, intron_loop_id]], trainable = True)
        # exit loop, needs_weight
        for id_in_after_loop_pattern in range(self.config.pattern_length_after_intron_loop):
            if id_in_after_loop_pattern == 0: # exit intron loop
                append_transition( l = [[intron_loop_id, intron_loop_id + 1]], trainable = True)
            else:
                append_transition(l = [[intron_loop_id + id_in_after_loop_pattern, intron_loop_id + id_in_after_loop_pattern + 1]])
        # exit intron
        # to first codon or first insert, these need a weight
        append_transition(l = [[intron_loop_id + self.config.pattern_length_after_intron_loop, self.str_to_state_id("c_0,0")]], trainable = True)
        append_transition(l = [[intron_loop_id + self.config.pattern_length_after_intron_loop, self.str_to_state_id("i_0,0")]], trainable = True)


        for codon_id in range(self.config.nCodons + 1):
            for c_or_i in ([0,1] if codon_id != self.config.nCodons else [1]): # only introns have id up tp nCodons
                for codon_state_id_in_current_codon in [0,1,2]:
                    state_before_intron = self.str_to_state_id(f"{'ci'[c_or_i]}_{codon_id},{codon_state_id_in_current_codon}")




                    # TODO: maybe name introns after the state they come from




                    #                                             atg
                    intron_start_id = self.str_to_state_id(f"int_{1+  codon_id * 3 + codon_state_id_in_current_codon + 3*self.config.nCodons * c_or_i},0")

                    append_transition(l = [[state_before_intron, intron_start_id]], trainable = True)
                    for id_in_before_loop_pattern in range(self.config.pattern_length_before_intron_loop):
                        append_transition(l = [[intron_start_id + id_in_before_loop_pattern, intron_start_id + id_in_before_loop_pattern + 1]])
                    intron_loop_id = intron_start_id + self.config.pattern_length_before_intron_loop
                    append_transition(l = [[intron_loop_id, intron_loop_id]], trainable = True)
                    # exit loop, needs_weight
                    for id_in_after_loop_pattern in range(self.config.pattern_length_after_intron_loop):
                        # this needs a weight
                        if id_in_after_loop_pattern == 0:
                            indices += [[intron_loop_id, intron_loop_id + 1]]
                        else: # theses dont need a weight
                            indices += [[intron_loop_id + id_in_after_loop_pattern, intron_loop_id + id_in_after_loop_pattern + 1]]

                    # exit intron
                    # here, no weight is needed
                    state_before_exit = intron_loop_id + self.config.pattern_length_after_intron_loop
                    if codon_state_id_in_current_codon in [0,1]:
                        state_after_intron = self.str_to_state_id(f"{'ci'[c_or_i]}_{codon_id},{codon_state_id_in_current_codon + 1}")
                        indices += [[state_before_exit, state_after_intron]]

                    else:# here, weights are needed
                        # transition to stop
                        if codon_id == self.config.nCodons -1 + c_or_i:
                            state_after_intron = self.str_to_state_id("stop1")
                            indices += [[state_before_exit, state_after_intron]]
                        else:
                            state_after_intron = self.str_to_state_id(f"c_{codon_id + 1 - c_or_i},0")
                            indices += [[state_before_exit, state_after_intron]]
                        state_after_intron = self.str_to_state_id(f"i_{codon_id},0")
                        indices += [[state_before_exit, state_after_intron]]

        return indices
    #
    # def A_indices_for_weights(self): # no shared parameters
    #
    #     indices = []
    #     # from ig 5'
    #     if not self.config.ig5_const_transition:
    #         indices += self.A_indices_ig5
    #
    #     indices += self.A_indices_enter_next_codon
    #
    #     if not self.config.no_inserts:
    #         indices += self.A_indices_begin_inserts
    #         indices += self.A_indices_end_inserts
    #         indices += self.A_indices_continue_inserts
    #
    #     indices += self.A_indices_enter_stop
    #
    #     if not self.config.no_deletes:
    #         indices += self.A_indices_deletes
    #
    #     if not self.config.ig3_const_transition:
    #         indices += self.A_indices_ig3
    #
    #     return indices
    @property
    def A_indices_ig5(self):
        ig5 = self.str_to_state_id("ig5'")
        startA = self.str_to_state_id("stA")
        return [[ig5,ig5], [ig5,startA]]
    @property
    def A_indices_ig3(self):
        ig3 = self.str_to_state_id("ig3'")
        ter = self.str_to_state_id("ter")
        return [[ig3,ig3], [ig3,ter]]
    @property
    def A_indices_enter_next_codon(self):# including ATG -> first codon
        indices = [[self.str_to_state_id("stG"), self.str_to_state_id("c_0,0")]]
        for i in range(self.config.nCodons-1):
            indices += [[self.str_to_state_id(f"c_{i},2"), self.str_to_state_id(f"c_{i+1}, 0")]]
        return indices
    @property
    def A_indices_enter_stop(self):
        return [[self.str_to_state_id(f"c_{self.config.nCodons-1},2"),self.str_to_state_id("stop1")]]
    @property
    def A_indices_deletes(self):
        indices = []
        # from stG
        for to_codon in range(1,self.config.nCodons):
            indices += [[self.str_to_state_id("stG"), self.str_to_state_id(f"c_{to_codon},0")]]
        indices += [[self.str_to_state_id("stG"), self.str_to_state_id("stop1")]]
        # from codons
        for after_codon in range(self.config.nCodons):
            for to_codon in range(after_codon + 1, self.config.nCodons):
                indices += [[self.str_to_state_id(f"c_{after_codon},2"), self.str_to_state_id(f"c_{to_codon},0")]]
            # to stop codon
            indices += [[self.str_to_state_id(f"c_{after_codon},2"), self.str_to_state_id("stop1")]]

        if self.config.deletions_and_insertions_not_only_between_codons:
            pass
            # from atg to inserts
            # from inserts to codons
            # from codons to inserts
            # from inserts to inserts
            # from atg to intron
            # from intron of codon to codon/insert
            # from intron of insert to insert/codon


        # i_delete = [3 + i*3 for i in range(self.config.nCodons) for j in range(self.config.nCodons-i)]
        # j_delete = [4 + j*3 for i in range(1,self.config.nCodons+1) for j in range(i,self.config.nCodons+1)]
        # nCodons = 4: [[3, 7], [3, 10], [3, 13], [3, 16], [6, 10], [6, 13], [6, 16], [9, 13], [9, 16], [12, 16]]
        return [[i,j] for i,j in zip(i_delete, j_delete)]
    @property
    def A_indices_begin_inserts(self):# including ATG -> first insert
        indices = [[self.str_to_state_id("stG"), self.str_to_state_id("i_0,0")]]
        for i in range(self.config.nCodons):
            indices += [[self.str_to_state_id(f"c_{i},2"), self.str_to_state_id(f"i_{i+1},2")]]
        return indices
    @property
    def A_indices_end_inserts(self): # including last insert -> stop1
        indices = []
        for i in range(self.config.nCodons):
            indices += [[self.str_to_state_id(f"i_{i},2"), self.str_to_state_id(f"c_{i},0")]]
        indices += [[self.str_to_state_id(f"i_{self.config.nCodons},2"), self.str_to_state_id("stop1")]]
        return indices
    @property
    def A_indices_continue_inserts(self):
        for i in range(self.config.nCodons +1):
            indices += [[self.str_to_state_id(f"i_{i},2"), self.str_to_state_id(f"i_{i},0")]]
        return indices
################################################################################
    def A_indices(self):
        return self.A_indices_for_weights + self.A_indices_for_constants
################################################################################
    def get_A_consts(self):
        if self.config.ig5_const_transition:
            # return tf.cast(tf.concat([[5.0,1], [1.0] * (len(self.A_indices_for_constants) -2)], axis = 0),dtype = self.config.dtype)
            if self.config.ig3_const_transition:
                return tf.cast(tf.concat([[self.config.ig5_const_transition,1], [1.0] * (len(self.A_indices_for_constants) -4), [self.config.ig5_const_transition,1]], axis = 0),dtype = self.config.dtype)
            else:
                return tf.cast(tf.concat([[self.config.ig5_const_transition,1], [1.0] * (len(self.A_indices_for_constants) -2)], axis = 0),dtype = self.config.dtype)
        return tf.cast([1.0] * len(self.A_indices_for_constants), dtype = self.config.dtype)
################################################################################
    def get_A_my_initial_guess_for_parameters(self):
        # für die ordnung die asserts vielleicht nach Config.py verschieben
        assert self.config.ig5_const_transition, "when using my initial guess for parameters also pass ig5_const_transition"
        assert self.config.ig3_const_transition, "when using my initial guess for parameters also pass ig3_const_transition"
        assert not self.config.no_deletes, "when using my initial guess for parameters do not pass no_deletes"
        assert not self.config.no_inserts, "when using my initial guess for parameters do not pass no_inserts"
        assert not self.config.use_weights_for_consts, "when using my initial guess for parameters do not pass use_weights_for_consts"

        my_weights = []
        # enter codon
        my_weights += [4] * len(self.A_indices_enter_next_codon)

        # begin_inserts
        my_weights += [1] * len(self.A_indices_begin_inserts)

        # end inserts
        my_weights += [4] * len(self.A_indices_end_inserts)

        # continue inserts
        my_weights += [1] * len(self.A_indices_continue_inserts)

        # enter stop
        my_weights += [4]

        # deletes                                  2 is just an arbitrary factor
        my_weights += [1 - j/2 for i in range(self.config.nCodons) for j in range(self.config.nCodons - i)]

        # cast
        # my_weights = tf.cast(my_weights, dtype = self.config.dtype)

        return my_weights

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
        if s [-1] == "2" and s != "stop2" and s != "ter2":
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
    def get_indices_for_emission_and_state(self, indices, state, mask, x_bases_must_preceed):
        # if self.order_transformed_input and emissions[-1] == "X":
        if mask[-1] == "X":
            indices += [[self.emission_tuple_to_id("X"), state]]
            return

        for ho_emission in self.get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state):
            indices += [[self.emission_tuple_to_id(ho_emission), state]]
################################################################################
    def B_indices_for_weights(self):
        nCodons = self.config.nCodons
        indices = []

        ig5 = "N" if not self.config.forced_gene_structure else "K" # T and G
        coding = "N" if not self.config.forced_gene_structure else "M" # A and C
        ig3 = ig5

        # ig 5'
        self.get_indices_for_emission_and_state(indices,0,ig5,0)
        # start a
        self.get_indices_for_emission_and_state(indices,1,"A",1)
        # start t
        self.get_indices_for_emission_and_state(indices,2,"AT",2)

        # codon_11
        self.get_indices_for_emission_and_state(indices,4,"ATG" + coding,2)
        # codon_12
        self.get_indices_for_emission_and_state(indices,5,"ATG" + coding*2,2)
        # all other codons
        for state in range(6, 6 + nCodons*3 -2):
            self.get_indices_for_emission_and_state(indices,state,coding,2)
        # stop
        self.get_indices_for_emission_and_state(indices,4 + nCodons*3,"T", self.config.order)
        self.get_indices_for_emission_and_state(indices,5 + nCodons*3,"TA", self.config.order)
        self.get_indices_for_emission_and_state(indices,5 + nCodons*3,"TG", self.config.order)
        # ig 3'
        self.get_indices_for_emission_and_state(indices,7 + nCodons*3,ig3, self.config.order)
        # inserts
        for state in range(8 + nCodons*3, 8 + nCodons*3 + (nCodons + 1)*3):
            self.get_indices_for_emission_and_state(indices,state,coding, self.config.order)

        self.get_indices_for_emission_and_state(indices,8 + nCodons*3 + (nCodons+1)*3,"X", self.config.order)


        return indices
################################################################################
    def B_indices_for_constants(self):
        nCodons = self.config.nCodons
        indices = []

        self.get_indices_for_emission_and_state(indices,3,"ATG",2)
        self.get_indices_for_emission_and_state(indices,6 + nCodons*3,"TAA", self.config.order)
        self.get_indices_for_emission_and_state(indices,6 + nCodons*3,"TAG", self.config.order)
        if self.config.order > 0:
            # bc then only the third pos is codon is of importance, and then "A" would be added twice
            self.get_indices_for_emission_and_state(indices,6 + nCodons*3,"TGA", self.config.order)

        return indices
################################################################################
    def B_indices(self):
        return self.B_indices_for_weights() + self.B_indices_for_constants()
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
            consts = self.A_consts
            values = tf.concat([weights, consts], axis = 0)
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
            # print("b dense")
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
    def export_to_dot_and_png(self, A_weights, B_weights, out_path = "this is still hard coded"):
        # TODO: add I parameters???
        import numpy as np
        n_labels = self.number_of_emissions ** (self.config.order + 1)
        nCodons = self.config.nCodons

        A = self.A(A_weights) if self.A_is_dense else tf.sparse.to_dense(self.A(A_weights))
        B = self.B(B_weights) if self.B_is_dense else tf.sparse.to_dense(self.B(B_weights))

        B_reshaped = tf.reshape(B, shape = (-1, self.config.alphabet_size, self.number_of_states))
        B_argmax = np.argmax(B_reshaped, axis = 1)

        id_to_base = {0:"A", 1:"C",2:"G",3:"T",4:"I",5:"Ter"}
        with open(f"output/{nCodons}codons/graph.{nCodons}codons.gv", "w") as graph:
            graph.write("DiGraph G{\nrankdir=LR;\n")
            # graph.write("nodesep=0.5; splines=polyline;")
            for from_state, row in enumerate(A):
                from_state_str = self.state_id_to_str(from_state)
                graph.write("\"" + from_state_str + "\"\n") #  this was to_state before

                graph.write("[\n")
                graph.write("\tshape = none\n")
                graph.write("\tlabel = <<table border=\"0\" cellspacing=\"0\"> \n")
                try:
                    color = {"c_":"teal", "i_": "crimson"}[from_state_str[0:2]]
                except:
                    color = "white"

                graph.write(f"\t\t<tr><td port=\"port1\" border=\"1\" bgcolor=\"{color}\">" + from_state_str + "</td></tr>\n")

                for k, most_likely_index in enumerate(B_argmax[:,from_state]):
                    emission_id = most_likely_index + k * self.config.alphabet_size
                    emission_str = self.emission_id_to_str(emission_id)
                    emi_prob = str(np.round(B[emission_id, from_state].numpy(),4))
                    graph.write(f"\t\t<tr><td port=\"port{k+2}\" border=\"1\">({emission_str + ' ' +emi_prob})</td></tr>\n" )
                graph.write("\t </table>>\n")
                graph.write("]\n")

                for to_state, prob in enumerate(row):
                    to_state_str = self.state_id_to_str(to_state)
                    if prob > 0:
                        prob = prob.numpy()
                        graph.write(f"\"{from_state_str}\" -> \"{to_state_str}\" [label = {str(np.round(prob, 4))[:6]} fontsize=\"{30*prob + 5}pt\"]\n")

            graph.write("}")
        # run(f"cat graph.{nCodons}codons.gv")
        from Utility import run
        run(f"dot -Tpng output/{nCodons}codons/graph.{nCodons}codons.gv -o output/{nCodons}codons/graph.{nCodons}codons.png")


    def I_as_dense_to_json_file(self, path, weights):
        with open(path, "w") as out_file:
            json.dump(self.I(weights).numpy().tolist(), out_file)

    # TODO: or do i want to have these functions in the cell, such that i dont have to pass the weights?
    def A_as_dense_to_str(self, weights, with_description = False):
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

    def A_as_dense_to_file(self, path, weights, with_description = False):
        with open(path, "w") as out_file:
            out_file.write(self.A_as_dense_to_str(weights, with_description))

    def A_as_dense_to_json_file(self, path, weights):
        with open(path, "w") as out_file:
            A = self.A(weights) if self.A_is_dense else tf.sparse.to_dense(self.A(weights))
            json.dump(A.numpy().tolist(), out_file)

    def B_as_dense_to_str(self, weights, with_description = False):
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
