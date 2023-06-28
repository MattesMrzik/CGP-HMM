#!/usr/bin/env python3
from Model import Model
import re
from itertools import product
import json
import numpy as np
import os
import tensorflow as tf

def non_class_A_log_prior(A, prior_matrix, prior_indices):
        alphas = tf.gather_nd(prior_matrix, prior_indices)
        tf.debugging.Assert(tf.math.reduce_all(alphas != 0), [alphas], name = "some_A_alphas_are_zero")
        tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(alphas)), [alphas], name = "some_A_alphas_are_not_finite")


        ps = tf.gather_nd(A, prior_indices)
        # this shouldnt be a formal requirement, but would result in inf loglikelihood
        tf.debugging.Assert(tf.math.reduce_all(ps != 0), [ps], name = "some_A_ps_are_zero")
        tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(ps)), [ps], name = "some_A_ps_are_not_finite")

        def is_greater_0(tensor):
            ones_mask = tf.math.greater(tensor, 0)
            return ones_mask

        # this should not be a formal requirement, since this is indipendent of the parameters
        # it is still done to get sensible results for the prior
        def lbeta_row_positive_values(row):
            return tf.math.lbeta(tf.boolean_mask(row, is_greater_0(row)))

        log_z = tf.map_fn(lbeta_row_positive_values, prior_matrix)

        log_z = tf.math.reduce_sum(tf.boolean_mask(log_z, tf.math.is_finite(log_z)))

        before_reduce_sum = tf.math.xlogy((alphas-1), ps)
        return tf.math.reduce_sum(before_reduce_sum) - log_z

def non_class_B_log_prior(B, prior_matrix, prior_indices, alphabet_size):
        alphas = tf.gather_nd(prior_matrix, prior_indices)

        tf.debugging.Assert(tf.math.reduce_all(alphas != 0), [alphas, tf.boolean_mask(prior_indices, alphas == 0)], name = "some_B_alphas_are_zero")
        tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(alphas)), [alphas], name = "some_B_alphas_are_not_finite")

        ps = tf.gather_nd(B, prior_indices)

        # this shouldnt be a formal requirement, but would result in inf loglikelihood
        tf.debugging.Assert(tf.math.reduce_all(ps != 0), [ps, tf.boolean_mask(prior_indices, ps == 0)], name = "some_B_ps_are_zero")
        tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(ps)), [ps], name = "some_B_ps_are_not_finite")

        prior_matrix_transposed = tf.transpose(prior_matrix)
        prior_matrix_transposed_reshaped = tf.reshape(prior_matrix_transposed, (-1, alphabet_size))

        def is_greater_0(tensor):
            ones_mask = tf.math.greater(tensor, 0)
            return ones_mask

        def lbeta_row_positive_values(row):
            return tf.math.lbeta(tf.boolean_mask(row, is_greater_0(row)))


        log_z = tf.map_fn(lbeta_row_positive_values, prior_matrix_transposed_reshaped)

        log_z = tf.math.reduce_sum(tf.boolean_mask(log_z, tf.math.is_finite(log_z)))

        before_reduce_sum = tf.math.xlogy((alphas-1), ps)

        return tf.math.reduce_sum(before_reduce_sum) - log_z

class CGP_HMM(Model):

    def __init__(self, config):
        Model.__init__(self, config)
        self.is_prepared = False
        self.is_made = False

    def prepare_model(self):
        # =================> states <============================================
        self.id_to_state, self.state_to_id = self.get_state_id_description_dicts()
        self.number_of_states = self.get_number_of_states()

        # =================> emissions <========================================
        self.emissions_state_size = self.get_emissions_state_size()
        self.number_of_emissions = self.get_number_of_emissions()
        # dictionaries to convert between emission tuples and ids
        self.emi_to_id, self.id_to_emi = self.get_dicts_for_emission_tuple_and_id_conversion()
        self.A_is_dense = self.config.A_is_dense
        self.A_is_sparse = self.config.A_is_sparse
        self.B_is_dense = self.config.B_is_dense
        self.B_is_sparse = self.config.B_is_sparse
        self.is_prepared = True

    def make_model(self):

        from Prior import Prior
        self.prior = Prior(self.config)

        # I
        self.I_indices = self.get_I_indices()

        # A
        self.make_preparations_for_A()

        # this holds the concentration paramteres for the prior of the transition matrix
        self.get_A_prior_matrix()

        # B
        self.make_preparations_for_B()
        self.get_B_prior_matrix()

        shape = (self.number_of_emissions, self.number_of_states)
        B_indices_complement = tf.where(tf.ones(shape, dtype = tf.float32) - tf.scatter_nd(self.B_indices, [1.0] * len(self.B_indices), shape = shape))
        self.B_indices_complement = tf.cast(B_indices_complement, dtype = tf.int32)

        # convert indices in B to ids
        self.B_weight_index_tuple_to_id_conversion_dict = self.get_index_tuple_to_id_conversion_dict(self.B_indices_for_trainable_parameters)
        self.B_const_index_tuple_to_id_conversion_dict = self.get_index_tuple_to_id_conversion_dict(self.B_indices_for_constant_parameters)

        self.B_initial_weights_for_trainable_parameters, \
        self.B_initial_weights_for_constant_parameters = self.make_B_initial_weights()


        # write prior concentration parameters to file
        if self.config.nCodons < 20:
            dir_name = f"{self.config.current_run_dir}/prior_calculation"
            import os
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            self.B_as_dense_to_file(f"{dir_name}/B_init_parameters_after_conversion.csv", self.B_initial_weights_for_trainable_parameters, with_description = True)


        self.is_made = True

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
        # gets the names of the states and returns two dictionaries
        # to convert between state names and state ids
        states = ["left_intron"]

        states += [f"ak_{i}" for i in range(self.config.akzeptor_pattern_len)]
        states += ["A", "AG"]
        codons = ["c_" + str(i) + "," + str(j) for i in range(self.config.nCodons) for j in range(3)]
        states += codons

        inserts = ["i_" + str(i) + "," + str(j) for i in range(1, self.config.nCodons) for j in range(3)]
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

    def state_id_to_str(self, id : int) -> str:
        return self.id_to_state[id]

    def str_to_state_id(self, s : str) -> int:
        return self.state_to_id[s]

    # =================> emissions <============================================
    def get_emissions_state_size(self):
        # without terminal symbol
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


        return emi_to_id, id_to_emi

    def emission_tuple_to_id(self, emission_tuple):
    # emission is either a tuple like [2,1,3] or "X"
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

################################################################################
################################################################################
################################################################################
    def A_kernel_size(self):
        return len(self.A_indices_for_weights)

    def B_kernel_size(self):
        return len(self.B_indices_for_trainable_parameters)
################################################################################
################################################################################
################################################################################
    def get_I_indices(self):
        return [[0,0]]
################################################################################
################################################################################
################################################################################
    def make_preparations_for_A(self):
        single_high_prob_kernel = self.config.single_high_prob_kernel
        indicies_for_constant_parameters = []
        indices_for_trainable_parameters = []
        initial_weights_for_consts = []
        initial_weights_for_trainable_parameters = []
        self.A_prior_indices =[]

        def append_transition(s1 = None, s2 = None, l = None, trainable = True, initial_weights = None, use_as_prior = False):
            if l == None: # -> make l and list containing single weight
                assert s1 != None and s2 != None, "s1 and s2 must be != None if l = None"
                if initial_weights == None:
                    initial_weights = 0
                assert type(initial_weights) in [int, float], f"if you append a single transition, you must pass either no initial weight or int or float, but it is {type(initial_weights)}"
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
                if trainable:
                    indices_for_trainable_parameters.append(entry)
                    initial_weights_for_trainable_parameters.append(weight)
                else:
                    indicies_for_constant_parameters.append(entry)
                    initial_weights_for_consts.append(weight)



        # beginning of transition matrix definition
        # indicies in matrix,
        # initial weights,
        # trainable or not
        # regulate with prior or not

        append_transition("left_intron", "right_intron", \
                        trainable = not self.config.exon_skip_const, \
                        initial_weights = self.config.exon_skip_init_weight)

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
        append_transition(l = self.A_indices_begin_inserts, use_as_prior=True, initial_weights = [-0.2] * len(self.A_indices_begin_inserts))

        for i in range(1, self.config.nCodons):
            append_transition(f"i_{i},0", f"i_{i},1", trainable = False)
            append_transition(f"i_{i},1", f"i_{i},2", trainable = False)

        append_transition(l = self.A_indices_end_inserts,      initial_weights = [0] * len(self.A_indices_end_inserts), use_as_prior=True)
        append_transition(l = self.A_indices_continue_inserts, initial_weights = [-1] * len(self.A_indices_end_inserts), use_as_prior=True)

        # deletes
        A_indices_deletes, A_init_weights_normal_deletes = self.A_indices_and_init_weights_deletes
        append_transition(l = A_indices_deletes, initial_weights = [x + 0.1 for x in A_init_weights_normal_deletes], use_as_prior=True)


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

        # I think i can only set initial weights wiht float32,
        initial_weights_for_consts = np.array(initial_weights_for_consts, dtype = np.float32)
        initial_weights_for_trainable_parameters = np.array(initial_weights_for_trainable_parameters, dtype = np.float32)


        self.A_indices_for_weights = indices_for_trainable_parameters
        self.A_indices_for_constants = indicies_for_constant_parameters
        self.A_initial_weights_for_trainable_parameters = initial_weights_for_trainable_parameters
        self.A_initial_weights_for_constants = initial_weights_for_consts
        self.A_indices = np.concatenate([self.A_indices_for_weights, self.A_indices_for_constants])

    @property
    def A_indices_enter_next_codon(self):
        indices = []
        for i in range(self.config.nCodons-1):
            indices += [[self.str_to_state_id(f"c_{i},2"), self.str_to_state_id(f"c_{i+1},0")]]
        return indices

    # deletes
    @property
    def A_indices_and_init_weights_deletes(self):
        indices = []
        init_weights = []
        # from codons
        for after_codon in range(self.config.nCodons):
            for to_codon in range(after_codon + 2, self.config.nCodons):
                indices += [[self.str_to_state_id(f"c_{after_codon},2"), self.str_to_state_id(f"c_{to_codon},0")]]
                init_weights += [-(to_codon - after_codon)/self.config.diminishing_factor]
        return indices, init_weights

    # currently not used
    @property
    def A_indices_deletes_after_intron_to_codon(self):
        indices = []
        for i in range(1,self.config.nCodons):# including to the last codon
            for j in range(3):
                indices += [[self.str_to_state_id("AG"), self.str_to_state_id(f"c_{i},{j}")]]
        return indices

    # currently not used
    @property
    def A_indices_deletes_after_codon_to_intron(self):
        indices = []
        for i in range(self.config.nCodons-1):# including to the first codon
            for j in range(3):
                indices += [[self.str_to_state_id(f"c_{i},{j}"), self.str_to_state_id("G")]]
        return indices

    # currently not used
    @property
    def A_indices_deletes_after_insert_to_codon(self):
        indices = []
        for codon_id in range(self.config.nCodons-2):
            for insert_id in range(2, self.config.nCodons):
                indices += [[self.str_to_state_id(f"c_{codon_id},2"), self.str_to_state_id(f"i_{insert_id}, 0")]]
        return indices

    # currently not used
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
        for i in range(self.config.nCodons - 1):
            indices += [[self.str_to_state_id(f"c_{i},2"), self.str_to_state_id(f"i_{i+1},0")]]
        return indices
    @property
    def A_indices_end_inserts(self):
        indices = []
        for i in range(1, self.config.nCodons):
            indices += [[self.str_to_state_id(f"i_{i},2"), self.str_to_state_id(f"c_{i},0")]]
         # including last insert -> GT
        if self.config.inserts_at_intron_borders:
                indices += [[self.str_to_state_id(f"i_{self.config.nCodons},2"), self.str_to_state_id("G")]]
        return indices
    @property
    def A_indices_continue_inserts(self):
        indices = []
        for i in range(1, self.config.nCodons):
            indices += [[self.str_to_state_id(f"i_{i},2"), self.str_to_state_id(f"i_{i},0")]]
        return indices
################################################################################
    def A_indices(self):
        return self.A_indices_for_weights + self.A_indices_for_constants
################################################################################
################################################################################
################################################################################
    def nucleotide_ambiguity_code_to_array(self, emission):
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
        # pad emission with prefix of Ns to bring it to length order + 1
        # or only take the last bases if emission is too long
        return ["N"] * (self.config.order - len(ho_emission) + 1) + list(ho_emission)[- self.config.order - 1:]
################################################################################
    def has_I_emission_after_base(self, ho_emission):
        # I is the place holder symbol used at the beginning of the sequence
        # emissions like AIA or TII are forbidden, IIC, ITA are allowed
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
            # bc b might me longer than 3 bases
            # only the last three bases are compared
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
        # this method relies on the names of the states
        # as they are definded in get_state_id_description_dicts()
        if s[-1] == "2" and s != "stop2" and s != "ter2" and s[0] in "ic":
            return True
        return False
################################################################################
    def get_emissions_that_fit_ambiguity_mask(self, ho_mask, x_bases_must_preceed, state):
        # getting the allowd base emissions in each slot
        # ie "NNA" and x_bases_must_preceed = 2 -> [[0,1,2,3], [0,1,2,3], [0]]
        # x_bases_must_preceed says how many bases must preceed the first base in the emission
        # if it is less then order +1, the placeholder symbol I is allowed in the beginning
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
    def prior_and_initial_parameter_for_state_and_emission(self, state:str, emission:str):
        emission = emission.lower()

        prior = -1.0
        initial_parameter = -1.0

        # if they are not altered, ie remain -1
        # then they will get no prior and initial parameter
        # is set to be unifrom over all emissions in the state

        if state in ["left_intron", "right_intron"] and emission[0] != "i":
            norm = sum([self.prior.get_intron_prob(emission[:-1] + base) for base in "ACGT"])
            assert abs(norm - 1 ) < 1e-3, f"norm is {norm} but should be one"
            prior = self.prior.get_intron_prob(emission) / norm
            initial_parameter = prior

        # codons
        if state[0] == "c": # eg. c_0,2, c_12,1
            codon_id = int(re.search(r"c_(\d+),\d+", state).group(1))
            if codon_id != 0:
                norm = sum([self.prior.get_exon_prob(emission[:-1] + base, window = int(state[-1])) for base in "ACGT"])
                assert abs(norm - 1 ) < 1e-3, f"norm is {norm} but should be one"
                prior = self.prior.get_exon_prob(emission, window = int(state[-1])) / norm
                initial_parameter = prior

        # inserts
        if state[0] == "i":
            codon_id = int(re.search(r"i_(\d+),\d+", state).group(1))
            if codon_id not in [0, self.config.nCodons]:
                norm = sum([self.prior.get_exon_prob(emission[:-1] + base, window = int(state[-1])) for base in "ACGT"])
                assert abs(norm - 1 ) < 1e-3, f"norm is {norm} but should be one"
                prior = self.prior.get_exon_prob(emission, window = int(state[-1])) / norm
                initial_parameter = prior

        # acceptor splice site
        if state.startswith("ak_"):
            ak_id = int(re.search(r"ak_(\d+)", state).group(1))
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
                assert norm != 0, "norm in ak is zero print id:0923htui4"
                prior = unscaled_prob/norm
                initial_parameter = prior
            else:
                norm = sum([self.prior.get_intron_prob(emission[:-1] + base) for base in "ACGT"])
                assert abs(norm - 1 ) < 1e-4, f"norm is {norm} but should be one"
                prior = self.prior.get_intron_prob(emission) / norm
                initial_parameter = prior

        # donor splice site
        if state.startswith("do_"):
            do_id = int(re.search(r"do_(\d+)", state).group(1))
            assert self.config.order == 2, "for this code snippet order is assumend to be 2"
            if do_id < self.config.dss_end:
                if do_id == 0:
                    after_pattern = f"{emission[2:]}.{{{self.config.dss_end-1}}}"
                elif do_id == 1:
                    after_pattern = f"{emission[1:]}.{{{self.config.dss_end-2}}}"
                elif do_id == 2:
                    after_pattern = f"{emission}.{{{self.config.dss_end-3}}}"
                else:
                    after_pattern = f".{{{do_id-2}}}{emission}.{{{self.config.dss_end - do_id - 1}}}"
                pattern = f".{{{self.config.dss_start}}}GT{after_pattern}"
                def replace_last_base_with_period(string):
                    modified_string = re.sub(r"([A-Za-z])(?=[^A-Za-z]*$)", r".", string)
                    return modified_string
                norm_pattern = f".{{{self.config.dss_start}}}GT{replace_last_base_with_period(after_pattern)}"
                unscaled_prob = self.prior.get_splice_site_matching_pattern_probs(description = "DSS", pattern = pattern)
                norm = self.prior.get_splice_site_matching_pattern_probs(description = "DSS", pattern = norm_pattern)
                assert norm != 0, "norm in ak is zero 2ß30tu9z8wg"
                prior = unscaled_prob/norm
                initial_parameter = prior
            else:
                norm = sum([self.prior.get_intron_prob(emission[:-1] + base) for base in "ACGT"])
                assert abs(norm - 1 ) < 1e-3, f"norm is {norm} but should be one"
                prior = self.prior.get_intron_prob(emission) / norm
                initial_parameter = prior

        return prior, initial_parameter
################################################################################
################################################################################
################################################################################
    def get_A_prior_matrix(self):
        dense_shape = [self.number_of_states, self.number_of_states]
        # since the concentraions paramteres are use for initial weights
        # as well, the initial weights are used here to create
        # a matrixs that holds the concentration parameters for the
        # transition matrix
        values = tf.concat([self.A_initial_weights_for_trainable_parameters, \
                            self.A_initial_weights_for_constants], axis = 0)

        A_init = tf.scatter_nd(self.A_indices, \
                                    values, \
                                    dense_shape)
        # A mask for all parameters
        A_mask = tf.scatter_nd(self.A_indices, \
                               [1] * len(self.A_indices), \
                               dense_shape)
        # softmax
        softmax_layer = tf.keras.layers.Softmax()

        A_init_stochastic = softmax_layer(A_init, tf.cast(A_mask, tf.int32))

        # A mask for priors concentration parameters
        self.A_prior_indices = tf.cast(self.A_prior_indices, tf.int32)

        self.A_prior_mask = tf.scatter_nd(self.A_prior_indices, \
                                   [1.0] * len(self.A_prior_indices), \
                                   dense_shape)
        A_init_stochastic = tf.cast(A_init_stochastic, self.config.dtype)
        self.A_prior_mask = tf.cast(self.A_prior_mask, self.config.dtype)
        self.A_prior_matrix = tf.cast(A_init_stochastic * self.A_prior_mask, \
                                      dtype = self.config.dtype) * self.config.priorA

        # write results to file for inspection
        if self.config.nCodons < 20:
            dir_path = f"{self.config.current_run_dir}/prior_calculation"
            self.A_as_dense_to_file(f"{dir_path}/A_init.csv", "dummy weight parameter", A = A_init, with_description = self.config.nCodons < 20)
            self.A_as_dense_to_file(f"{dir_path}/A_init_stochastic.csv", "dummy weight parameter", A = A_init_stochastic, with_description = self.config.nCodons < 20)
            self.A_as_dense_to_file(f"{dir_path}/A_prior_mask.csv", "dummy weight parameter", A = self.A_prior_mask, with_description = self.config.nCodons < 20)
            self.A_as_dense_to_file(f"{dir_path}/A_prior_matrix.csv", "dummy weight parameter", A = self.A_prior_matrix, with_description = self.config.nCodons < 20)


################################################################################
    def assert_a_is_compatible_with_direchlet_prior(self):

        # if there is a prior for a parameter in a row,
        # then every parameter has to have a prior in that row

        A_mask = tf.scatter_nd(self.A_indices, \
                            [1.0] * len(self.A_indices), \
                            (self.number_of_states, self.number_of_states))

        def get_mask_rows_that_have_prior(tensor):
            row_sums = tf.math.reduce_sum(tensor, axis = -1)
            mask = tf.math.greater(row_sums, 0)
            return mask

        rows_that_matter = get_mask_rows_that_have_prior(self.A_prior_matrix)

        diff_mask = A_mask - self.A_prior_mask
        diff_rows_that_matter = tf.boolean_mask(diff_mask, rows_that_matter)

        tf.debugging.Assert(tf.math.reduce_all(diff_rows_that_matter == 0), [diff_rows_that_matter], name = "some_A_paras_havent_got_prior")

################################################################################
    def get_A_log_prior(self, A_kernel):
        if self.config.priorA == 0:
            return 0
        A_dense = tf.sparse.to_dense(self.A(A_kernel))
        self.assert_a_is_compatible_with_direchlet_prior()
        return non_class_A_log_prior(A_dense, self.A_prior_matrix, self.A_prior_indices)

################################################################################
    def get_B_prior_matrix(self):
        dense_shape = [self.number_of_emissions, self.number_of_states]
        self.B_prior_matrix = tf.scatter_nd(self.B_prior_indices, self.B_priors, shape = dense_shape)
        self.B_prior_matrix = tf.cast(self.B_prior_matrix, dtype = self.config.dtype) * self.config.priorB

        if self.config.nCodons < 20:
            dir_path = f"{self.config.current_run_dir}/prior_calculation"
            self.B_as_dense_to_file(f"{dir_path}/B_prior_matrix.csv", "dummy weight parameter", B = self.B_prior_matrix, with_description = self.config.nCodons < 20)
################################################################################
    def assert_b_is_compatible_with_direchlet_prior(self):
        B_mask = tf.scatter_nd(self.B_indices, \
                            [1.0] * len(self.B_indices), \
                            (self.number_of_emissions, self.number_of_states))

        B_prior_mask = tf.scatter_nd(self.B_prior_indices, \
                            [1.0] * len(self.B_priors), \
                            (self.number_of_emissions, self.number_of_states))

        other_shape = (self.number_of_states, -1, self.config.alphabet_size)
        reshaped_B_mask = tf.reshape(tf.transpose(B_mask), other_shape)
        reshaped_prior_mask = tf.reshape(tf.transpose(B_prior_mask), other_shape)
        reshaped_B_prior_matrix = tf.reshape(tf.transpose(self.B_prior_matrix), other_shape)


        def get_mask_rows_that_have_prior(tensor):
            row_sums = tf.math.reduce_sum(tensor, axis = -1)
            mask = tf.math.greater(row_sums, 0)
            return mask

        rows_that_matter = get_mask_rows_that_have_prior(reshaped_B_prior_matrix)

        diff_mask = reshaped_B_mask - reshaped_prior_mask
        diff_rows_that_matter = tf.boolean_mask(diff_mask, rows_that_matter)
        tf.debugging.Assert(tf.math.reduce_all(diff_rows_that_matter == 0), [diff_rows_that_matter], name = "some_B_paras_havent_got_prior", summarize = -1)
################################################################################
    def get_B_log_prior(self, B_kernel):
        if self.config.priorB == 0:
            return 0
        self.assert_b_is_compatible_with_direchlet_prior()
        return non_class_B_log_prior(self.B(B_kernel), self.B_prior_matrix, self.B_prior_indices, self.config.alphabet_size)
################################################################################
################################################################################
################################################################################
    def get_indices_for_emission_and_state(self, state, mask, x_bases_must_preceed, trainable = None):
        state_id = self.str_to_state_id(state)
        # if self.order_transformed_input and emissions[-1] == "X":
        indices = self.B_indices_for_trainable_parameters if trainable else self.B_indices_for_constant_parameters
        initial_parameters = self.B_initial_trainable_para_setting if trainable else self.B_initial_constant_para_setting
        if mask[-1] == "X":
            indices += [[self.emission_tuple_to_id("X"), state_id]]
            initial_parameters.append(1.0)
            return
        # ho_emission stands for higher order emission, so not A,C,G,T but [1,2,0] (^=CGA)
        for ho_emission in self.get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state_id):
            p,i =  self.prior_and_initial_parameter_for_state_and_emission(state, self.emission_tuple_to_str(ho_emission))

            # if initial parameter is zero, set it to epsilon,
            # since otherwise the prior loss would be infinte
            if i == 0:
                i = 1e-3

            initial_parameters.append(i)
            index = [self.emission_tuple_to_id(ho_emission), state_id]
            if p != -1:

                # since the concentration parameters must be greater than zero
                # to be defined, they are set to epsilon
                if p == 0:
                    p = 1e-3
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
        self.B_initial_trainable_para_setting = []
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

        # Here, states and their emissions are set
        # if a want to enforce a mask or
        # want to make them not trainable,
        # or change x bases must preceed
        append_emission("left_intron","N", 0)
        append_emission("A", "A", self.config.order, trainable = False)
        append_emission("AG", "AG", self.config.order, trainable = False)

        append_emission("c_0,0", "AGN")
        # for c_0,1 and c_0,2 i cant force AG to be emitted before, since the exon can be entered in phase 1 or 2

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
    def I(self):
        initial_matrix = tf.scatter_nd([[0,0]], [1.0], [1, self.number_of_states])
        return initial_matrix

################################################################################
    def A(self, weights):
        if self.config.trace_verbose:
            print("model.A")
        values = tf.concat([weights, tf.cast(self.A_initial_weights_for_constants,self.config.dtype)], axis = 0)
        dense_shape = [self.number_of_states, self.number_of_states]

        if self.config.A_is_sparse:
            transition_matrix = tf.sparse.SparseTensor(indices = self.A_indices, \
                                                       values = values, \
                                                       dense_shape = dense_shape)

            transition_matrix = tf.sparse.reorder(transition_matrix)
            transition_matrix = tf.sparse.softmax(transition_matrix, name = "A_sparse")

        if self.config.A_is_dense:
            transition_matrix = tf.scatter_nd(self.A_indices, values, dense_shape)
            softmax_layer = tf.keras.layers.Softmax()
            mask = tf.scatter_nd(self.A_indices, [1.0] * len(self.A_indices), dense_shape)
            transition_matrix = softmax_layer(transition_matrix, mask)

        return transition_matrix
################################################################################
    def make_B_initial_weights(self):
        initial_parameters_for_weights = np.zeros(len(self.B_indices_for_trainable_parameters))
        initial_parameters_for_consts = np.zeros(len(self.B_indices_for_constant_parameters))

        concated_init_para_settings = tf.concat([self.B_initial_trainable_para_setting, \
                                                 self.B_initial_constant_para_setting], axis = 0)

        dense_shape = [self.number_of_emissions, self.number_of_states]
        emission_matrix = tf.scatter_nd(self.B_indices, concated_init_para_settings, dense_shape)

        if self.config.nCodons < 20:
            import os
            dir_name = f"{self.config.current_run_dir}/prior_calculation"
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            self.B_as_dense_to_file(f"{dir_name}/B_init_parameters_before_conversion.csv", self.B_initial_trainable_para_setting, with_description = True, B = emission_matrix)

        # some paramters are currently set to -1
        # if the dont have a hand crafter initial parameter
        # since we have categorical distributions, we need to convert them
        # to a probability distribution.
        # this is done by calculating the sum of all parameters that
        # hand crafted got initial parameters.
        # then is calculated how much is missing for the sum to be one
        # this rest is then evenly distributed over all parameters that
        # have no hand crafted initial parameter (were set to -1)
        for state in range(tf.shape(emission_matrix)[1]):
            for emission_id_4 in range(0,tf.shape(emission_matrix)[0], self.config.alphabet_size):
                sum_with_out_zeros_and_negatives_ones = 0.0
                number_of_values_that_can_be_flattened = 0.0
                found_negative_ones = 0.0
                for emission_id in range(emission_id_4, emission_id_4 + self.config.alphabet_size):
                    if emission_matrix[emission_id, state] != 0:
                        number_of_values_that_can_be_flattened += 1
                    if emission_matrix[emission_id, state] != -1:
                        sum_with_out_zeros_and_negatives_ones += tf.cast(emission_matrix[emission_id, state], tf.float32)
                    else:
                        found_negative_ones += 1

                for emission_id in range(emission_id_4, emission_id_4 + self.config.alphabet_size):
                    if emission_matrix[emission_id, state] == -1:
                        weight = tf.math.log((1-sum_with_out_zeros_and_negatives_ones)/found_negative_ones)
                    else:
                        if self.config.flatten_B_init and number_of_values_that_can_be_flattened != 0:
                            flattened_value = emission_matrix[emission_id, state] * (1-self.config.flatten_B_init) + 1.0/number_of_values_that_can_be_flattened * self.config.flatten_B_init
                            weight = tf.math.log(tf.cast(flattened_value, tf.float32))
                        else:
                            weight = tf.math.log(tf.cast(emission_matrix[emission_id, state], tf.float32))
                    indx = (emission_id, state)
                    if list(indx) in self.B_indices_for_trainable_parameters:
                        initial_parameters_for_weights[self.B_weight_tuple_id(indx)] = weight
                    elif list(indx) in self.B_indices_for_constant_parameters:
                        initial_parameters_for_consts[self.B_const_tuple_id(indx)] = weight
        initial_parameters_for_weights = np.array(initial_parameters_for_weights, np.float32)
        initial_parameters_for_consts = np.array(initial_parameters_for_consts, np.float32)

        return initial_parameters_for_weights, initial_parameters_for_consts

################################################################################
    def B(self, weights):
        if self.config.trace_verbose:
            print("model.B")
        try:
            consts = tf.cast(self.B_initial_weights_for_constant_parameters, self.config.dtype)
        except:
            # this is not actually used, just for debugging
            # and check if make_B_initial_weights() works correctly
            consts = tf.cast(self.B_initial_constant_para_setting, self.config.dtype)
        values = tf.concat([weights, consts], axis = 0)
        dense_shape = [self.number_of_emissions, \
                       self.number_of_states]

        if self.config.B_is_sparse:
            emission_matrix = tf.sparse.SparseTensor(indices = self.B_indices, \
                                                     values = values, \
                                                     dense_shape = dense_shape)

            emission_matrix = tf.sparse.reorder(emission_matrix)
            emission_matrix = tf.sparse.transpose(emission_matrix)
            # for sparse only sparse.softmax works, which has no arg "axis"
            # so i have to reshape it
            emission_matrix = tf.sparse.reshape(emission_matrix, shape = (self.number_of_states, -1, self.config.alphabet_size))
            emission_matrix = tf.sparse.softmax(emission_matrix)
            emission_matrix = tf.sparse.reshape(emission_matrix, shape = (self.number_of_states, self.number_of_emissions), name = "B_sparse")
            emission_matrix = tf.sparse.transpose(emission_matrix)

        if self.config.B_is_dense:
            shape_to_apply_softmax_to = (-1, self.config.alphabet_size, self.number_of_states)
            emission_matrix = tf.scatter_nd(self.B_indices, values, dense_shape)
            mask = tf.scatter_nd(self.B_indices, [1.0] * len(self.B_indices), dense_shape)
            # reshape
            emission_matrix = tf.reshape(emission_matrix, shape = shape_to_apply_softmax_to)
            mask            = tf.reshape(mask,            shape = shape_to_apply_softmax_to)
            # softmax layer to apply along a specific axis and use a mask
            softmax_layer = tf.keras.layers.Softmax(axis = 1)
            emission_matrix = softmax_layer(emission_matrix, mask)
            # this leaves [0.25, 0.25, 0.25, 0.25] in columns where the mask has only zeros
            emission_matrix = tf.reshape(emission_matrix, shape = dense_shape, name = "B_dense")
            # removing the [0.25, 0.25, 0.25, 0.25] artefact
            emission_matrix = tf.tensor_scatter_nd_min(emission_matrix, \
                                                       self.B_indices_complement, \
                                                       [0.0] * len(self.B_indices_complement))
        return emission_matrix
################################################################################
################################################################################
################################################################################

    def I_as_dense_to_json_file(self, path):
        with open(path, "w") as out_file:
            json.dump(self.I().numpy().tolist(), out_file)

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
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, "w") as out_file:
            out_file.write(self.B_as_dense_to_str(weights, with_description, B = B))

    def B_as_dense_to_json_file(self, path, weights):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))

        with open(path, "w") as out_file:
            B = self.B(weights) if self.B_is_dense else tf.sparse.to_dense(self.B(weights))
            json.dump(B.numpy().tolist(), out_file)

