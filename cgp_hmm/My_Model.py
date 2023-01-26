#!/usr/bin/env python3
from Model import Model
import re

class My_Model(Model):

    # this overwrites the init from Model. alternatively i can omit it
    def __init__(self, config):
        Model.__init__(self, config)
        self.use_sparse = True
        self.use_dense = not self.use_sparse

        # TODO: calculate indices here and save them as attribute

    def number_of_states(self):
        number_of_states = 1
        # start
        number_of_states += 3
        # codons
        number_of_states += 3 * self.config.nCodons
        # codon inserts
        number_of_states += 3 * (self.config.nCodons + 1)
        # stop
        number_of_states += 3
        # ig 3'
        number_of_states += 1
        # terminal
        number_of_states += 1
        return number_of_states

    @classmethod
    def get_state_id_description_list(cls, nCodons):
        # if this is changed, also change state_is_third_pos_in_frame()
        states = re.split(" ", "ig5' stA stT stG")
        states += ["c_" + str(i) + "," + str(j) for i in range(nCodons) for j in range(3)]
        states += re.split(" ", "stop1 stop2 stop3 ig3'")
        states += ["i_" + str(i) + "," + str(j) for i in range(nCodons+1) for j in range(3)]
        states += ["ter1", "ter2"]
        return states

    def state_id_to_description(id, nCodons, state_id_description_list = None):
        if not state_id_description_list:
            state_id_description_list = get_state_id_description_list(nCodons)
        # print("nCodons =", nCodons)
        # print("id =", id)
        # print("state_id_to_descriptcation =", state_id_description_list)
        return state_id_description_list[id]

    def description_to_state_id(des, nCodons, state_id_description_list = None):
        if not state_id_description_list:
            state_id_description_list = get_state_id_description_list(nCodons)
        try:
            return state_id_description_list.index(des)
        except:
            return -1

    def emissions_state_size(alphabet_size, order):# with out terminal symbol
        if order == 0:
            return alphabet_size1

        #      [IACGT] x [ACGT]^order                         + IIA IIC IIG IIT (if order == 2)
        return (alphabet_size + 1) * alphabet_size ** (order) + sum([alphabet_size ** i for i in range(1, order)])

    # added 4, of which the last one corresponds to the terminal symbol
    # the other ones are dummy, they make the columns in B divisable by 4
    def n_emission_columns_in_B(alphabet_size, order):
        return emissions_state_size(alphabet_size, order) + 4

    def number_of_emissions(self):
        from Utility import n_emission_columns_in_B #TODO this should rather be emission state size
        return n_emission_columns_in_B(self.config.alphabet_size, self.config.order)

    @classmethod
    def get_dicts_for_emission_tuple_and_id_conversion(cls, config = None, alphabet_size = None, order = None):
        if config == None:
            assert alphabet_size != None, "get_dicts_for_emission_tuple_and_id_conversion must be provided with config or (alphabet_size and order)"
            assert order != None, "get_dicts_for_emission_tuple_and_id_conversion must be provided with config or (alphabet_size and order)"
        if alphabet_size == None:
            alphabet_size = config.alphabet_size
        if order == None:
            order = config.order

        emi_to_id = {}
        id_to_emi = {}
        if order == 0:
            emi_to_id = dict([(tuple([base]), id) for id, base in enumerate(list(range(alphabet_size)) + ["X"])])
            id_to_emi = dict([(id, tuple([base])) for id, base in enumerate(list(range(alphabet_size)) + ["X"])])
        else:
            import Utility
            from itertools import product
            id = 0
            for emission_tuple in product(list(range(alphabet_size + 1)), repeat = order + 1):
                if not Utility.has_I_emission_after_base(emission_tuple, alphabet_size = alphabet_size, order = order):
                    id_to_emi[id] = emission_tuple
                    emi_to_id[emission_tuple] = id
                    id += 1
            id_to_emi[id] = tuple("X")
            emi_to_id[tuple("X")] = id

        # print("emi_to_id =", emi_to_id)
        # print("id_to_emi =", id_to_emi)
        if config != None:
            config.id_to_emi = id_to_emi
            config.emi_to_id = emi_to_id
        else:
            return emi_to_id, id_to_emi

    # emission is either a tuple like [2,1,3] or "X"
    def higher_order_emission_to_id(self, emission_tuple):
        return get_dicts_for_emission_tuple_and_id_conversion(alphabet_size = self.config.alphabet_size, order = self.config.order)[0][tuple(emission_tuple)]


    def id_to_higher_order_emission(id, alphabet_size, order, as_string = False):
        return get_dicts_for_emission_tuple_and_id_conversion(alphabet_size = alphabet_size, order = order)[1][id]


    def emi_tuple_to_str(emi_tuple):
        if emi_tuple[0] == "X":
            return "X"
        return "".join(list(map(lambda x: "ACGTI"[x], emi_tuple)))
################################################################################
################################################################################
################################################################################
    def I_kernel_size(self):
        return len(self.I_indices())

    def A_kernel_size(self):
        if self.config.use_weights_for_consts:
            return len(self.A_indices_for_weights()) + len(self.A_indices_for_constants())
        return len(self.A_indices_for_weights())

    def B_kernel_size(self):
        if self.config.use_weights_for_consts:
            return len(self.B_indices_for_weights()) + len(self.B_indices_for_constants())
        return len(self.B_indices_for_weights())
################################################################################
################################################################################
################################################################################
    # TODO: maybe use small letters instead of capital ones
    def I_indices(self):
        # start and codons
        indices = [[i,0] for i in range(3 + self.config.nCodons*3)]
        # inserts
        indices += [[i,0] for i in range(8 + self.config.nCodons*3, 8 + self.config.nCodons*3 + (self.config.nCodons + 1)*3)]

        return indices
################################################################################
################################################################################
################################################################################
    def A_indices_for_constants(self):
        # from start a
        indices = [[1,2]]
        # from start t
        indices += [[2,3]]

        # first to second codon position
        indices += [[4 + i*3, 5 + i*3] for i in range(self.config.nCodons)]
        # second to third codon position
        indices += [[5 + i*3, 6 + i*3] for i in range(self.config.nCodons)]

        # inserts
        offset = 8 + 3*self.config.nCodons
        # begin inserts

        # first to second position in insert
        indices += [[offset + i*3, offset + 1 + i*3] for i in range(self.config.nCodons + 1)]
        # second to third position in insert
        indices += [[offset + 1 + i*3, offset + 2 + i*3] for i in range(self.config.nCodons + 1)]
        # ending an insert

        # stop T
        indices += [[4 +  self.config.nCodons*3, 5 + self.config.nCodons*3]]

        # second to third position in stop
        indices += [[5 +  self.config.nCodons*3, 6 + self.config.nCodons*3]]

        # stop -> ig 3'
        indices += [[6 +  self.config.nCodons*3, 7 + self.config.nCodons*3]]

        index_of_terminal_1 = 8 + self.config.nCodons*3 + (self.config.nCodons + 1) *3
        indices += [[index_of_terminal_1, index_of_terminal_1]]

        return indices

    def A_indices_for_weights(self): # no shared parameters
        # from ig 5'
        indices = [[0,0], [0,1]]

        # enter codon
        indices += [[3 + i*3, 4 + i*3] for i in range(self.config.nCodons)]

        if not self.config.no_inserts:
            offset = 8 + 3*self.config.nCodons
            # begin inserts
            indices += [[3 + i*3, offset + i*3] for i in range(self.config.nCodons + 1)]
            # ending an insert
            indices += [[offset + 2 + i*3, 4 + i*3] for i in range(self.config.nCodons + 1)]
            # continuing an insert
            indices += [[offset + 2 + i*3, offset + i*3] for i in range(self.config.nCodons +1)]

        # exit last codon
        indices += [[3 + self.config.nCodons*3, 4 + self.config.nCodons*3]]

        # deletes
        if not self.config.no_deletes:
            i_delete = [3 + i*3 for i in range(self.config.nCodons) for j in range(self.config.nCodons-i)]
            j_delete = [4 + j*3 for i in range(1,self.config.nCodons+1) for j in range(i,self.config.nCodons+1)]
            indices += [[i,j] for i,j in zip(i_delete, j_delete)]

        # ig -> ig, terminal_1
        index_of_terminal_1 = 8 + self.config.nCodons*3 + (self.config.nCodons + 1) *3
        indices += [[7 + self.config.nCodons*3, 7 + self.config.nCodons*3], [7 + self.config.nCodons*3, index_of_terminal_1]]

        return indices

    def A_indices(self):
        return self.A_indices_for_weights() + self.A_indices_for_constants()
################################################################################
################################################################################
################################################################################
    @classmethod
    def nucleotide_ambiguity_code_to_array(emission):
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
    @classmethod
    def strip_or_pad_emission_with_n(cls, config, ho_emission):
        return ["N"] * (config.order - len(ho_emission) + 1) + list(ho_emission)[- config.order - 1:]
    ################################################################################
    @classmethod
    def has_I_emission_after_base(ho_emission, config = None, alphabet_size = None, order = None): # or is only I
        if config == None:
            assert alphabet_size != None, "has_I_emission_after_base must be provided with self.config or (alphabet_size and order)"
            assert order != None, "has_I_emission_after_base must be provided with self.config or (alphabet_size and order)"
        if alphabet_size == None:
            alphabet_size = config.alphabet_size
        if order == None:
            order = config.order

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
    @classmethod
    def emission_is_stop_codon(ho_emission):
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
    @classmethod
    def state_is_third_pos_in_frame(config, state):
        des = state_id_to_description(state, config.nCodons, config.state_id_description_list)
        if des [-1] == "2" and des != "stop2" and des != "ter2":
            return True
        return False
    ################################################################################
    @classmethod
    def get_emissions_that_fit_ambiguity_mask(cls, config, ho_mask, x_bases_must_preceed, state):

        # getting the allowd base emissions in each slot
        # ie "NNA" and x_bases_must_preceed = 2 -> [][0,1,2,3], [0,1,2,3], [0]]
        allowed_bases = [0] * (config.order + 1)
        for i, emission in enumerate(strip_or_pad_emission_with_n(config, ho_mask)):
            allowed_bases[i] = self.nucleotide_ambiguity_code_to_array(emission)
            if i < config.order - x_bases_must_preceed:
                allowed_bases[i] += [4] # initial emission symbol

        allowed_ho_emissions = []
        state_is_third_pos_in_frame_bool = self.state_is_third_pos_in_frame(config, state)
        for ho_emission in product(*allowed_bases):
            if not self.has_I_emission_after_base(ho_emission, config = config) \
            and not (state_is_third_pos_in_frame_bool and self.semission_is_stop_codon(ho_emission)):
                allowed_ho_emissions += [ho_emission]

        return allowed_ho_emissions

    ################################################################################
    def get_indices_for_emission_and_state(self, indices, state, mask, x_bases_must_preceed):
        # if self.order_transformed_input and emissions[-1] == "X":
        if mask[-1] == "X":
            indices += [[state, higher_order_emission_to_id("X", self.config.alphabet_size, self.config.order)]]
            return

        for ho_emission in self.get_emissions_that_fit_ambiguity_mask(self.config, mask, x_bases_must_preceed, state):
            indices += [[state, higher_order_emission_to_id(ho_emission, self.config.alphabet_size, self.config.order)]]
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

        self.get_indices_for_emission_and_state(self.config, indices,3,"ATG",2)
        self.get_indices_for_emission_and_state(self.config, indices,6 + nCodons*3,"TAA", self.config.order)
        self.get_indices_for_emission_and_state(self.config, indices,6 + nCodons*3,"TAG", self.config.order)
        if self.config.order > 0:
            # bc then only the third pos is codon is of importance, and then "A" would be added twice
            self.get_indices_for_emission_and_state(self.config, indices,6 + nCodons*3,"TGA", self.config.order)

        return indices
    def B_indices(self):
        return self.B_indices_for_weights() + self.B_indices_for_constants()
################################################################################
################################################################################
################################################################################
    def I(weights):
        # always has to to be dense, since R must the same on the main and off branch, and off branch R is dense and main R = I
        initial_matrix = tf.sparse.SparseTensor(indices = self.I_indices(), values = weights, dense_shape = [self.number_of_states(),1])
        initial_matrix = tf.sparse.reorder(initial_matrix)
        initial_matrix = tf.sparse.reshape(initial_matrix, (1,self.number_of_states()), name = "I_sparse")
        initial_matrix = tf.sparse.softmax(initial_matrix, name = "I_sparse")
        return tf.sparse.to_dense(initial_matrix, name = "I_dense")
################################################################################
    def A(weights):
        if self.self.config.use_weights_for_consts:
            values =  self.transition_kernel
            indices = self.indices_for_A
        else:
            consts = tf.cast([1.0] * len(self.indices_for_constants_A), dtype = self.self.config.dtype)
            values = tf.concat([self.transition_kernel, consts], axis = 0)
            indices = self.indices_for_weights_A + self.indices_for_constants_A

        transition_matrix = tf.sparse.SparseTensor(indices = indices, \
                                                   values = values, \
                                                   dense_shape = [self.number_of_states()] * 2)

        transition_matrix = tf.sparse.reorder(transition_matrix)
        transition_matrix = tf.sparse.softmax(transition_matrix, name = "A_sparse")

        if self.use_sparse:
            return transition_matrix
        return tf.sparse.to_dense(transition_matrix, name = "A_dense")
################################################################################
    def B(weights):
        dense_shape = [self.number_of_states(), \
                       self.number_of_emissions()]

        emission_matrix = tf.sparse.SparseTensor(indices = self.B_indices(), \
                                                 values = weights, \
                                                 dense_shape = dense_shape)

        emission_matrix = tf.sparse.reorder(emission_matrix)
        emission_matrix = tf.sparse.reshape(emission_matrix, shape = (self.number_of_states(), -1, self.self.config.alphabet_size))
        emission_matrix = tf.sparse.softmax(emission_matrix)
        emission_matrix = tf.sparse.reshape(emission_matrix, shape = (self.number_of_states(), -1))

        emission_matrix = tf.sparse.transpose(emission_matrix, name = "B_sparse")

        if self.use_sparse:
            return emission_matrix
        return tf.sparse.to_dense(emission_matrix, name = "B_dense")
################################################################################
################################################################################
################################################################################

if __name__ == '__main__':
    from Config import Config
    config = Config("main_programm")
    f = Full_Model(config)
    print(f.A())
