#!/usr/bin/env python3
from Model import Model
import re
from itertools import product
import tensorflow as tf

class My_Model(Model):

    # this overwrites the init from Model. alternatively i can omit it
    def __init__(self, config):
        Model.__init__(self, config)

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


        self.I_indices = self.I_indices()

        self.A_indices_for_weights = self.A_indices_for_weights()
        self.A_indices_for_constants = self.A_indices_for_constants()
        self.A_indices = self.A_indices_for_weights + self.A_indices_for_constants

        self.B_indices_for_weights = self.B_indices_for_weights()
        self.B_indices_for_constants = self.B_indices_for_constants()
        self.B_indices = self.B_indices_for_weights + self.B_indices_for_constants

        shape = (self.number_of_emissions, self.number_of_states)
        B_indices_complement = tf.where(tf.ones(shape, dtype = tf.float32) - tf.scatter_nd(self.B_indices, [1.0] * len(self.B_indices), shape = shape))
        self.B_indices_complement = tf.cast(B_indices_complement, dtype = tf.int32)

    # =================> states <===============================================
    def get_number_of_states(self):
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

    def get_state_id_description_list(self):
        # if this is changed, also change state_is_third_pos_in_frame()
        states = re.split(" ", "ig5' stA stT stG")
        states += ["c_" + str(i) + "," + str(j) for i in range(self.config.nCodons) for j in range(3)]
        states += re.split(" ", "stop1 stop2 stop3 ig3'")
        states += ["i_" + str(i) + "," + str(j) for i in range(self.config.nCodons+1) for j in range(3)]
        states += ["ter1", "ter2"]
        return states

    def state_id_to_str(self, id):
        return self.state_id_description_list[id]

    def str_to_state_id(self, s):
        try:
            return state_id_description_list.index(s)
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
        return len(self.I_indices)

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
        return self.A_indices_for_weights + self.A_indices_for_constants
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
        # always has to to be dense, since R must the same on the main and off branch, and off branch R is dense and main R = I
        initial_matrix = tf.sparse.SparseTensor(indices = self.I_indices, values = weights, dense_shape = [self.number_of_states,1])
        initial_matrix = tf.sparse.reorder(initial_matrix)
        initial_matrix = tf.sparse.reshape(initial_matrix, (1,self.number_of_states), name = "I_sparse")
        initial_matrix = tf.sparse.softmax(initial_matrix, name = "I_sparse")
        return tf.sparse.to_dense(initial_matrix, name = "I_dense")
################################################################################
    def A(self, weights):
        if self.config.use_weights_for_consts:
            values = weights
        else:
            consts = tf.cast([1.0] * len(self.A_indices_for_constants), dtype = self.config.dtype)
            values = tf.concat([weights, consts], axis = 0)
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
            emission_matrix = tf.sparse.SparseTensor(indices = self.B_indices, \
                                                     values = values, \
                                                     dense_shape = dense_shape)

            emission_matrix = tf.sparse.reorder(emission_matrix)
            emission_matrix = tf.sparse.transpose(emission_matrix)
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
            # softmax
            softmax_layer = tf.keras.layers.Softmax(axis = 1)
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



if __name__ == '__main__':
    from Config import Config
    config = Config("main_programm")
    f = My_Model(config)
    import numpy as np
    print(f.A(np.ones(13)))
