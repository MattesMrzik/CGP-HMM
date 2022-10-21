#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from itertools import product
from Utility import higher_order_emission_to_id

def prRed(skk): print("Cell\033[93m {}\033[00m" .format(skk))
# def prRed(s): pass

class CgpHmmCell(tf.keras.layers.Layer):
# class CgpHmmCell(tf.keras.layers.Layer):
    def __init__(self, nCodons, order_transformed_input):
        # super(CgpHmmCell, self).__init__()
        super(CgpHmmCell, self).__init__()

        self.nCodons = nCodons

        self.alphabet_size = 4 # without terminal symbol and without "papped left flank" symbol

        self.order = 2 # order = 0 -> emission prob depends only on current emission

        self.order_transformed_input = order_transformed_input

        self.checksquare = False

        if self.order_transformed_input:
            if self.checksquare:
                self.state_size = [tf.TensorShape([1,18]), 1,         1, 7] #self.calc_number_of_states()
            else:
                self.state_size = [self.calc_number_of_states(), 1,      1]
        else:
            #                  alpha                         old loglik, count, old_input
            self.state_size = [self.calc_number_of_states(), 1,          1] +  ([tf.TensorShape([self.order, self.alphabet_size + 2])] if self.order > 0 else [])

        # call_order: inputs = [1 126]
        # call_order: E = [18 1]
        # call_order: A = [18 18]
        # call_order: B = [18 126]
        # call_order: old_forward = [18 18] # why square here
        # call_order: old_loglik = [1 1]
        # call_order: count = [1 1]
        # call_order: count = 1
        # call_order: R if = [18 1]
        # call_order: alpha= [18 1]
        # call_order: use_sparse: loglik = [18 1]
        #
        # call_order: inputs = [1 126]
        # call_order: R if = [18 1]
        # call_order: E = [18 1]
        # call_order: A = [18 18]
        # call_order: B = [18 126]
        # call_order: old_forward = [1 18] #  why not square here
        # call_order: old_loglik = [1 1]
        # call_order: count = [1 1]
        # call_order: count = 1
        # call_order: alpha= [18 1]
        # call_order: use_sparse: loglik = [18 1]


        # self.init = True

        self.use_sparse = True



    def calc_number_of_states(self):
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
        # second terminal
        number_of_states += 1

        if self.order_transformed_input:
            number_of_states -= 1 #  we only need one terminal state

        return number_of_states

    def calc_number_of_transition_parameters(self):
        s = 1 # ig5'
        s += 1 # delete
        s += (self.nCodons + 1) * 2 # enter/exit insert
        s += self.nCodons # enter codon
        s += 1 # exit last codon

        return(s)

    def build(self, input_shape):
        self.transition_kernel = self.add_weight(shape = (self.calc_number_of_transition_parameters(),), # todo: (self.state_size[0], ) is this shape good?
                                                 initializer="random_normal",
                                                 trainable=True)


        self.emission_kernel = self.add_weight(shape = (self.calc_number_of_states() * 6**(self.order + 1), ),
                                              initializer="random_normal",
                                              trainable=True)

        self.init_kernel = self.add_weight(shape = (self.calc_number_of_states(),),
                                           initializer = "random_normal",
                                           trainable=True)



    def get_indices_and_values_from_transition_kernel_higher_order(self, w, nCodons):
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


        if self.order_transformed_input:
            # terminal -> terminal
            indices += [[index_of_terminal_1, index_of_terminal_1]]
            values = tf.concat([values, [1]], axis = 0)
        else:
            # terminal_1 -> terminal_1, a mix of true bases and X are emitted
            # terminal_1 -> terminal_2, only X are emitted
            indices += [[index_of_terminal_1, index_of_terminal_1], [index_of_terminal_1, index_of_terminal_1 +1]]
            values = tf.concat([values, [1] * 2], axis = 0)

            # terminal_2 -> terminal_2
            indices += [[index_of_terminal_1 +1, index_of_terminal_1 +1]]
            values = tf.concat([values, [1]], axis = 0)



        return indices, values


    @property
    def A(self):
        if self.use_sparse:
            return self.A_sparse()
        else:
            return self.A_dense()

    def A_sparse(self):
        indices, values = self.get_indices_and_values_from_transition_kernel_higher_order(self.transition_kernel, self.nCodons)
        transition_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [self.calc_number_of_states()] * 2)
        transition_matrix = tf.sparse.reorder(transition_matrix)
        transition_matrix = tf.sparse.softmax(transition_matrix)
        return transition_matrix

    def A_dense(self):
        return tf.sparse.to_dense(self.A_sparse())

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

    def get_indices_and_values_for_emission_higher_order_for_a_state(self, weights, k, indices, values, state, emissions, x_emissions_must_preceed, trainable = True):

        if self.order_transformed_input and emissions[-1] == "X":
            indices += [[state,(self.alphabet_size + 1) ** (self.order +1) ]]
            values[0] = tf.concat([values[0], [1]], axis = 0)
            return

        count_weights = 0
        allowed_emissions = [[state]]
        for i, e in enumerate(["N"] * (self.order - len(emissions) + 1) + list(emissions)[-self.order-1:]):
            allowed_emissions += [self.nucleotide_ambiguity_code_to_array(e) + ([4] if i < (self.order - x_emissions_must_preceed) else [] ) ]
        # might be faster, but contains emissions like [state, emission, 4, emission] (order = 0) where an emission is followd by "padded left flank", which will never occur
        # indices += list(product(*allowed_emissions))
        for x in product(*allowed_emissions):
            found_emission = False
            for i in range(1,self.order +1):
                if found_emission and x[i] == 4:
                    # print("not adding ", x)
                    break
                if x[i] != 4:
                    found_emission = True
            else:
                if self.order_transformed_input:
                    indices += [[state, higher_order_emission_to_id(x[1:], self.alphabet_size, self.order)]] # because first entry is state
                else:
                    indices += [x]
                count_weights += 1
        if trainable:
            values[0] = tf.concat([values[0], weights[k[0]:k[0] + count_weights]], axis = 0)
            k[0] += count_weights
        else:
            values[0] = tf.concat([values[0], [1] * count_weights], axis = 0)

    def get_indices_and_values_from_emission_kernel_higher_order_v02(self, w, nCodons, alphabet_size):
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

        if self.order_transformed_input:
            self.get_indices_and_values_for_emission_higher_order_for_a_state(\
                         weights,k,indices,values,8 + nCodons*3 + (nCodons+1)*3,"X",self.order)
        else:
            # terminal 1
            for i in range(1,self.order + 1):
                self.get_indices_and_values_for_emission_higher_order_for_a_state(\
                         weights,k,indices,values,8 + nCodons*3 + (nCodons+1)*3,"X" * i,self.order)
            # terminal 2
            self.get_indices_and_values_for_emission_higher_order_for_a_state(\
                         weights,k,indices,values,9 + nCodons*3 + (nCodons+1)*3,"X" * (self.order +1),self.order, trainable = False)

        return indices, values[0]

    @property
    def B(self):
        if self.use_sparse:
            return self.B_sparse()
        else:
            return self.B_dense()

    def B_sparse(self):
        indices, values = self.get_indices_and_values_from_emission_kernel_higher_order_v02(self.emission_kernel, self.nCodons, self.alphabet_size)

        if self.order_transformed_input:
            emission_matrix = tf.sparse.SparseTensor(indices = indices, \
                                                     values = values, \
                                                     dense_shape = [self.calc_number_of_states(), \
                                                                    (self.alphabet_size + 1) ** (self.order + 1) + 1])
            emission_matrix = tf.sparse.reorder(emission_matrix)
            emission_matrix = tf.sparse.softmax(emission_matrix)
        else:
            # [state, oldest emission, ..., second youngest emisson, current emission]
            emission_matrix = tf.sparse.SparseTensor(indices = indices, \
                                                     values = values, \
                                                     dense_shape = [self.state_size[0], \
                                                                    self.alphabet_size + 2] + \
                                                                    [self.alphabet_size + 2] * self.order)
            emission_matrix = tf.sparse.reorder(emission_matrix)
            emission_matrix = tf.sparse.reshape(emission_matrix, (self.state_size[0],-1))
            emission_matrix = tf.sparse.softmax(emission_matrix)
            emission_matrix = tf.sparse.reshape(emission_matrix, \
                                                [self.state_size[0],self.alphabet_size + 2] + \
                                                [self.alphabet_size + 2] * self.order)
        return emission_matrix

    def B_dense(self):
        return tf.sparse.to_dense(self.B_sparse())

    def get_indices_and_values_from_initial_kernel(self, weights, nCodons):
        k = 0

        # start and codons
        indices = [[i,0] for i in range(3 + nCodons*3)]
        values = weights[k:k + 3 + nCodons*3]
        k += 3 + nCodons*3
        # inserts
        indices += [[i,0] for i in range(8 + nCodons*3, 8 + nCodons*3 + (nCodons + 1)*3)]
        values = tf.concat([values, weights[k:k + (nCodons + 1)*3]], axis = 0)
        k += (nCodons + 1)*3

        return indices, values

    @property
    def I(self):
        if self.use_sparse:
            return self.I_sparse()
        else:
            return self.I_dense()

    def I_sparse(self):
        indices, values = self.get_indices_and_values_from_initial_kernel(self.init_kernel, self.nCodons)
        initial_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [self.calc_number_of_states(),1])
        initial_matrix = tf.sparse.reorder(initial_matrix)
        initial_matrix = tf.sparse.reshape(initial_matrix, (1,self.calc_number_of_states()))
        initial_matrix = tf.sparse.softmax(initial_matrix)
        initial_matrix = tf.sparse.reshape(initial_matrix, (self.calc_number_of_states(),1))
        return initial_matrix

    def I_dense(self):
        return tf.sparse.to_dense(self.I_sparse())

    def init_cell(self):
        self.inita = True
        self.correct_order = 0

    def call_order_transformed_input(self, inputs, states, training = None, verbose = False):

        if self.checksquare:
            old_forward, old_loglik, count, checksquare = states
        else:
            old_forward, old_loglik, count= states

        count = count + 1
        inputs = tf.dtypes.cast(inputs, tf.float32) # tried this to fix:
        # TypeError: Failed to convert elements of SparseTensor(indices=Tensor("cgp_hmm_layer/rnn/cgp_hmm_cell/SparseReorder:0", shape=(672, 2), dtype=int64), values=Tensor("cgp_hmm_layer/rnn/cgp_hmm_cell/SparseSoftmax/SparseSoftmax:0", shape=(672,), dtype=float32), dense_shape=Tensor("cgp_hmm_layer/rnn/cgp_hmm_cell/SparseTensor_1/dense_shape:0", shape=(2,), dtype=int64)) to Tensor. Consider casting elements to a supported type. See https://www.tensorflow.org/api_docs/python/tf/dtypes for supported TF dtypes.
        tf.print(self.correct_order,"call_order: inputs =", tf.shape(inputs))
        self.correct_order += 1

        if self.use_sparse:
            # inputs is shape batch * 126 (= (4+1)^3+1)
            # E = tf.linalg.matvec(tf.sparse.to_dense(self.B), tf.transpose(inputs)) # this works but isnt sparse
            # E = tf.linalg.matvec(inputs, self.B, b_is_sparse = True)

            # E = tf.sparse.sparse_dense_matmul(self.B, inputs) # todo: why does this also work? the deminesions shoudnt match
            E = tf.sparse.sparse_dense_matmul(self.B, tf.transpose(inputs))

            tf.print(self.correct_order,"call_order: E =", tf.shape(E))
            self.correct_order += 1
            tf.print(self.correct_order,"call_order: A =", tf.shape(self.A))
            self.correct_order += 1
            tf.print(self.correct_order,"call_order: B =", tf.shape(self.B))
            self.correct_order += 1
            tf.print(self.correct_order,"call_order: old_forward =", tf.shape(old_forward))
            self.correct_order += 1
            tf.print(self.correct_order,"call_order: old_loglik =", tf.shape(old_loglik))
            self.correct_order += 1
            tf.print(self.correct_order,"call_order: count =", tf.shape(count))
            self.correct_order += 1
            if self.checksquare:
                tf.print(self.correct_order,"call_order: checksquare =", tf.shape(checksquare))
                self.correct_order += 1


            tf.print(self.correct_order,"call_order: count =", count[0,0])
            self.correct_order += 1

            if self.inita:
                # tf.print("self.init = ", self.init)
                self.inita = False
            if count[0,0] == 1:
                R = self.I_dense()
                tf.print(self.correct_order,"call_order: R if =", tf.shape(R))
                self.correct_order += 1
            else:
                # R = tf.linalg.matvec(self.A, old_forward, transpose_a = True)
                R = tf.sparse.sparse_dense_matmul(self.A, old_forward, adjoint_b = True) # todo: can use transposed shape in state_size, then i can save this adjoint and alpha ? tf.transpose(alpha)

                Z_i_minus_1 = tf.reduce_sum(old_forward, axis=-1, keepdims = True)
                R /= Z_i_minus_1
                tf.print(self.correct_order,"call_order: R else =", tf.shape(R))
                self.correct_order += 1
            alpha = E * R
            alpha = tf.transpose(alpha)
            tf.print(self.correct_order,"call_order: alpha=", tf.shape(alpha))
            self.correct_order += 1

            if self.use_sparse:
                loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis=-1, keepdims = True, name = "loglik")) # todo keepdims = True?
                tf.print(self.correct_order,"call_order: use_sparse: loglik =", tf.shape(loglik))
                self.correct_order += 1
            else:
                loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis=-1, keepdims = True, name = "loglik")) # todo keepdims = True?

        if not self.use_sparse:
            E = tf.matmul(inputs, tf.transpose(self.B))

            if self.inita:
                # tf.print("self.init = ", self.init)
                self.inita = False
            if count[0,0] == 1:
                R = tf.transpose(self.I)
            else:
                R = tf.linalg.matvec(self.A, old_forward, transpose_a = True)
                Z_i_minus_1 = tf.reduce_sum(old_forward, axis=-1, keepdims = True)
                R /= Z_i_minus_1
            alpha = E * R

            loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis=-1, keepdims = True, name = "loglik")) # todo keepdims = True?
        tf.print()

        if self.checksquare:
            return [alpha, inputs, count, checksquare], [alpha, loglik, count, checksquare]
        else:
            return [alpha, inputs, count], [alpha, loglik, count]

    def call_old_inputs(self, inputs, states, training = None, verbose = False):
        if self.order > 0:
            # old inputs is from newest to oldest
            old_forward, old_loglik, count, old_inputs = states
        else:
            old_forward, old_loglik, count = states

        count = count + 1 # counts i in alpha(q,i)
        # tf.print("count =", count[0,0])
        # tf.print("inputs[0] =", inputs[0])

        # shape may be (batch_size,1) and not (batchsize,) thats why the second 0 is required
        if count[0,0] == 1: #todo: maby use states all zero
        # if self.init:
            # tf.print("count[0,0] =", count[0,0], "self.init =", self.init)
            batch_size = tf.shape(inputs)[0]

            if self.order > 0:
                old_inputs = tf.concat([tf.zeros((batch_size,self.order,4)), \
                                        tf.ones((batch_size,self.order,1)), \
                                        tf.zeros((batch_size,self.order,1))], axis = 2)


            R = tf.transpose(self.I)
            # E = tf.linalg.matmul(inputs, tf.transpose(self.B))

            E = tf.tensordot(inputs, tf.transpose(self.B), axes = (1,0))

            for i in range(self.order):
                old_inputs_i_expanded = tf.expand_dims(old_inputs[:,i,:], axis = -1)
                for j in range(i + 1, self.order):
                    old_inputs_i_expanded = tf.expand_dims(old_inputs_i_expanded, axis = -1)
                E = tf.multiply(old_inputs_i_expanded, E)
                E = tf.reduce_sum(E, axis = 1) # axis 0 is batch, so this has to be 1

            alpha = E * R
            loglik = tf.math.log(tf.reduce_sum(alpha, axis=-1, keepdims = True, name = "loglik")) # todo keepdims = True?
            self.init = False

        else:
            # tf.print("count[0,0] =", count[0,0], "self.init =", self.init)

            # # Is the density of A larger than approximately 15%? maybe just use dense matrix
            # R = tf.sparse.sparse_dense_matmul(self.A, old_forward, adjoint_a = True)

            R = tf.linalg.matvec(self.A, old_forward, transpose_a = True)

            E = tf.tensordot(inputs, tf.transpose(self.B), axes = (1,0))

            # todo: immer neue tensoren vielleicht langsam und doppelte for schleife
            # liebe ienfach index berechnen zb ACT 1*4 + 2*4 + 4*4
            for i in range(self.order):
                old_inputs_i_expanded = tf.expand_dims(old_inputs[:,i,:], axis = -1)
                for j in range(i + 1, self.order):
                    old_inputs_i_expanded = tf.expand_dims(old_inputs_i_expanded, axis = -1)
                E = tf.multiply(old_inputs_i_expanded, E)
                E = tf.reduce_sum(E, axis = 1) # axis 0 is batch, so this has to be 1

            Z_i_minus_1 = tf.reduce_sum(old_forward, axis=-1, keepdims = True)
            R /= Z_i_minus_1
            alpha = E * R
            loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis=-1, keepdims = True, name = "loglik")) # todo keepdims = True?

        # loglik = tf.squeeze(loglik)

        if self.order > 0:
            #                                                                        batch, order, one_hot, select every old input but the oldest in last position, this last one gets pushed off the tensor by the current input which is added at first position
            new_old_inputs = tf.concat([tf.expand_dims(inputs, axis = 1), old_inputs[:, :-1, :]], axis = 1)
            return [alpha, inputs, count], [alpha, loglik, count, new_old_inputs]
        else:
            #       return sequences        states
            return [alpha, inputs, count], [alpha, loglik, count]


    def call(self, inputs, states, training = None, verbose = False):
        if self.order_transformed_input:
            return self.call_order_transformed_input(inputs, states, training, verbose)
        else:
            return self.call_old_inputs(inputs, states, training, verbose)
