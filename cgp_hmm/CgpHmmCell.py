#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from itertools import product

def prRed(skk): print("Cell\033[93m {}\033[00m" .format(skk))
# def prRed(s): pass

class CgpHmmCell_onedim(tf.keras.layers.Layer):
# class CgpHmmCell(tf.keras.layers.Layer):
    def __init__(self):
        # super(CgpHmmCell, self).__init__()
        super(CgpHmmCell_onedim, self).__init__()

        self.nCodons = 2

        self.alphabet_size = 4 # without terminal symbol and without "papped left flank" symbol

        self.order = 2 # order = 0 -> emission prob depends only on current emission
        #                  alpha                         old loglik count old_input
        self.state_size = [self.calc_number_of_states(), 1,         1] + ([tf.TensorShape([self.order, self.alphabet_size + 2])] if self.order > 0 else [])



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

        return number_of_states

    def calc_number_of_transition_parameters(self):
        s = 1 # ig5'
        s += 1 # delete
        s += (self.nCodons + 1) * 2 # enter/exit insert
        s += self.nCodons # enter codon

        return(s)

    def build(self, input_shape):
        self.transition_kernel = self.add_weight(shape = (self.calc_number_of_transition_parameters(),), # todo: (self.state_size[0], ) is this shape good?
                                                 initializer="random_normal",
                                                 trainable=True)


        self.emission_kernel = self.add_weight(shape = (self.calc_number_of_states() * 6**(self.order + 1), ),
                                              initializer="random_normal",
                                              trainable=True)

        self.init_kernel = self.add_weight(shape = (self.state_size[0],),
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
        values = tf.concat([values, [1-w[k-1]]], axis = 0)

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
        # but: TypeError: Cannot convert [0.5, 0.5] to EagerTensor of dtype int32
        values = tf.concat([values, [1] * 2], axis = 0) # this parameter doesnt have to be learned (i think)

        # terminal_1 -> terminal_1, terminal_1 -> terminal_2
        indices += [[index_of_terminal_1, index_of_terminal_1], [index_of_terminal_1, index_of_terminal_1 +1]]
        values = tf.concat([values, [1] * 2], axis = 0)

        # terminal_2 -> terminal_2
        indices += [[index_of_terminal_1 +1, index_of_terminal_1 +1]]
        values = tf.concat([values, [1]], axis = 0)



        return indices, values


    @property
    def A(self):
        indices, values = self.get_indices_and_values_from_transition_kernel_higher_order(self.transition_kernel, self.nCodons)
        transition_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [self.state_size[0]] * 2)
        transition_matrix = tf.sparse.reorder(transition_matrix)
        transition_matrix = tf.sparse.softmax(transition_matrix)
        transition_matrix = tf.sparse.to_dense(transition_matrix)
        return transition_matrix

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
                indices += [x]
                count_weights += 1
        if trainable:
            values[0] = tf.concat([values[0], weights[k[0]:k[0] + count_weights]], axis = 0)
            k[0] += count_weights
        else:
            values[0] = tf.concat([values[0], [1] * count_weights], axis = 0)


    def get_indices_and_values_from_emission_kernel_higher_order_v02(self, w, nCodons, alphabet_size):
        indices = []
        values = [[]] # will contain one tensor at index 0, wrapped it in a list such that it can be passed by reference
        weights = w
        k = [0]

        # ig 5'
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,0,"N",0)
        # start a
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,1,"A",0)
        # start t
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,2,"AT",0)
        # start g
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,3,"ATG",0, trainable = False)
        # codon_11
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,4,"ATGN",0)
        # codon_12
        self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,5,"ATGNN",0)
        # all other codons
        for state in range(6, 6 + nCodons*3 -2):
            self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,state,"N",0)
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
            self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,state,"N",0)
        # terminal 1
        for i in range(1,self.order + 1):
            self.get_indices_and_values_for_emission_higher_order_for_a_state(weights,k,indices,values,8 + nCodons*3 + (nCodons+1)*3,"X" * i,self.order)
        # terminal 2
        self.get_indices_and_values_for_emission_higher_order_for_a_state(    weights,k,indices,values,9 + nCodons*3 + (nCodons+1)*3,"X"*(self.order +1),self.order, trainable = False)

        return indices, values[0]

    @property
    def B(self):

        indices, values = self.get_indices_and_values_from_emission_kernel_higher_order_v02(self.emission_kernel, self.nCodons, self.alphabet_size)

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
        emission_matrix = tf.sparse.to_dense(emission_matrix)

        return emission_matrix

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
        indices, values = self.get_indices_and_values_from_initial_kernel(self.init_kernel, self.nCodons)
        initial_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [self.state_size[0],1])
        initial_matrix = tf.sparse.reorder(initial_matrix)
        initial_matrix = tf.sparse.reshape(initial_matrix, (1,self.state_size[0]))
        initial_matrix = tf.sparse.softmax(initial_matrix)
        initial_matrix = tf.sparse.reshape(initial_matrix, (self.state_size[0],1))
        initial_matrix = tf.sparse.to_dense(initial_matrix)
        return initial_matrix

    def call(self, inputs, states, training = None, verbose = False):
        if self.order > 0:
            old_forward, old_loglik, count, old_inputs = states
        else:
            old_forward, old_loglik, count = states

        count = count + 1 # counts i in alpha(q,i)

        # shape may be (batch_size,1) and not (batchsize,) thats why the second 0 is required
        if count[0,0] == 1:
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

        else:
            # # Is the density of A larger than approximately 15%? maybe just use dense matrix
            # R = tf.sparse.sparse_dense_matmul(self.A, old_forward, adjoint_a = True)

            R = tf.linalg.matvec(self.A, old_forward, transpose_a = True)

            E = tf.tensordot(inputs, tf.transpose(self.B), axes = (1,0))

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


class CgpHmmCell(CgpHmmCell_onedim):
    def __init__(self):
        super(CgpHmmCell, self).__init__()
