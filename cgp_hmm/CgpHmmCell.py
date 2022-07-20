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
        self.use_mask = False
        self.use_higher_order = True

        # ig 5'
        number_of_states = 1
        # start
        number_of_states += 3
        # codons
        number_of_states += 3 * self.nCodons
        # codon inserts
        number_of_states += 3 * (self.nCodons + 1)
        # stop
        number_of_states += 7 if not self.use_higher_order else 3
        # ig 3'
        number_of_states += 1
        # terminal
        number_of_states += 1



        self.alphabet_size = 4 # without terminal symbol
        #                  alpha             loglik, count, y_{i-2},                y_{i-1}
        self.state_size = [number_of_states, 1,      1,     self.alphabet_size + 2, self.alphabet_size + 2]


    def build(self, input_shape):
        if self.use_higher_order:
            self.transition_kernel = self.add_weight(shape = (self.state_size[0]*10,), # todo: (self.state_size[0], ) is this shape good?
                                                     initializer="random_normal",
                                                     trainable=True)

            # shape = ((2 + self.nCodons*3 + (self.nCodons+1)*3)*self.alphabet_size , )

            # multiply this by 4*4=16 as an upper bound to how many parameters are needed
            shape = (((2 + self.nCodons*3 + (self.nCodons+1)*3)*self.alphabet_size)*16 , )
            self.emission_kernel = self.add_weight(shape = shape,
                                                  initializer="random_normal",
                                                  trainable=True)

            self.init_kernel = self.add_weight(shape = (self.state_size[0],),
                                               initializer = "random_normal",
                                               trainable=True)
        elif self.use_mask:
            # the number of values in transition_kernel doesnt need to match state_size
            # these are all non zero values in the transisiotn matrix
            # todo we need more than just self.state_size[0] weights
            self.transition_kernel = self.add_weight(shape = (self.state_size[0]*5,), # todo: (self.state_size[0], ) is this shape good?
                                                     initializer="random_normal",
                                                     trainable=True)
            shape = ((2 + self.nCodons*3 + (self.nCodons+1)*3)*self.alphabet_size , )
            self.emission_kernel = self.add_weight(shape = shape,
                                                  initializer="random_normal",
                                                  trainable=True)
            self.init_kernel = self.add_weight(shape = (self.state_size[0],),
                                               initializer = "random_normal",
                                               trainable=True)
        else:
            self.transition_kernel = self.add_weight(shape = (self.state_size[0], \
                                                              self.state_size[0]),
                                                     initializer="random_normal",
                                                     trainable=True)
            self.emission_kernel = self.add_weight(shape = (self.state_size[0], \
                                                            self.alphabet_size),
                                                  initializer="random_normal",
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
        k += nCodons + 1
        # continuing an insert
        indices += [[offset +2 + i*3, offset + i*3] for i in range(nCodons +1)]
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

        # ig -> ig, terminal
        indices += [[7 + nCodons*3, 7 + nCodons*3], [7 + nCodons*3, 8 + nCodons*3 + (nCodons + 1) *3]]
        # values = tf.concat([values, [.5] * 2], axis = 0) # this parameter doesnt have to be learned (i think)
        # .5 can be any other number, since softmax(x,x) = [.5, .5]
        # but: TypeError: Cannot convert [0.5, 0.5] to EagerTensor of dtype int32
        values = tf.concat([values, [1] * 2], axis = 0) # this parameter doesnt have to be learned (i think)

        # terminal -> terminal
        indices += [[8 + nCodons*3 + (nCodons + 1) *3, 8 + nCodons*3 + (nCodons + 1) *3]]
        values = tf.concat([values, [1]], axis = 0)

        return indices, values

    def get_indices_and_values_from_transition_kernel(self, w, nCodons):
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
        values = tf.concat([values, w[k: k + nCodons]], axis = 0)
        k += nCodons
        # first to second codon position
        indices += [[4 + i*3, 5 + i*3] for i in range(nCodons)]
        values = tf.concat([values, [1] * nCodons], axis = 0)
        # second to third codon position
        indices += [[5 + i*3, 6 + i*3] for i in range(nCodons)]
        values = tf.concat([values, [1] * nCodons], axis = 0)

        # inserts
        offset = 12 + 3*nCodons
        # begin inserts

        use_inserts = False
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
        k += nCodons + 1
        # continuing an insert
        indices += [[offset +2 + i*3, offset + i*3] for i in range(nCodons +1)]
        values = tf.concat([values, 1-w[k: k + nCodons +1]], axis = 0)
        k += nCodons + 1

        # deletes
        i_delete = [3 + i*3 for i in range(nCodons) for j in range(nCodons-i)]
        j_delete = [4 + j*3 for i in range(1,nCodons+1) for j in range(i,nCodons+1)]
        indices += [[i,j] for i,j in zip(i_delete, j_delete)]
        # print("deletes =", [1-w[k] * w[k]**((j-i)/3) for i,j in zip(i_delete, j_delete)])
        values = tf.concat([values, [1-w[k] * w[k]**int((j-i)/3) for i,j in zip(i_delete, j_delete)]], axis = 0)
        k += 1

        # stop T -> ...
        indices += [[4 + nCodons*3, 5 + i + nCodons*3] for i in range(3)]
        values = tf.concat([values, w[k: k + 3]], axis = 0)
        k += 3

        # second to third position in stop
        indices += [[5 + i + nCodons*3, 8 + i + nCodons*3] for i in range(3)]
        values = tf.concat([values, [1]*3], axis = 0)

        # stop -> ig 3'
        indices += [[8 + i + nCodons*3, 11 + nCodons*3] for i in range(3)]
        values = tf.concat([values, [1] * 3], axis = 0)

        # ig -> ig, terminal
        indices += [[11 + nCodons*3, 11 + nCodons*3], [11 + nCodons*3, 12 + nCodons*3 + (nCodons + 1) *3]]
        # values = tf.concat([values, [.5] * 2], axis = 0) # this parameter doesnt have to be learned (i think)
        # .5 can be any other number, since softmax(x,x) = [.5, .5]
        # but: TypeError: Cannot convert [0.5, 0.5] to EagerTensor of dtype int32
        values = tf.concat([values, [1] * 2], axis = 0) # this parameter doesnt have to be learned (i think)

        # terminal -> terminal
        indices += [[12 + nCodons*3 + (nCodons + 1) *3, 12 + nCodons*3 + (nCodons + 1) *3]]
        values = tf.concat([values, [1]], axis = 0)

        return indices, values

    @property
    def A(self):
        if self.use_higher_order:
            indices, values = self.get_indices_and_values_from_transition_kernel_higher_order(self.transition_kernel, self.nCodons)
        elif self.use_mask:
            indices, values = self.get_indices_and_values_from_transition_kernel(self.transition_kernel, self.nCodons)

        transition_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [self.state_size[0]] * 2)
        transition_matrix = tf.sparse.reorder(transition_matrix)
        transition_matrix = tf.sparse.softmax(transition_matrix)
        transition_matrix = tf.sparse.to_dense(transition_matrix)
        # else:
        #     transition_matrix = tf.nn.softmax(self.transition_kernel, axis=-1, name="A")
        return transition_matrix

        # used these for manual run, cant use these for training, since the add_weight method isnt used
        # return np.array([0.1,  0.2,   0.3,  0.2,  0.2,\
        #                  0.2,  0.2,   0.2,  0.2,  0.2,\
        #                  0.2,  0.15,  0.15, 0.3 , 0.2,\
        #                  0.3,  0.2,   0.4,  0.0,  0.1,\
        #                  0,    0.2,   0.5,  0.3,  0.0], dtype = np.float32).reshape((5,5))
    def get_indices_and_values_from_emission_kernel_higher_order(self, w, nCodons, alphabet_size):
        k = 0
        # ig 5'
        #           state, prevprev emission, prev emission, current emission,
        indices = [[0,i,j,k] for i in range(alphabet_size + 1) for j in range(max(alphabet_size, i+1)) for k in range(alphabet_size)]

        #                                                     0aig, 0cig, 0gig, 0tig
        values = w[k:k + (alphabet_size+1)*alphabet_size**2 + 4]
        k += (alphabet_size+1)*alphabet_size**2 + 4

        # start a
        indices += [[1,i,j,0] for i in range(alphabet_size + 1) for j in range(alphabet_size)]
        values = tf.concat([values,  w[k:k + (alphabet_size+1)*alphabet_size]], axis = 0)
        k += (alphabet_size+1)*alphabet_size

        # start t
        indices += [[2,i,0,3] for i in range(alphabet_size)]
        values = tf.concat([values,  w[k:k + alphabet_size]], axis = 0)
        k += alphabet_size

        # start g
        indices += [[3,0,3,2]]
        print("-------------------------------------------")
        print("values and [1]")
        print(values)
        print([1])
        print("-------------------------------------------")
        values = tf.concat([values,  [1]], axis = 0)
        print("concat done")

        number_of_added_const_parameters = 1

        # codons
        # this needs to be adjusted if markov chain can start in a state other than ig5'
        # first two positions of first codons must be handled seperatly
        indices += [[4,3,2, i] for i in range(alphabet_size)]
        values = tf.concat([values, w[k:k + alphabet_size]], axis = 0)
        k += 4

        indices += [[5,2,i, j] for i in range(alphabet_size) for j in range(alphabet_size)]
        values = tf.concat([values, w[k:k+alphabet_size**2]], axis = 0)
        k += alphabet_size**2

        # all other codons
        indices += [[6 + i, j, k, l] for i in range(nCodons * 3 - 2) \
                                     for j, k, l in product(range(alphabet_size), repeat = 3)]
        values = tf.concat([values, w[k:k + (nCodons * 3 - 2) * alphabet_size**3]], axis = 0)
        k +=  (nCodons * 3 - 2) * alphabet_size**3

        # stop t                          t
        indices += [[4 + nCodons*3, i, j, 3] for i,j in product(range(alphabet_size), repeat = 2)]
        values = tf.concat([values, w[k:k + alphabet_size**2]], axis = 0)
        k += alphabet_size**2

        # second stop                  t  a
        indices += [[5 + nCodons*3, i, 3, 0] for i in range(alphabet_size)]
        values = tf.concat([values, w[k: k + alphabet_size]], axis = 0)
        k += alphabet_size

        # second stop                  t  g
        indices += [[5 + nCodons*3, i, 3, 2] for i in range(alphabet_size)]
        values = tf.concat([values, w[k: k + alphabet_size]], axis = 0)
        k += alphabet_size

        # third stop                t  a  a
        indices += [[6 + nCodons*3, 3, 0, 0]]
        values = tf.concat([values, [1]], axis = 0)

        # third stop                t  a  g
        indices += [[6 + nCodons*3, 3, 0, 2]]
        values = tf.concat([values, [1]], axis = 0)

        # third stop                t  g  a
        indices += [[6 + nCodons*3, 3, 2, 0]]
        values = tf.concat([values, [1]], axis = 0)

        # ig3'
        indices += [[7 + nCodons*3, i,j,k] for i,j,k in product(range(alphabet_size), repeat=3)]# what about terminal symbol
        values = tf.concat([values, w[k:k + alphabet_size**3]], axis = 0)
        k += alphabet_size**3

        # inserts
        indices += [[8 + nCodons*3 + l, i,j,k] for l in range(3*(nCodons+1)) \
                                               for i,j,k in product(range(alphabet_size), repeat=3)]
        values = tf.concat([values, w[k:k + alphabet_size**3*3*(nCodons+1)]], axis = 0)
        k += alphabet_size**3*3*(nCodons+1)

        # terminal                                       terminal symbol
        indices += [[8 + nCodons*3 + (nCodons+1)*3, i,j,5] for i,j in product(range(alphabet_size), repeat=2)]
        # cant assign the emissions {acgt} ter. ter. and ter. ter. ter. in this loop,
        # bc i need these emission to not add to calculated probability of seq
        values = tf.concat([values, w[k:k + alphabet_size**2]], axis = 0)# todo: maybe just assign all equal probability
        k += alphabet_size**2

        indices += [[8 + nCodons*3 + (nCodons+1)*3, i, 5, 5] for i in range(alphabet_size)]
        values = tf.concat([values, w[k:k + alphabet_size]], axis = 0)
        k += alphabet_size

        indices += [[8 + nCodons*3 + (nCodons+1)*3, 5, 5, 5]]
        values = tf.concat([values, [1]], axis = 0)

        return indices, values


    def get_indices_and_values_from_emission_kernel(self, w, nCodons, alphabet_size):
        k = 0
        # ig 5'
        indices = [[0,i] for i in range(alphabet_size)]
        values = w[:4]

        k += 4
        # start a
        indices += [[1,0]]
        values = tf.concat([values, [1]], axis = 0)

        # start t
        indices += [[2,3]]
        values = tf.concat([values, [1]], axis = 0)
        # start g
        indices += [[3,2]]
        values = tf.concat([values, [1]], axis = 0)

        # codons
        indices += [[4 + i, j] for i in range(nCodons*3) for j in range(alphabet_size)]
        values = tf.concat([values, w[k: k + nCodons*3*alphabet_size]], axis = 0)
        k += nCodons*3*alphabet_size

        # stop t
        indices += [[4 + nCodons*3, 3]]
        values = tf.concat([values, [1]], axis = 0)
        # stop tAa
        indices += [[5 + nCodons*3, 0]]
        values = tf.concat([values, [1]], axis = 0)
        # stop tGa
        indices += [[6 + nCodons*3, 2]]
        values = tf.concat([values, [1]], axis = 0)
        # stop tAg
        indices += [[7 + nCodons*3, 0]]
        values = tf.concat([values, [1]], axis = 0)

        # stop taA
        indices += [[8 + nCodons*3, 0]]
        values = tf.concat([values, [1]], axis = 0)
        # stop tgA
        indices += [[9 + nCodons*3, 0]]
        values = tf.concat([values, [1]], axis = 0)
        # stop taG
        indices += [[10 + nCodons*3, 2]]
        values = tf.concat([values, [1]], axis = 0)

        # ig 3'
        indices += [[11 + nCodons*3, i] for i in range(alphabet_size)]
        values = tf.concat([values, w[k: k + alphabet_size]], axis = 0)
        k += alphabet_size

        # inserts
        indices += [[12 + nCodons*3 + i, j] for i in range(3*(nCodons+1)) for j in range(alphabet_size)]
        values = tf.concat([values, w[k: k + (nCodons+1)*3*alphabet_size]], axis = 0)
        k += (nCodons+1)*3*alphabet_size

        # terminal
        indices += [[12 + nCodons*3 + (nCodons + 1) *3, alphabet_size]]
        values = tf.concat([values, [1]], axis = 0)

        return indices, values

    @property
    def B(self):
        if self.use_higher_order:
            print("self.emission_kernel")
            print(self.emission_kernel)
            indices, values = self.get_indices_and_values_from_emission_kernel_higher_order(self.emission_kernel, self.nCodons, self.alphabet_size)
            print("shapes")
            print(tf.shape(indices))
            print(tf.shape(values))
            emission_matrix = tf.sparse.SparseTensor(indices = indices, \
                                                     values = values, \
                                                     dense_shape = [self.state_size[0], \
                                                                    self.alphabet_size + 2, \
                                                                    self.alphabet_size + 2, \
                                                                    self.alphabet_size + 2])
            emission_matrix = tf.sparse.reorder(emission_matrix)
            # todo: already init matrix in this shape, then only reshape it once
            # to make it ready to use
            emission_matrix = tf.sparse.reshape(emission_matrix, (self.state_size[0],-1))
            emission_matrix = tf.sparse.softmax(emission_matrix)
            emission_matrix = tf.sparse.reshape(emission_matrix, (self.state_size[0],self.alphabet_size + 2,self.alphabet_size + 2,self.alphabet_size + 2))
            emission_matrix = tf.sparse.to_dense(emission_matrix)


        elif self.use_mask:
            indices, values = self.get_indices_and_values_from_emission_kernel(self.emission_kernel, self.nCodons, self.alphabet_size)
            emission_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [self.state_size[0], self.alphabet_size + 1])
            emission_matrix = tf.sparse.reorder(emission_matrix)
            emission_matrix = tf.sparse.softmax(emission_matrix)
            emission_matrix = tf.sparse.to_dense(emission_matrix)
        else:
            emission_matrix = tf.nn.softmax(self.emission_kernel, axis=-1, name="B")
        return emission_matrix

        # return np.array([0.1,  0.2,   0.3,  0.4 ,\
        #                  0.2,  0.15,  0.15, 0.5 ,\
        #                  0.3,  0.2,   0.5,  0   ,\
        #                  0,    0.2,   0.5,  0.3 ,\
        #                  0.25, 0.25, 0.25,  0.25], dtype = np.float32).reshape((5,4))

    # def get_initial_state(self):
    #     return [[1,0,0,0,0],[0,0,0,0,0]]
    @property
    def I(self):
        # the markov chain can only start in first state,
        # easy to generalize
        initial_matrix = tf.sparse.SparseTensor(indices = [[0,0]], values = [self.init_kernel[0]], dense_shape = [self.state_size[0],1])
        initial_matrix = tf.sparse.reorder(initial_matrix)
        initial_matrix = tf.sparse.softmax(initial_matrix)
        initial_matrix = tf.sparse.to_dense(initial_matrix)
        return initial_matrix

    def mask_call(self, inputs, states, training = None, verbose = False):
        old_forward, old_loglik, count, old_inputs_2, old_inputs_1 = states
        count = count + 1 # counts i in alpha(q,i)

        if count[0,0] == 1:
            # without initial_dist
            # #                                                                only allow to start in first state, reshape to vector
            # alpha = tf.reshape(tf.linalg.matmul(inputs, tf.transpose(self.B))[:,0], (tf.shape(inputs)[0],1)) # todo use transpose_b = True
            # z = tf.zeros((tf.shape(inputs)[0], self.state_size[0] - 1), dtype = tf.float32)
            # alpha = tf.concat((alpha, z),1) # 1 is axis
            # loglik = tf.math.log(tf.reduce_sum(alpha, axis=-1, keepdims = True, name = "loglik")) # todo keepdims = True?

            batch_size = tf.shape(inputs)[0]
            old_inputs_1 = tf.concat([tf.zeros((batch_size,4)), tf.ones((batch_size,1)), tf.zeros((batch_size,1))], axis = 1)
            old_inputs_2 = old_inputs_1

            R = tf.transpose(self.I)
            # E = tf.linalg.matmul(inputs, tf.transpose(self.B))

            tf.print("self.B")
            tf.print(tf.shape(self.B))

            E = tf.tensordot(inputs, tf.transpose(self.B), axes = (1,0))
            old_inputs_1_expanded = tf.expand_dims(old_inputs_1, axis = -1)
            old_inputs_1_expanded = tf.expand_dims(old_inputs_1_expanded, axis = -1)
            # now: old_inputs_1 has shape[batchsize 4 1 1]
            # now it can be broadcasted to [batchsize 4 4 #states]
            E = tf.multiply(old_inputs_1_expanded, E)
            # reduce sum is along axis that is as large as emission alphabet_size
            E = tf.reduce_sum(E, axis = 1) # axis 0 is batch, so this has to be 1
            old_inputs_2_expanded = tf.expand_dims(old_inputs_2, axis = -1)
            E = tf.multiply(old_inputs_2_expanded, E)
            E = tf.reduce_sum(E, axis = 1)
            alpha = E * R
            loglik = tf.math.log(tf.reduce_sum(alpha, axis=-1, keepdims = True, name = "loglik")) # todo keepdims = True?

        else:
            # R = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(self.A), old_forward)

            # # Is the density of A larger than approximately 15%? maybe just use dense matrix
            # R = tf.sparse.sparse_dense_matmul(self.A, old_forward, adjoint_a = True)

            R = tf.linalg.matvec(self.A, old_forward, transpose_a = True)

            # E = tf.linalg.matmul(inputs, tf.transpose(self.B))
            E = tf.tensordot(inputs, tf.transpose(self.B), axes = (1,0))
            old_inputs_1_expanded = tf.expand_dims(old_inputs_1, axis = -1)
            old_inputs_1_expanded = tf.expand_dims(old_inputs_1_expanded, axis = -1)
            # now: old_inputs_1 has shape[batchsize 4 1 1]
            # now it can be broadcasted to [batchsize 4 4 #states]
            E = tf.multiply(old_inputs_1_expanded, E)
            # reduce sum is along axis that is as large as emission alphabet_size
            E = tf.reduce_sum(E, axis = 1) # axis 0 is batch, so this has to be 1
            old_inputs_2_expanded = tf.expand_dims(old_inputs_2, axis = -1)
            E = tf.multiply(old_inputs_2_expanded, E)
            E = tf.reduce_sum(E, axis = 1)

            Z_i_minus_1 = tf.reduce_sum(old_forward, axis=-1, keepdims = True)
            R /= Z_i_minus_1
            alpha = E * R
            loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis=-1, keepdims = True, name = "loglik")) # todo keepdims = True?

        # loglik = tf.squeeze(loglik)
        #       return sequences        states
        return [alpha, inputs, count], [alpha, loglik, count, old_inputs_1, inputs]

    def dense_call(self, inputs, states, training = None, verbose = False):
        old_forward, old_loglik, count = states
        count = count + 1 # counts i in alpha(q,i)
        if count[0,0] == 1:
            alpha = tf.reshape(tf.linalg.matmul(inputs, tf.transpose(self.B))[:,0], (tf.shape(inputs)[0],1)) # todo use transpose_b = True
            z = tf.zeros((tf.shape(inputs)[0], self.state_size[0] - 1), dtype = tf.float32)
            alpha = tf.concat((alpha, z),1) # 1 is axis#prRed("alpha =")
            loglik = tf.math.log(tf.reduce_sum(alpha, axis=-1, keepdims = True, name = "loglik")) # todo keepdims = True?
        else:
            prRed("old_forward =")
            print(old_forward)
            prRed("self.A = ")
            print(self.A)
            R = tf.linalg.matvec(self.A, old_forward, transpose_a = True)
            E = tf.linalg.matmul(inputs, tf.transpose(self.B))
            Z_i_minus_1 = tf.reduce_sum(old_forward, axis=-1, keepdims = True)
            R /= Z_i_minus_1
            alpha = E * R
            loglik = old_loglik + tf.math.log(tf.reduce_sum(alpha, axis=-1, keepdims = True, name = "loglik")) # todo keepdims = True?

            # loglik = tf.squeeze(loglik)
        return [alpha, inputs, count], [alpha, loglik, count]

    def log_call(self, inputs, states, training = None, verbose = False):
        id = np.random.randint(100)
        old_forward, old_loglik, count = states

        if verbose:
            prRed("in call of CgpHmmCell")
            prRed("count " + str(count[0,0]))
            print("id =", id, "inputs = ", inputs)
            print("states ", states)
            print("A = ", self.A) # states x states
            print("B = ", self.B) # states x emissions


        count = count + 1 # counts i in alpha(q,i)

        #            if count[0,0] == 1: # 0 for first seq in batch, second 0 bc shape is [batch_size, 1]
        # tensorflow.python.framework.errors_impl.OperatorNotAllowedInGraphError: Exception encountered when calling layer "cgp_hmm_cell" (type CgpHmmCell).
        #
        # using a `tf.Tensor` as a Python `bool` is not allowed: AutoGraph did convert this function. This might indicate you are trying to use an unsupported feature.

        # todo: also use multiplication with 0 instead of "if"

        if count[0,0] == 1: # 0 for first seq in batch, second 0 bc shape is [batch_size, 1]
            #       column vector                                                    batch_size          one column
            alpha = tf.reshape(tf.linalg.matmul(inputs, tf.transpose(self.B))[:,0], (tf.shape(inputs)[0],1)) # todo use transpose_b = True
            z = tf.zeros((tf.shape(inputs)[0], self.state_size[0] - 1), dtype = tf.float32)
            alpha = tf.concat((alpha, z),1) # 1 is axis#prRed("alpha =")
            alpha = tf.math.log(alpha)

            if verbose:
                prRed("alpha =")
                print(alpha)

        else:
            R = tf.linalg.matvec(self.A, tf.math.exp(old_forward), transpose_a = True)
            R = tf.math.log(R)
            E = tf.linalg.matmul(inputs, tf.transpose(self.B))
            E = tf.math.log(E)

            if verbose:
                prRed("R =")
                print(R)
                prRed("E =")
                print(E)

            alpha = E + R

            if verbose:
                prRed("alpha =")
                print(alpha)

        loglik = tf.math.log(tf.reduce_sum(tf.math.exp(alpha), axis=-1, keepdims=True, name="loglik"))

        if verbose:
            prRed("loglik =")
            print(loglik[0,0])

        #      return sequences = True, cell state
        return [alpha, inputs, count], [alpha, loglik, count]



    def call(self, inputs, states, training = None, verbose = False):
        if self.use_mask or self.use_higher_order:
            return self.mask_call(inputs, states,training = training, verbose = verbose)
        else:
            return self.dense_call(inputs, states, training = training, verbose = verbose)

class CgpHmmCell(CgpHmmCell_onedim):
    def __init__(self):
        super(CgpHmmCell, self).__init__()
