#!/usr/bin/env python3
from Model import Model

class Full_Model_Sparse(Model):

    # this overwrites the init from Model. alternatively i can omit it
    def __init__(self, self.config):
        Model.__init__(self, self.config)
        self.use_sparse = True
        self.use_dense = not self.use_sparse

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

    # def number_of_emissions(self):
    #     from Utility import n_emission_columns_in_B #TODO this should rather be emission state size
    #     return n_emission_columns_in_B(self.config.alphabet_size, self.config.order)
################################################################################
################################################################################
################################################################################

    # TODO: maybe use small letters instead of capital ones
    def I_indices(self):
        return [(q,) for q in range(self.number_of_states())]
################################################################################
    def A_indices(self):
        return [[q1,q2] for q1 in range(self.number_of_states()) for q2 in range(self.number_of_states())]
################################################################################
    def B_indices(self):
        return [[q, emission] for q in range(self.number_of_states()) for emission in range(self.number_of_emissions()]
################################################################################
################################################################################
################################################################################
    def I_kernel_size(self):
        return self.number_of_states()

    def A_kernel_size(self):
        return self.number_of_states() ** 2

    def B_kernel_size(self):
        return self.number_of_states() * self.number_of_emissions()
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
        transition_matrix = tf.sparse.SparseTensor(indices = self.A_indices(), \
                                                   values = weigths, \
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
    from self.config import self.config
    self.config = self.config("main_programm")
    f = Full_Model(self.config)
    print(f.A())
