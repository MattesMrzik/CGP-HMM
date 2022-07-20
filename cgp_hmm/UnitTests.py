#!/usr/bin/env python3
import unittest
import numpy as np
import tensorflow as tf
import Training
from CgpHmmLayer import CgpHmmLayer
from CgpHmmCell import CgpHmmCell
from Utility import state_id_to_description

import Utility

class TestCgpHmmCell(unittest.TestCase):
    def off_test_get_indices_and_values_from_transition_kernel(self):
        cell = CgpHmmCell()
        #                                                                                        weights, 2 codons
        indices, values = cell.get_indices_and_values_from_transition_kernel(np.array(list(range(10,100))),2)
        print("indices =", indices)
        print("values =", values)
        transition_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [cell.state_size[0]] * 2)
        transition_matrix = tf.sparse.reorder(transition_matrix)
        print(tf.sparse.to_dense(transition_matrix))
    def off_test_get_indices_and_values_from_emission_kernel(self):
        cell = CgpHmmCell()
        #                                                                                  100 weights,     2 codons, 4 = alphabet_size
        indices, values = cell.get_indices_and_values_from_emission_kernel(np.array(list(range(10,100))),2,4)
        print("indices =", len(indices))
        print("values =", len(values))
        emission_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [cell.state_size[0],5])
        emission_matrix = tf.sparse.reorder(emission_matrix)
        print(tf.sparse.to_dense(emission_matrix))

    def test_get_indices_and_values_from_transition_kernel_higher_order(self):
        cell = CgpHmmCell()
        #                                                                                        weights, 2 codons
        indices, values = cell.get_indices_and_values_from_transition_kernel_higher_order(np.array(list(range(10,100))),2)
        print("indices =", indices)
        print("values =", values)
        transition_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [cell.state_size[0]] * 2)
        transition_matrix = tf.sparse.reorder(transition_matrix)
        print(tf.sparse.to_dense(transition_matrix))

    def off_test_get_indices_and_values_from_emission_kernel_higher_order(self):
        cell = CgpHmmCell()
        #                                                                                  100 weights,     2 codons, 4 = alphabet_size
        indices, values = cell.get_indices_and_values_from_emission_kernel_higher_order(np.array(list(range(10,10000))),2,4)
        print("indices =", len(indices))
        print("values =", len(values))
        emission_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [cell.state_size[0],6,6,6])
        emission_matrix = tf.sparse.reorder(emission_matrix)
        emission_matrix = tf.sparse.to_dense(emission_matrix)
        for state in range(len(emission_matrix)):
            tf.print(state_id_to_description(state, 2))
            tf.print(emission_matrix[state], summarize = -1)
            tf.print("---------------------------------------------")


class TestViterbi(unittest.TestCase):
    def off_test_Viterbi(self):# todo also test with passing a0
        state_space_size = 3
        emission_space_size = 4
        a = tf.nn.softmax(np.random.rand(state_space_size, state_space_size))
        b = tf.nn.softmax(np.random.rand(state_space_size, emission_space_size))
        a0 = tf.nn.softmax(np.random.rand(state_space_size))
        n = 6
        l = 5
        state_seq, emission_seq = Utility.generate_state_emission_seqs(a,b,n,l,a0)
        # print("true", np.argmax( state_seq, axis = -1))

        for i, y in enumerate(emission_seq):
            # print(i, np.argmax(seq, axis = -1))
            print("always on state 0")
            x = Utility.viterbi_log_version(a,b,y)
            print("x of viterbi =\t", x)
            print(Utility.P_of_X_Y(a,b,x,y))
            x = Utility.brute_force_viterbi_log_version(a,b,y)
            print("x of brute force =\t", x)
            print(Utility.P_of_X_Y(a,b,x,y))
            self.assertTrue(all(Utility.viterbi_log_version(a,b,y) \
                                == \
                                Utility.brute_force_viterbi_log_version(a,b,y)))
        for i, y in enumerate(emission_seq):
            # print(i, np.argmax(seq, axis = -1))
            print("--> a0")
            x = Utility.viterbi_log_version(a,b,y, a0)
            print("x of viterbi =\t", x)
            print(Utility.P_of_X_Y(a,b,x,y,a0))
            x = Utility.brute_force_viterbi_log_version(a,b,y, a0)
            print("x of brute force =\t", x)
            print(Utility.P_of_X_Y(a,b,x,y,a0))
            self.assertTrue(all(Utility.viterbi_log_version(a,b,y, a0) \
                                == \
                                Utility.brute_force_viterbi_log_version(a,b,y, a0)))

class Test_Helpers(unittest.TestCase):
    def off_test_P_of_X_Y(self):
        state_space_size = 3
        emission_space_size = 4
        a = tf.nn.softmax(np.random.rand(state_space_size, state_space_size))
        b = tf.nn.softmax(np.random.rand(state_space_size, emission_space_size))
        a0 = tf.nn.softmax(np.random.rand(state_space_size))
        n = 100
        l = 5
        state_seq, emission_seq = Utility.generate_state_emission_seqs(a,b,n,l)
        for i, (x,y) in enumerate(zip(state_seq, emission_seq)):
            # print(Utility.P_of_X_Y(a,b,x,y,a0), " =?= ", np.exp(Utility.P_of_X_Y_log_version(a,b,x,y,a0)))
            self.assertAlmostEqual(Utility.P_of_X_Y(a,b,x,y), \
                                   np.exp(Utility.P_of_X_Y_log_version(a,b,x,y)), \
                                   delta = 0.0000001)

        for i, (x,y) in enumerate(zip(state_seq, emission_seq)):
            # print(Utility.P_of_X_Y(a,b,x,y,a0), " =?= ", np.exp(Utility.P_of_X_Y_log_version(a,b,x,y,a0)))
            self.assertAlmostEqual(Utility.P_of_X_Y(a,b,x,y,a0), \
                                   np.exp(Utility.P_of_X_Y_log_version(a,b,x,y,a0)), \
                                   delta = 0.0000001)

class TestForward(unittest.TestCase):
    def off_test_tf_scaled_forward_to_manual_scaled_forward(self):
        pass
    # manual forward <-- using the z of manual --> manual scaled forward
    def off_test_manual_scaled_forward_to_manual_true_forward(self):
        pass
    # tf scaled forward <-- using the z of tf --> manual forward
    def off_test_tf_scaled_transformed_forward_to_manual_true_forward(self):
        pass
    # need to test manual true forward ie check if sum q alpha qn is same as brute force p(y)


    # this is kept just in case
    def off_test_tf_log_forward(self):
        # rename method log_call() in CgpHmmCell.py to call()

        n = 6
        l = 5

        cgp_hmm_layer = CgpHmmLayer()

        # manually adjust these 2 variables to be the same as in CgpHmmCell
        state_space_size = 2
        emission_space_size = 4

        cgp_hmm_layer.build((n,l,emission_space_size))

        a = tf.nn.softmax(np.random.rand(state_space_size, state_space_size))
        b = tf.nn.softmax(np.random.rand(state_space_size, emission_space_size))
        state_seq, emission_seq = Utility.generate_state_emission_seqs(a,b,n,l, one_hot=True) # todo check with a0
        # these a and b are not the ones used as parameters in the rnn,
        # this doesnt matter as long as rnn and manual forward use the same parameters

        alpha_seq, inputs_seq, \
        count_seq, alpha_state, \
        loglik_state, count_state = cgp_hmm_layer.F(tf.cast(emission_seq, \
                                                            dtype = tf.float32)) # cast input to in or float

        a = cgp_hmm_layer.C.A
        b = cgp_hmm_layer.C.B

        # print return sequences
        for i in range(tf.shape(alpha_seq)[0]): # = batch_size
            manual_forward, _ =  Utility.forward_log_version(a,b,np.argmax(emission_seq[i], axis=-1))
            for j in range(tf.shape(alpha_seq)[1]): # = seq length
                # tf.print(count_seq[i,j], inputs_seq[i,j], alpha_seq[i,j])
                for q in range(state_space_size):
                    # print(i,j,q)
                    # print("alpha =", alpha_seq)
                    # print("manual_forward =", manual_forward)
                    print(alpha_seq[i,j,q])
                    self.assertAlmostEqual(alpha_seq[i,j,q], \
                                           manual_forward[q, j], \
                                           delta = 0.000001)

if __name__ == '__main__':
    unittest.main()
