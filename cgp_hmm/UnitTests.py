#!/usr/bin/env python3
import unittest
import numpy as np
import tensorflow as tf
import Training
from CgpHmmLayer import CgpHmmLayer

import Utility

class TestViterbi(unittest.TestCase):
    def atest_Viterbi(self):# todo also test with passing a0
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
    def atest_P_of_X_Y(self):
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
            self.assertAlmostEqual(Utility.P_of_X_Y(a,b,x,y), np.exp(Utility.P_of_X_Y_log_version(a,b,x,y)), delta = 0.0000001)

        for i, (x,y) in enumerate(zip(state_seq, emission_seq)):
            # print(Utility.P_of_X_Y(a,b,x,y,a0), " =?= ", np.exp(Utility.P_of_X_Y_log_version(a,b,x,y,a0)))
            self.assertAlmostEqual(Utility.P_of_X_Y(a,b,x,y,a0), np.exp(Utility.P_of_X_Y_log_version(a,b,x,y,a0)), delta = 0.0000001)

class TestForward(unittest.TestCase):

    def test_forward(self):

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

        alpha_seq, inputs_seq, count_seq,  alpha_state, loglik_state, count_state = cgp_hmm_layer.F(tf.cast(emission_seq, dtype = tf.float32))# cast input to in or float

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
                    self.assertAlmostEqual(alpha_seq[i,j,q], manual_forward[q, j], delta = 0.000001)

if __name__ == '__main__':
    unittest.main()
