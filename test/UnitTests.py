#!/usr/bin/env python3
import unittest
import numpy as np
import tensorflow as tf
from itertools import product


import sys
sys.path.insert(0, "../src")
sys.path.insert(0, "../scripts")
import Utility

class Test_A_prior(unittest.TestCase):
    def test_A_log_prior(self):
        from CGP_HMM import non_class_A_log_prior
        A = tf.constant([[0.2, 0.8, 0],\
                         [0.4, 0.4, 0.2], \
                         [0.1, 0,   0.9]])

        prior_matrix = tf.constant([[0.2, 0.8, 0],\
                                    [0.4, 0.4, 0.2], \
                                    [0,   0,   0]])


        prior_indices = tf.where(prior_matrix)

        # 0.34154892 = log ( a = p = 0.2, 0.8    *   a = p = 0.4, 0.4, 0.2 )
        self.assertAlmostEqual(non_class_A_log_prior(A, prior_matrix, prior_indices).numpy(), np.log(0.34154892), places = 5)

        prior_matrix *= 2
        self.assertAlmostEqual(non_class_A_log_prior(A, prior_matrix, prior_indices).numpy(), np.log(1.460929), places = 5)


    def test_B_log_prior(self):
        from CGP_HMM import non_class_B_log_prior
        # 3 states
        # 2 condition space size
        # 4 letter alphabet
        B = tf.constant(\
            [[0.2, 0.8, 0],\
             [0.4, 0.1, 0], \
             [0.1, 0,   0], \
             [0.3, 0.1, 1], \
\
             [0.1, 0,   0.5], \
             [0.2, 0,   0], \
             [0.3, 0,   0.5], \
             [0.4, 0,   0]])
        prior_matrix = tf.constant(\
            [[0.2, 0,   0],\
             [0.4, 0,   0], \
             [0.1, 0,   0], \
             [0.3, 0,   0], \
\
             [0.1, 0,   0.5], \
             [0.2, 0,   0], \
             [0.3, 0,   0.5], \
             [0.4, 0,   0]])

        prior_indices = tf.where(prior_matrix)

        self.assertAlmostEqual(non_class_B_log_prior(B, prior_matrix, prior_indices, 4).numpy(), np.log(0.101751395), places = 5)

class TestViterbi(unittest.TestCase):
    def test_Viterbi(self):
        state_space_size = 3
        emission_space_size = 4
        a = tf.nn.softmax(np.random.rand(state_space_size, state_space_size))
        b = tf.nn.softmax(np.random.rand(state_space_size, emission_space_size))
        a0 = tf.nn.softmax(np.random.rand(state_space_size))
        n = 6
        l = 5
        state_seq, emission_seq = Utility.generate_state_emission_seqs(a,b,n,l,a0)

        # always starting in state 0
        for i, y in enumerate(emission_seq):
            x = Utility.viterbi_log_version(a,b,y)
            x = Utility.brute_force_viterbi_log_version(a,b,y)
            self.assertTrue(all(Utility.viterbi_log_version(a,b,y) \
                                == \
                                Utility.brute_force_viterbi_log_version(a,b,y)))

        for i, y in enumerate(emission_seq):
            x = Utility.viterbi_log_version(a,b,y, a0)
            x = Utility.brute_force_viterbi_log_version(a,b,y, a0)
            self.assertTrue(all(Utility.viterbi_log_version(a,b,y, a0) \
                                == \
                                Utility.brute_force_viterbi_log_version(a,b,y, a0)))

class Test_matrix_indices(unittest.TestCase):
    pass
    # for now they are checked by hand

class Test_farward(unittest.TestCase):
    pass
    # pls use:
    # python3 cgphmm.py --fasta ... --AB dd --manual_forwad

class Test_B_prior_from_plb_files(unittest.TestCase):
    pass
    # todo

if __name__ == '__main__':
    unittest.main()
