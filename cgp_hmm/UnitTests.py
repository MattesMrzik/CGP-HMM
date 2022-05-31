#!/usr/bin/env python3
import unittest
import numpy as np
import tensorflow as tf

import Utility

class TestViterbi(unittest.TestCase):
    def test_Viterbi(self):# todo also test with passing a0
        state_space_size = 3
        emission_state_size = 4
        a = tf.nn.softmax(np.random.rand(state_space_size, state_space_size))
        b = tf.nn.softmax(np.random.rand(state_space_size, emission_state_size))
        a0 = tf.nn.softmax(np.random.rand(state_space_size))
        n = 50
        l = 5
        state_seq, emission_seq = Utility.generate_state_emission_seqs(a,b,n,l)
        # print("true", np.argmax( state_seq, axis = -1))

        for i, y in enumerate(emission_seq):
            # print(i, np.argmax(seq, axis = -1))
            y = np.argmax(y, axis = -1)
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
            y = np.argmax(y, axis = -1)
            x = Utility.viterbi_log_version(a,b,y, a0)
            print("x of viterbi =\t", x)
            print(Utility.P_of_X_Y(a,b,x,y,a0))
            x = Utility.brute_force_viterbi_log_version(a,b,y, a0)
            print("x of brute force =\t", x)
            print(Utility.P_of_X_Y(a,b,x,y,a0))
            self.assertTrue(all(Utility.viterbi_log_version(a,b,y, a0) \
                                == \
                                Utility.brute_force_viterbi_log_version(a,b,y, a0)))


class TestForward(unittest.TestCase):

    def test_forward(self):
        pass

if __name__ == '__main__':
    unittest.main()
