#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

def prRed(skk): print("Cell\033[93m {}\033[00m" .format(skk))
# def prRed(s): pass

class CgpHmmCell(tf.keras.layers.Layer):
    def __init__(self):
        super(CgpHmmCell, self).__init__()
        self.state_size = [2,1,1]
        self.alphabet_size = 4


    def build(self, input_shape):
        prRed("build of CgpHmmCell")
        self.transition_kernel = self.add_weight(shape = (self.state_size[0], self.state_size[0]),
                                              initializer="random_normal",
                                              trainable=True)
        self.emission_kernel = self.add_weight(shape = (self.state_size[0], self.alphabet_size),
                                              initializer="random_normal",
                                              trainable=True)
    @property
    def A(self):
        transition_matrix = tf.nn.softmax(self.transition_kernel, axis=-1, name="A")
        return transition_matrix

        # used these for manual run, cant use these for training, since the add_weight method isnt used
        # return np.array([0.1,  0.2,   0.3,  0.2,  0.2,\
        #                  0.2,  0.2,   0.2,  0.2,  0.2,\
        #                  0.2,  0.15,  0.15, 0.3 , 0.2,\
        #                  0.3,  0.2,   0.4,  0.0,  0.1,\
        #                  0,    0.2,   0.5,  0.3,  0.0], dtype = np.float32).reshape((5,5))


    @property
    def B(self):
        emission_matrix = tf.nn.softmax(self.emission_kernel, axis=-1, name="B")
        return emission_matrix

        # return np.array([0.1,  0.2,   0.3,  0.4 ,\
        #                  0.2,  0.15,  0.15, 0.5 ,\
        #                  0.3,  0.2,   0.5,  0   ,\
        #                  0,    0.2,   0.5,  0.3 ,\
        #                  0.25, 0.25, 0.25,  0.25], dtype = np.float32).reshape((5,4))

    # def get_initial_state(self):
    #     return [[1,0,0,0,0],[0,0,0,0,0]]

    def call(self, inputs, states, training = None, verbose = True):
        id = np.random.randint(100)
        old_forward, old_loglik, count = states

        if verbose:
            prRed("in call of CgpHmmCell")
            prRed("count " + str(count[0,0]))
            print("id =", id, "inputs = ", inputs)
            print("states ", states)
            print("A = ", self.A) # 5x5
            print("B = ", self.B) # 5x4


        count = count + 1 # counts i in alpha(q,i)
        alpha = 0
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


        # todo  where is the size of the cell state specified? output_size?

        # i think the second return argument is the cell state, which is used as input to next cell
        # the first argument is stored in return sequences

        return [alpha, inputs, count], [alpha, loglik, count] # todo warum soll hier die likelihood doppelt zurück gegeben werden?

        # wenn state_size = [5,2], dann muss nur da zweite argument die richtige shape haben, das erste scheint egal
        # return likelihood, [alpha, old_loglik]
