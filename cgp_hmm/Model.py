#!/usr/bin/env python3
import json
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
class Model(ABC):

    def __init__(self, config):
        self.config = config

    # kernel sizes
    @abstractmethod
    def I_kernel_size(self):
        pass

    def A_kernel_size(self):
        pass

    def B_kernel_size(self):
        pass

    # matrices
    @abstractmethod
    def I(self, weights):
        pass

    @abstractmethod
    def A(self, weights):
        pass

    @abstractmethod
    def B(self, weights):
        pass



    @abstractmethod
    def get_number_of_states():
        pass

    @abstractmethod
    def get_number_of_emissions():
        pass

    @abstractmethod
    def state_id_to_str():
        pass

    @abstractmethod
    def str_to_state_id():
        pass

    @abstractmethod
    def emission_id_to_str():
        pass

    @abstractmethod
    def str_to_emission_id():
        pass

    def write_model(self):
        pass
    def read_model(self):
        pass
    def find_indices_of_zeros():
        pass
################################################################################
################################################################################
################################################################################
    def export_to_dot_and_png(self, A_weights, B_weights, out_path = "this is still hard coded"):
        # TODO: add I parameters???
        n_labels = self.number_of_emissions ** (self.config.order + 1)
        nCodons = self.config.nCodons

        A = self.A(A_weights) if self.A_is_dense else tf.sparse.to_dense(self.A(A_weights))
        B = self.B(B_weights) if self.B_is_dense else tf.sparse.to_dense(self.B(B_weights))

        B_reshaped = tf.reshape(B, shape = (-1, self.config.alphabet_size, self.number_of_states))
        B_argmax = np.argmax(B_reshaped, axis = 1)

        id_to_base = {0:"A", 1:"C",2:"G",3:"T",4:"I",5:"Ter"}
        with open(f"{self.config.out_path}/output/{nCodons}codons/graph.gv", "w") as graph:
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
        run(f"dot -Tpng {self.config.out_path}/output/{nCodons}codons/graph.gv -o output/{nCodons}codons/graph.png")
