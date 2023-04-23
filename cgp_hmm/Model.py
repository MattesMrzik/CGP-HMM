#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
class Model(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def make_model(self):
        pass

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

    def get_dense_A_and_B(self, A_weights = None, B_weights = None, A = None, B = None):
        assert A_weights is not None or A is not None, "export_to_dot_and_png(): weight for A or A must be passed"
        assert B_weights is not None or B is not None, "export_to_dot_and_png(): weight for B or B must be passed"

        if A is None:
            A = self.A(A_weights) if self.A_is_dense else tf.sparse.to_dense(self.A(A_weights))
        if B is None:
            B = self.B(B_weights) if self.B_is_dense else tf.sparse.to_dense(self.B(B_weights))

        return A, B

    # def get_most_likely_emission_seq_from_B(self, A_weights, B_weights, A, B):
    #     '''starting in initial state. what base has highest emission prob
    #     fix this base and look in next state, what base has highest emisson prob
    #     given the fixed base. repeat and return the base seq'''

    #     result = ""

    #     A, B = self.get_dense_A_and_B(A_weights, B_weights, A, B)

    #     for state_id in range(self.number_of_emissions):



################################################################################
    def export_to_dot_and_png(self, A_weights = None, \
                              B_weights = None, \
                              A = None, \
                              B = None, \
                              name = "graph", \
                              to_png = False, \
                              max_transition_prob = 0
    ):
        ''' A and B must be dense'''

        dir_path = f"{self.config.current_run_dir}/dot"

        if not os.path.exists(dir_path):
            os.system(f"mkdir -p {dir_path}")



        # TODO: add I parameters???
        # n_labels = self.number_of_emissions ** (self.config.order + 1)
        A, B = self.get_dense_A_and_B(A_weights, B_weights, A, B)


        B_reshaped = tf.reshape(B, shape = (-1, self.config.alphabet_size, self.number_of_states))
        B_argmax = np.argmax(B_reshaped, axis = 1)

        # id_to_base = {0:"A", 1:"C",2:"G",3:"T",4:"I",5:"Ter"}

        def write_file(with_emissions = True, name_appendix = ""):
            gv_path = f"{dir_path}/{name}_{name_appendix}.gv"
            png_path =f"{dir_path}/{name}_{name_appendix}.png"
            with open(gv_path, "w") as graph:
                graph.write("DiGraph G{\nrankdir=LR;\n")
                # graph.write("nodesep=0.5; splines=polyline;")
                for current_state_id, row in enumerate(A):
                    # TODO if current state is not reached by high prob, dont draw it

                    current_state_str = self.state_id_to_str(current_state_id)

                    # setting color for codons and inserts
                    try:
                        color = {"c_":"teal", "i_": "crimson"}[current_state_str[0:2]]
                    except:
                        color = "white"
                    graph.write("\"" + current_state_str + "\"\n") #  this was to_state before
                    if not with_emissions:
                        graph.write(f"[style=filled, fillcolor={color}]\n")

                    def write_emissions():
                        # writing current node with emissions
                        graph.write("[\n")
                        graph.write("\tshape = none\n")
                        graph.write("\tlabel = <<table border=\"0\" cellspacing=\"0\"> \n")

                        # drawing current state
                        graph.write(f"\t\t<tr><td port=\"port1\" border=\"1\" bgcolor=\"{color}\">" + current_state_str + "</td></tr>\n")

                        # for current state write emission
                        # for every NN write most likely emission:
                        for k, most_likely_index in enumerate(B_argmax[:,current_state_id]):
                            emission_id = most_likely_index + k * self.config.alphabet_size
                            emission_str = self.emission_id_to_str(emission_id)
                            emi_prob = str(np.round(B[emission_id, current_state_id].numpy(),4))
                            graph.write(f"\t\t<tr><td port=\"port{k+2}\" border=\"1\">({emission_str + ' ' +emi_prob})</td></tr>\n" )
                        graph.write("\t </table>>\n")
                        graph.write("]\n")

                    if with_emissions:
                        write_emissions()

                    # writing the transistions from the current state to the next ones
                    for next_state, prob in enumerate(row):
                        to_state_str = self.state_id_to_str(next_state)
                        if prob > max_transition_prob:
                            prob = prob.numpy()
                            graph.write(f"\"{current_state_str}\" -> \"{to_state_str}\" [label = {str(np.round(prob, 4))[:6]} fontsize=\"{30*prob + 5}pt\"]\n")

                graph.write("}")
            if to_png:
                command = f"dot -Tpng {gv_path} -o {png_path}"
                print(f"running: {command}")
                os.system(command)

        write_file()
        write_file(with_emissions = False, name_appendix="no_emission")

