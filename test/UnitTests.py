#!/usr/bin/env python3
import unittest
import numpy as np
import tensorflow as tf
from itertools import product
import os
import re



import argparse

parser = argparse.ArgumentParser(
    description='description')
parser.add_argument('-f', action='store_true', help ="test forward algo")
parser.add_argument('-v', action='store_true', help ="test python viterbi algo") # todo: also test cc viterbi

args = parser.parse_args()


class Test_A_prior(unittest.TestCase):

    # import tensorflow_probability as tfp
    # tfd = tfp.distributions
    # def prob(a,p = None):
    #     if p is None:
    #         p = a
    #     dist = tfd.Dirichlet(tf.constant(a))
    #     return dist.prob(p)


    def test_A_log_prior(self):
        from My_internal_exon_model import non_class_A_log_prior
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

        # TODO maybe add for loop over scale prior

    def test_B_log_prior(self):
        from My_internal_exon_model import non_class_B_log_prior
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

class TestUtiliy(unittest.TestCase):
    def off_test_state_id_description_conversion(self):
        nCodons = 2
        manual = "ig5' stA stT stG c_0,0 c_0,1 c_0,2 c_1,0 c_1,1 c_1,2 stop1 stop2 stop3 ig3' i_0,0 i_0,1 i_0,2 i_1,0 i_1,1 i_1,2 i_2,0 i_2,1 i_2,2 ter1 ter2".split(" ")
        state_id_description_list = Utility.get_state_id_description_list(nCodons)
        for i, des in enumerate(manual):
            self.assertEqual(des, state_id_description_list[i])
        for i, des in enumerate(manual):
            self.assertEqual(des, Utility.state_id_to_description(i, nCodons))
        for i, des in enumerate(manual):
            self.assertEqual(des, Utility.state_id_to_description(i, nCodons, state_id_description_list))

        for i, des in enumerate(manual):
            self.assertEqual(i, state_id_description_list.index(des))
        for i, des in enumerate(manual):
            self.assertEqual(i, Utility.description_to_state_id(des, nCodons))
        for i, des in enumerate(manual):
            self.assertEqual(i, Utility.description_to_state_id(des, nCodons, state_id_description_list))


    # 4 is alphabet_size
    def off_test_higher_order_emission_to_id(self):
        self.assertEqual(higher_order_emission_to_id([0,0,0], 4, 2), 0)
        self.assertEqual(higher_order_emission_to_id([0,0,1], 4, 2), 1)
        self.assertEqual(higher_order_emission_to_id([0,2,3], 4, 2), 13)
        self.assertEqual(higher_order_emission_to_id([2,2,2], 4, 2), 62)
        self.assertEqual(higher_order_emission_to_id([2,2,2], 4, 2), 62)
        self.assertEqual(higher_order_emission_to_id("X", 4, 2), 125)
        self.assertEqual(higher_order_emission_to_id(5, 4, 2), 125)
        self.assertEqual(higher_order_emission_to_id([5], 4, 2), 125)

    def off_test_id_to_higher_order_emission(self):
        self.assertEqual(id_to_higher_order_emission(0,4,2),[0,0,0])
        self.assertEqual(id_to_higher_order_emission(1,4,2),[0,0,1])
        self.assertEqual(id_to_higher_order_emission(13,4,2),[0,2,3])
        self.assertEqual(id_to_higher_order_emission(62,4,2),[2,2,2])
        self.assertEqual(id_to_higher_order_emission(125,4,2),[5])

class TestCgpHmmCell(unittest.TestCase):

    def off_test_get_indices_for_weights_for_A(self):
        local_config = config.copy()
        local_config["nCodons"] = 100
        cell = CgpHmmCell(local_config)
        indices_for_weights_A = cell.get_indices_for_weights_for_A()
        print(indices_for_weights_A)


    def off_test_state_is_third_pos_in_frame(self):

        local_config = config.copy()
        local_config["nCodons"] = 2
        Utility.get_indices_for_config(local_config)
        cell = CgpHmmCell(local_config) # for 2 codons

        for i in range(cell.number_of_states):
            if i in [6,9,16,19,22]:
                self.assertTrue(cell.state_is_third_pos_in_frame(i))
            else:
                self.assertFalse(cell.state_is_third_pos_in_frame(i))


    def off_test_emission_is_stop_codon(self):
        cell = CgpHmmCell(config)# when writing this test, order was always set to 2
        self.assertFalse(cell.emission_is_stop_codon([4,4,4]))
        self.assertFalse(cell.emission_is_stop_codon([3,0,1]))
        self.assertFalse(cell.emission_is_stop_codon([0,0,0]))

        self.assertTrue(cell.emission_is_stop_codon([3,0,0]))
        self.assertTrue(cell.emission_is_stop_codon([3,0,2]))
        self.assertTrue(cell.emission_is_stop_codon([3,2,0]))

    def off_test_get_emissions_that_fit_ambiguity_mask(self):
        cell = CgpHmmCell(config)# when writing this test, order was always set to 2
        d = dict(zip("ACGTI", range(5)))

        def remove_stops_and_initial_after_base(allowed_ho_emissions, state):
            purged_allowed_ho_emissions = []
            for ho_emission in allowed_ho_emissions:
                if cell.emission_is_stop_codon(ho_emission) and cell.state_is_third_pos_in_frame(state):
                    continue
                if cell.has_I_emission_after_base(ho_emission):
                    continue
                purged_allowed_ho_emissions += [ho_emission]
            return purged_allowed_ho_emissions

        mask = "A"
        state = 0
        allowed_ho_emissions = "ACGT ACGT A"
        allowed_ho_emissions = [tuple(map(lambda y:d[y], ho_emission)) for ho_emission in product(*[list(x) for x in allowed_ho_emissions.split(" ")])]
        x_bases_must_preceed = 2
        self.assertEqual(cell.get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state), allowed_ho_emissions)

        mask = "ANT"
        state = 0
        allowed_ho_emissions = "A ACGT T"
        allowed_ho_emissions = [tuple(map(lambda y:d[y], ho_emission)) for ho_emission in product(*[list(x) for x in allowed_ho_emissions.split(" ")])]
        x_bases_must_preceed = 2
        self.assertEqual(cell.get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state), allowed_ho_emissions)

        mask = "AN"
        state = 0
        allowed_ho_emissions = "ACGTI A ACGT"
        allowed_ho_emissions = [tuple(map(lambda y:d[y], ho_emission)) for ho_emission in product(*[list(x) for x in allowed_ho_emissions.split(" ")])]
        x_bases_must_preceed = 1
        self.assertEqual(cell.get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state), allowed_ho_emissions)

        mask = "ATG"
        state = 0
        allowed_ho_emissions = "AI IT G"
        allowed_ho_emissions = [tuple(map(lambda y:d[y], ho_emission)) for ho_emission in product(*[list(x) for x in allowed_ho_emissions.split(" ")])]
        x_bases_must_preceed = 0
        allowed_ho_emissions = sorted(remove_stops_and_initial_after_base(allowed_ho_emissions, state))
        self.assertEqual(sorted(cell.get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state)), allowed_ho_emissions)

        mask = "ATN"
        state = 0
        allowed_ho_emissions = "AI TI ACGT"
        allowed_ho_emissions = [tuple(map(lambda y:d[y], ho_emission)) for ho_emission in product(*[list(x) for x in allowed_ho_emissions.split(" ")])]
        x_bases_must_preceed = 0
        allowed_ho_emissions = sorted(remove_stops_and_initial_after_base(allowed_ho_emissions, state))
        self.assertEqual(sorted(cell.get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state)), allowed_ho_emissions)

        mask = "N"
        state = 6
        allowed_ho_emissions = "ACGTI ACGTI ACGT"
        allowed_ho_emissions = [tuple(map(lambda y:d[y], ho_emission)) for ho_emission in product(*[list(x) for x in allowed_ho_emissions.split(" ")])]
        x_bases_must_preceed = 0
        allowed_ho_emissions = sorted(remove_stops_and_initial_after_base(allowed_ho_emissions, state))
        self.assertEqual(sorted(cell.get_emissions_that_fit_ambiguity_mask(mask, x_bases_must_preceed, state)), allowed_ho_emissions)

    def off_test_has_I_emission_after_base(self):
        cell = CgpHmmCell(config)# when writing this test, order was always set to 2
        self.assertFalse(cell.has_I_emission_after_base([4,4,4]))
        self.assertFalse(cell.has_I_emission_after_base([4,4,2]))
        self.assertFalse(cell.has_I_emission_after_base([4,1,2]))
        self.assertFalse(cell.has_I_emission_after_base([0,1,2]))

        self.assertTrue(cell.has_I_emission_after_base([4,2,4]))
        self.assertTrue(cell.has_I_emission_after_base([1,4,4]))
        self.assertTrue(cell.has_I_emission_after_base([0,2,4]))
        self.assertTrue(cell.has_I_emission_after_base([3,4,3]))

    def off_test_strip_or_pad_emission_with_n(self):
        cell = CgpHmmCell(config)# when writing this test, order was always set to 2
        self.assertEqual(cell.strip_or_pad_emission_with_n("A"), list("NNA"))
        self.assertEqual(cell.strip_or_pad_emission_with_n("AC"), list("NAC"))
        self.assertEqual(cell.strip_or_pad_emission_with_n("AAA"), list("AAA"))
        self.assertEqual(cell.strip_or_pad_emission_with_n("AAAA"), list("AAA"))

    # these test just print stuff

    def off_test_get_indices_and_values_for_transition_kernel(self):
        cell = CgpHmmCell(2)
        #                                                                                        weights, 2 codons
        indices, values = cell.get_indices_and_values_for_transition_kernel(np.array(list(range(10,100))),cell.nCodons)
        print("indices =", indices)
        print("values =", values)
        transition_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [cell.state_size[0]] * 2)
        transition_matrix = tf.sparse.reorder(transition_matrix)
        print(tf.sparse.to_dense(transition_matrix))

    def off_test_get_indices_and_values_from_emission_kernel(self):
        cell = CgpHmmCell()
        #                                                                                  100 weights,     2 codons, 4 = alphabet_size
        indices, values = cell.get_indices_and_values_from_emission_kernel(np.array(list(range(10,100))),cell.nCodons,4)
        print("indices =", len(indices))
        print("values =", len(values))
        emission_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [cell.state_size[0],5])
        emission_matrix = tf.sparse.reorder(emission_matrix)
        print(tf.sparse.to_dense(emission_matrix))

    def off_test_get_indices_and_values_for_transition_kernel_higher_order(self):
        cell = CgpHmmCell(4)# todo add arguments
        #                                                                                        weights, 2 codons
        indices, values = cell.get_indices_and_values_for_transition_kernel_higher_order(np.array(list(range(10,100))),cell.nCodons)
        transition_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [cell.state_size[0]] * 2)
        transition_matrix = tf.sparse.reorder(transition_matrix)
        transition_matrix = tf.sparse.to_dense(transition_matrix)

        print("-\t", end = "")
        for i in range(len(transition_matrix)):
            print(Utility.state_id_to_description(i, cell.nCodons), end = "\t")
        print()
        for i in range(len(transition_matrix)):
            print(Utility.state_id_to_description(i, cell.nCodons), end = "\t")
            for j in range(len(transition_matrix[i])):
                if i == j:
                    print("\033[92m", transition_matrix[i,j].numpy(),"\033[0m", sep = "", end = "\t")
                else:
                    print(transition_matrix[i,j].numpy(), end = "\t")
            print()

    def off_test_get_indices_and_values_from_emission_kernel(self):
        cell = CgpHmmCell(2)
        #                                                                                  100 weights,     2 codons, 4 = alphabet_size
        indices, values = cell.get_indices_and_values_from_emission_kernel(np.array(list(range(10,10000))),cell.nCodons,4)
        print("indices =", len(indices))
        print("values =", len(values))
        emission_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [cell.state_size[0],6,6,6])
        emission_matrix = tf.sparse.reorder(emission_matrix)
        emission_matrix = tf.sparse.to_dense(emission_matrix)
        for state in range(len(emission_matrix)):
            tf.print(state_id_to_description(state, cell.nCodons))
            tf.print(emission_matrix[state], summarize = -1)
            tf.print("---------------------------------------------")

    def off_test_get_indices_and_values_for_emission_and_state(self):
        cell = CgpHmmCell(2)
        indices = []
        values = [[]]
        weights = list(range(100))
        k = [0]
        # cell.get_indices_and_values_for_emission_and_state(0,1,indices,3,4,"ACX",0)
        cell.get_indices_and_values_for_emission_and_state(weights,k,indices,values,0,"AAAT",1)
        cell.get_indices_and_values_for_emission_and_state(weights,k,indices,values,1,"AATG",1, trainable = False)
        cell.get_indices_and_values_for_emission_and_state(weights,k,indices,values,2,"AAAT",2)
        cell.get_indices_and_values_for_emission_and_state(weights,k,indices,values,3,"AATG",1, trainable = False)
        cell.get_indices_and_values_for_emission_and_state(weights,k,indices,values,4,"N",0)
        for i, (index,value) in enumerate(zip(indices, values[0])):
            print("i = ", i, index, value)

    def off_test_get_indices_and_values_from_emission_kernel_v02(self):
        cell = CgpHmmCell(2)
        #                                                                                  100 weights,     2 codons, 4 = alphabet_size
        indices, values = cell.get_indices_and_values_from_emission_kernel_v02(np.array(list(range(10,10000)),dtype = tf.float32),cell.nCodons,4)
        print("indices =", len(indices))
        print("values =", len(values))
        emission_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [cell.state_size[0],6,6,6])
        emission_matrix = tf.sparse.reorder(emission_matrix)
        emission_matrix = tf.sparse.to_dense(emission_matrix)
        for state in range(len(emission_matrix)):
            tf.print(state_id_to_description(state, cell.nCodons))
            tf.print(emission_matrix[state], summarize = -1)
            tf.print("---------------------------------------------")

    def off_test_get_indices_and_values_from_initial_kernel(self):
        cell = CgpHmmCell(2)
        indices, values = cell.get_indices_and_values_from_initial_kernel(np.array(list(range(100))), cell.nCodons)
        print("indices =", indices)
        print("values =", values)
        initial_matrix = tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = [cell.state_size[0],1])
        initial_matrix = tf.sparse.reorder(initial_matrix)
        initial_matrix = tf.sparse.to_dense(initial_matrix)
        print(initial_matrix)

class TestViterbi(unittest.TestCase):
    def off_test_Viterbi(self):
        if args.v:
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
    # todo: test also if X can be reached earlier

    def off_test_tf_scaled_forward_to_manual_scaled_forward(self):
        if args.f:
            import ReadData
            import WriteData
            from Utility import run
            import json

            local_config = config.copy()
            local_config["nCodons"] = 2
            # see layer.py
            local_config["write_return_sequnces"] = True # doesnt write A, B, E, R, alpha, ..., but only the return_seqs if rnn
            Utility.get_indices_for_config(local_config)

            use_batch_begin_write_weights__layer_call_write_inputs = True
            if not use_batch_begin_write_weights__layer_call_write_inputs:
                # >seq1
                # ACATGCAAGGTTAATTG
                # >seq2
                # CACATGCAAGGTTAAT
                # >seq3
                # ACATGCAAGGTTA
                input_seqs = ["ACATGCAAGGTTAATTG", "CACATGCAAGGTTAAT", "ACATGCAAGGTTA"]
                input_seqs = ["ACATGCAAGGTTAATTG", "CCCATGGTACGCTAAG", "AGATGCCCTGGTAGA"]
            if use_batch_begin_write_weights__layer_call_write_inputs:


                # or get seqs from batch_begin_write_weights__layer_call_write_inputs
                input_seqs = ReadData.get_batch_input_from_tf_printed_file(f"{local_config['out_path']}/output/{local_config['nCodons']}codons/batch_begin_write_weights__layer_call_write_inputs/current_inputs.txt")
                input_seqs = [[Utility.id_to_higher_order_emission(np.argmax(one_hot_emission), local_config["alphabet_size"], local_config["order"]) [-1] \
                              for one_hot_emission in seq] for seq in input_seqs]
                input_seqs = ["".join(["ACGT"[base] if base != 5 else "" for base in seq]) for seq in input_seqs]

                # TODO: set weights of cell from current files
                # TODO: not only compare the manual run to the fit + return seqs in unit test, but also to
                # the alphas that can be obtained from verbose from the main_programm run which created also the current.txt files
            # print(input_seqs)
            max_len = max([len(seq) for seq in input_seqs])

            os.system("mkdir -p output/for_unit_tests")

            Utility.print_config(local_config)

            # bc i only use the first seq in manual forward,
            # i have to shuffle the data seq, such that every seq is at the first position at some point
            for i in range(len(input_seqs)):
                local_config["fasta_path"] = f"output/for_unit_tests/{i}_out.seqs.2codons.fa"
                # if i != 1:
                #     continue
                shuffeld = [input_seqs[(j+i)%len(input_seqs)] for j in range(len(input_seqs))]

                with open(local_config["fasta_path"], "w") as file:
                    for j, s in enumerate(shuffeld):
                        file.write(f">seq{j}\n")
                        file.write(s)
                        file.write("\n")
                file.close()

                # run(f"cat {local_config['fasta_path']}")

                os.system("rm ./output/for_unit_tests/return_sequnces.txt")

                model, cgp_hmm_layer = Training.make_model(local_config)
                learning_rate = .1

                optimizer = tf.optimizers.Adam(learning_rate)

                model.compile(optimizer = optimizer)

                data_set, seqs = Training.make_dataset(local_config)


                cell = CgpHmmCell(local_config)
                cell.init_kernel = model.get_weights()[0]
                cell.transition_kernel = model.get_weights()[1]
                cell.emission_kernel = model.get_weights()[2]

                print_shapes = False
                if print_shapes:
                    print("cell.init_kernel       =", tf.shape(cell.init_kernel))
                    print("cell.transition_kernel =", tf.shape(cell.transition_kernel))
                    print("cell.emission_kernel   =", tf.shape(cell.emission_kernel))
                    print("indices_for_I          =", tf.shape(local_config["indices_for_I"]))
                    print("indices_for_weights_A  =", tf.shape(local_config["indices_for_weights_A"]))
                    print("indices_for_weights_B  =",tf.shape(local_config["indices_for_weights_B"]))

                # get weights must be before this here, since in fit the weights are change in the end
                history = model.fit(data_set, epochs=1, steps_per_epoch = 1)

                alpha, z = Utility.forward_felix_version(cell.A_dense, \
                                                         cell.B_dense, \
                                                         seqs[0] \
                                                         + [Utility.higher_order_emission_to_id("X", local_config["alphabet_size"], local_config["order"])] \
                                                         * (max_len - len(seqs[0])), \
                                                         a0 = cell.I_dense)


                def write_info():
                    os.system("rm ./output/for_unit_tests/manual_forward.txt")
                    outstream = f"file://./output/for_unit_tests/manual_forward.txt"
                    tf.print(alpha, summarize = -1, output_stream = outstream)

                    WriteData.write_order_transformed_B_to_csv(cell.B_dense, f"output/for_unit_tests/B.csv", local_config["order"], local_config["nCodons"])
                    WriteData.write_to_file(cell.A_dense, f"output/for_unit_tests/A.txt")
                    WriteData.write_to_file(tf.transpose(cell.B_dense), f"output/for_unit_tests/B.txt")
                    WriteData.write_to_file(cell.I_dense, f"output/for_unit_tests/I.txt")

                    with open("./output/for_unit_tests/seq.txt","w") as file:
                        file.write(','.join([str(id) for id in seqs[0]]))
                        file.write("\n")

                # this file is written in layer when "write_return_sequnces" is True in config
                with open("output/for_unit_tests/return_sequnces.txt", "r") as file:
                    for j, line in enumerate(file):
                        line = line.split(";")[2]
                        line = line[1:-2].split(" ")
                        line = list(map(float, line))
                        for k, entry in enumerate(line):
                            places = 6
                            epsilon = float(f"1e-{places}")
                            if abs(entry - alpha[k,j]) > epsilon:
                                print("i =", j, ", state =", k)
                                print(f"tf = {entry} != {alpha[k,j]} hand")
                            #                      tf     hand
                            self.assertAlmostEqual(entry, alpha[k,j], places = places)

                print("-----------------------------------------------------------")


    # manual forward <-- using the z of manual --> manual scaled forward
    # def test_manual_scaled_forward_to_manual_true_forward(self):
    #     pass
    # # tf scaled forward <-- using the z of tf --> manual forward
    # def test_tf_scaled_transformed_forward_to_manual_true_forward(self):
    #     pass
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
