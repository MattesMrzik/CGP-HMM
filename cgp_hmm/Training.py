#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint
import os
import json
import re

import Utility
from Utility import run
from CallBacks import get_call_backs
from Utility import transform_verbose_txt_to_csv
from Utility import append_time_ram_stamp_to_file

from CgpHmmLayer import CgpHmmLayer
from ReadData import read_data_with_order
from tensorflow.python.client import device_lib

import time

def make_model(config):
    start = time.perf_counter()
    run_id = randint(0,100)

    # TODO: https://www.tensorflow.org/guide/keras/masking_and_padding

    append_time_ram_stamp_to_file(start, f"Traning.make_model() start {run_id}", config.bench_path)

    # another None added automatically for yet unkown batch_size
    sequences = tf.keras.Input(shape = (None, config.model.number_of_emissions), name = "sequences", dtype = config.dtype)

    cgp_hmm_layer = CgpHmmLayer(config) # init of layer

    loglik = cgp_hmm_layer(sequences) # layer is build, then called. it seems i cant call build before to avoid building it here again
    print(tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik))

    model = tf.keras.Model(inputs = sequences, outputs = [tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik)]) #  the output of the model is the value that is computed by a final layer that picks the loglike of the [alpha, loglik, count]

    append_time_ram_stamp_to_file(start, f"Traning.make_model() end   {run_id}", config.bench_path)
    return model, cgp_hmm_layer


def make_dataset(config):
    start = time.perf_counter()
    run_id = randint(0,100)
    append_time_ram_stamp_to_file(start, f"Training.make_dataset() start {run_id}", config.bench_path)

    from itertools import product
    codons = []
    for codon in product("ACGT", repeat = 3):
        codon = "".join(codon)
        if codon not in ["TAA", "TGA", "TAG"]:
            codons += [codon]

    if config.forced_gene_structure:
        codons = ["AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC"]
    alphabet = ["A","C","G","T"]

    if config.generate_new_seqs:
        if config.use_simple_seq_gen:
            num_seqs = 100
            seqs = {}
            with open(config.fasta_path, "w") as file:
                max_left_flank_len = (config.seq_len - config.gen_len -6)//2
                max_right_flank_len = config.seq_len - config.gen_len - 6 - max_left_flank_len

                min_left_flank_len = max_left_flank_len if config.dont_strip_flanks else 1
                min_right_flank_len = max_right_flank_len if config.dont_strip_flanks else 1

                for seq_id in range(num_seqs):
                    if config.forced_gene_structure:
                        alphabet = ["T","G"]
                    ig5 = "".join(np.random.choice(alphabet, np.random.randint(min_left_flank_len, max_left_flank_len +1))) # TODO: also check if low = 2
                    atg = "ATG"
                    # coding = "".join(np.random.choice(["A","C","G","T"], config["nCodons"] * 3))
                    coding = "".join(np.random.choice(codons, config.nCodons))
                    stop = np.random.choice(["TAA","TGA","TAG"])
                    ig3 = "".join(np.random.choice(alphabet, np.random.randint(min_right_flank_len, max_right_flank_len +1)))

                    seqs[f">use_simple_seq_gen_{seq_id}"] = ig5 + atg + coding + stop + ig3
                for key, value in seqs.items():
                    file.write(key + "\n")
                    file.write(value + "\n")


        else:
            command = f"python3 {config.src_path}/useMSAgen.py -c {config.nCodons} \
                          {'-n 100'} \
                          {'-l' + str(config.seq_len)} \
                          {'-cd ' + str(config.coding_dist) if config.coding_dist else ''} \
                          {'-ncd ' + str(config.noncoding_dist) if config.noncoding_dist else ''}\
                          {'--dont_strip_flanks' if config.dont_strip_flanks else ''} \
                          {'-p ' + config.src_path if config.src_path else ''}"
            command = re.sub("\s+", " ", command)
            run(command)


    seqs = read_data_with_order(config, add_one_terminal_symbol = True)


    ds = tf.data.Dataset.from_generator(lambda: seqs,
                                         tf.as_dtype(tf.int32), # has to be int, bc one_hot doesnt work for floats
                                         tf.TensorShape([None]))

    index_of_terminal = config.model.emission_tuple_to_id("X")
    ds = ds.padded_batch(config.batch_size, padding_values = index_of_terminal)

    def to_one_hot(seq):
        return tf.cast(tf.one_hot(seq, config.model.number_of_emissions), dtype=config.dtype)

    ds = ds.map(to_one_hot)
    # TODO: shuffle dataset?
    ds = ds.repeat()

    append_time_ram_stamp_to_file(start, f"Training.make_dataset() end   {run_id}", config.bench_path)
    return ds, seqs

# from memory_profiler import profile
# @profile
################################################################################
################################################################################
################################################################################
def fit_model(config):

    num_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
    print("Using", num_gpu, "GPUs. device_lib.list_local_devices()")

    num_gpu_logical = len(tf.config.experimental.list_logical_devices('GPU'))
    print("Using", num_gpu_logical, "GPUs.tf.config.experimental.list_logical_devices('GPU')")

    # setting the optimizer
    if config.optimizer == "Adam":
        if config.clip_gradient_by_value:
            optimizer = tf.optimizers.Adam(config.learning_rate, clipvalue = config.clip_gradient_by_value)
            config.optimizer = f"Adam with learning_rate {config.learning_rate} and clipvalue {config.clip_gradient_by_value}"
        else:
            optimizer = tf.optimizers.Adam(config.learning_rate)
            config.optimizer = f"Adam with learning_rate {config.learning_rate} and no clipping"
        print("config.optimizer =", config.optimizer)

    elif config.optimizer == "SGD":
        optimizer = tf.optimizers.SGD(config.learning_rate)
        config.optimizer = f"SGD with learning_rate {config.learning_rate}"
        print("config.optimizer =", config.optimizer)
    else:
        print("setting optimizer didnt work")
        exit(1)

    data_set, seqs = make_dataset(config)

    index_of_terminal = config.model.emission_tuple_to_id("X")

################################################################################
    if config.get_gradient_for_current_txt or config.get_gradient_from_saved_model_weights:

        # when accessing config["model"] in cell.call -> recursion error
        model, cgp_hmm_layer = make_model(config)
        model.compile(optimizer = optimizer)
        if config.get_gradient_from_saved_model_weights:
            model.load_weights(f"{config.src_path}/output/{config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/current_weights")
            # config["model"] = model
            # print('config["model"]', config["model"])
            config.weights = model.get_weights()

        layer = CgpHmmLayer(config)
        layer.build(None)
        layer.C.build(None)
        import ReadData
        # assuming that inputs are formatted in shape batch, seq_len, one_hot_dim = 32, l, 126
        input = ReadData.get_batch_input_from_tf_printed_file(f"{config.src_path}/output/{config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/current_inputs.txt")
        with tf.GradientTape() as tape:
            y, alpha_seq = layer(input) # eventuell wird hier die cell nochmal gebaut und das weight setzen davor bringt nichts

            dy_dx = tape.gradient(y,  [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])

            for g, name in zip(dy_dx, "IAB"):
                tf.print(f"gradient for {name}", g)
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(g)), [g], name = name, summarize = -1)


        exit()
################################################################################
    elif config.get_gradient_of_first_batch:
        layer = CgpHmmLayer(config)
        layer.build(None)
        layer.C.build(None)

        first_batch = seqs[:32] # not one hot, not padded
        # pad seqs:
        max_len_seq_in_batch = max([len(seq) for seq in first_batch])
        #                                                                              every seq has at least one terminal symbol
        first_batch = [seq + [index_of_terminal] * (max_len_seq_in_batch - len(seq) + 1) for seq in first_batch]

        # print("first_batch =", "\n".join([str(seq) for seq in first_batch]))

        # one_hot:
        first_batch = tf.one_hot(first_batch, n_emission_columns_in_B(config.alphabet_size, config.order)) # bc terminal symbol is not counted in emissions_state_size

        # print("first_batch =", "\n".join([str(seq) for seq in first_batch]))
        with tf.GradientTape() as tape:
            tape.watch([layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
            y = layer(first_batch)
            dy_dx = tape.gradient(y,  [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
            if not dy_dx:
                print("list dy_dx =", round(dy_dx,3))
            print()
            for g in dy_dx:
                print("dy_dx =", g)
                # print("dy_dx.numpy() =", g.numpy())
                print()
        exit()


################################################################################
    elif config.manual_forward:
            config.A_dense = True
            config.A_sparse = False
            config.B_dense = True
            config.B_sparse = False

            layer = CgpHmmLayer(config)


            layer.build(None)
            layer.C.build(None)
            cell = layer.C
            batchsize = config.batch_size
            low = 0
            n = len(seqs)
            high = min(batchsize, n)
            for epoch in range(config.epochs):
                for step in range(config.steps_per_epoch):
                    # print(f"low = {low}, high = {high}")
                    batch = seqs[low : high]

                    low = low + batchsize if low + batchsize < n else 0
                    if high == n:
                        high = min(batchsize, n)
                    elif high + batchsize > n:
                        high = n
                    else:
                        high += batchsize

                    max_len_seq_in_batch = max([len(seq) for seq in batch])
                    # print("max_len_seq_in_batch =", max_len_seq_in_batch)
                    batch = [seq + [index_of_terminal] * (max_len_seq_in_batch - len(seq) + 1) for seq in batch]
                    batch = tf.one_hot(batch, config.model.number_of_emissions)
                    # tf.print("batch = ", batch, summarize = -1)


                    # felix
                    alpha = tf.matmul(batch[:,0,:], cell.B) * cell.I
                    prod_zi =  1
                    loglike = tf.math.log(tf.reduce_sum(alpha, axis = 1))
                    for i in range(1, max_len_seq_in_batch + 1):
                        # E * R
                        z_i_minus_1 = tf.reduce_sum(alpha, axis = 1, keepdims = True)
                        # print("i =", i)
                        # print("z_i_minus_1 =", z_i_minus_1)
                        prod_zi *= z_i_minus_1
                        alpha =  tf.matmul(batch[:,i,:], cell.B) * tf.matmul(alpha, cell.A)
                        # print("alpha =", alpha)
                        alpha = tf.math.divide(alpha, z_i_minus_1)
                        loglike += tf.math.log(tf.reduce_sum(alpha, axis = 1))
                    # print("alpha =", alpha)
                    print("\n=========> felix <===========")
                    print("mean(loglike += log(sum_q(alpha))", tf.reduce_mean(loglike))
                    print("mean(log(sum_q(alpha * prod_zi)))", tf.reduce_mean(tf.math.log(tf.reduce_sum(alpha * prod_zi, axis=1))))
                    # is there another way to get P(Y)

                    # only one reduce sum
                    alpha = tf.matmul(batch[:,0,:], cell.B) * cell.I
                    z_0 = tf.reduce_sum(alpha, axis =1, keepdims = True)
                    loglike = tf.math.log(tf.reduce_sum(alpha, axis = 1))
                    alpha = tf.math.divide(alpha, z_0)
                    prod_zi = z_0
                    for i in range(1, max_len_seq_in_batch + 1):
                        # E * R
                        alpha =  tf.matmul(batch[:,i,:], cell.B) * tf.matmul(alpha, cell.A)
                        loglike += tf.math.log(tf.reduce_sum(alpha, axis = 1))

                        z_i = tf.reduce_sum(alpha, axis = 1, keepdims = True)
                        prod_zi *= z_i
                        alpha = tf.math.divide(alpha, z_i)

                    print("\n=========> mattes <===========")
                    print("mean(loglike += log(sum_q(alpha)) + log(sum_q(alpha))", tf.reduce_mean(loglike + tf.math.log(tf.reduce_sum(alpha, axis = 1))))
                    print("mean(log(sum_q(alpha * prod_zi)))", tf.reduce_mean(tf.math.log(tf.reduce_sum(alpha * prod_zi, axis=1))))


                    # manual_forward
                    alpha = tf.matmul(batch[:,0,:], cell.B) * cell.I
                    for i in range(1, max_len_seq_in_batch + 1):
                        # E * R
                        alpha =  tf.matmul(batch[:,i,:], cell.B) * tf.matmul(alpha, cell.A)

                    loglike = tf.math.log(tf.reduce_sum(alpha, axis=1))
                    mean_loglike = tf.reduce_mean(loglike)

                    print("\n=========> alpha dp <===========")
                    print("mean(log(sum_q(alpha)))=", mean_loglike)
                    print()

            exit()

    elif config.manual_training_loop:
        assert self.conifg.return_seqs, "if manual training loop then return seqs must be true"
        #  get the very first init weights of a run that resulted in nan
        # maybe this is not necessary, since i can run with --dont_generate_new_seqs flag,
        # and even though the kernels are always initialized differently nan always occur
        print("config.alpha_i_gradient =", config.alpha_i_gradient)

        layer = CgpHmmLayer(config)
        layer.build(None)
        layer.C.build(None)

        batchsize = 32
        low = 0
        n = len(seqs)
        high = min(batchsize, n)
        for epoch in range(config.epochs):
            for step in range(config.steps_per_epoch):
                # print(f"low = {low}, high = {high}")
                batch = seqs[low : high]

                low = low + batchsize if low + batchsize < n else 0
                if high == n:
                    high = min(batchsize, n)
                elif high + batchsize > n:
                    high = n
                else:
                    high += batchsize

                max_len_seq_in_batch = max([len(seq) for seq in batch])
                # print("max_len_seq_in_batch =", max_len_seq_in_batch)
                batch = [seq + [index_of_terminal] * (max_len_seq_in_batch - len(seq) + 1) for seq in batch]
                batch = tf.one_hot(batch, n_emission_columns_in_B(config.alphabet_size, config.order))
                # tf.print("batch = ", batch, summarize = -1)

                with tf.GradientTape() as tape:

                    tape.watch([layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
                    y, alpha_seq = layer(batch)
                    # alpha_seq [batch, i, q]
                    # print("tf.shape(alpha_seq) =", tf.shape(alpha_seq))

                    print("start of calculation of loglikes from alpha")
                    loglikes = [tf.math.log(tf.math.reduce_sum(alpha_seq[:,0,:], axis = -1, keepdims = True))]
                    for i in range(1, tf.shape(alpha_seq)[1]):
                        loglikes.append(loglikes[-1] + tf.math.log(tf.math.reduce_sum(alpha_seq[:,i,:], axis = -1, keepdims = True)))
                    print("end  of calculation of loglikes from alpha")

                    # print("y =", y) # this has shape batch_size
                    # print("this should be the same as:")
                    # print("loglikes[-1] =", loglikes[-1])
                    # print("and it is indeed")

                    # TODO: now i can differentiate the first loglike wrt the kernels
                    # lets hope that not the hole graph is backpropagated for this differential
                    # then for long seqs where the normal backprobs does some nan
                    # then i can find the last index i for which the gradient exists

                    # TODO: this y is not yet reduced mean
                    # what criterium is used when y is vector and not scalar?
                    gradient_summarize = 10
                    print(f"epoch({epoch}), step({step}) the loss is: {tf.math.reduce_mean(y)}")

                    if config.alpha_i_gradient == -1: # ie. differentiate y
                        gradient = tape.gradient(-1*tf.math.reduce_mean(y),  [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
                    elif config.alpha_i_gradient == -2: # ie. differentiate alpha(n-1), this should be the same as when y is differentiated
                        gradient = tape.gradient(-1*tf.math.reduce_mean(loglikes[-1]),  [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
                    else:
                        gradient =  tape.gradient(-1*tf.math.reduce_mean(loglikes[config.alpha_i_gradient]),  [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
                    # tf.print("gradient =", gradient, summarize = gradient_summarize)


                    ik = [float(x) for x in gradient[0]]
                    ak = [float(x) for x in gradient[1]]
                    bk = [float(x) for x in gradient[2]]

                    with open(f"{config.src_path}/output/{config.nCodons}codons/gradient_ik_for_alpha{config.alpha_i_gradient}.txt","w") as file:
                        json.dump(ik, file)
                    with open(f"{config.src_path}/output/{config.nCodons}codons/gradient_ak_for_alpha{config.alpha_i_gradient}.txt","w") as file:
                        json.dump(ak, file)
                    with open(f"{config.src_path}/output/{config.nCodons}codons/gradient_bk_for_alpha{config.alpha_i_gradient}.txt","w") as file:
                        json.dump(bk, file)

                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[0])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_I", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[1])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_A", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[2])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_B", summarize = config.assert_summarize)
                    gradient[0] = config.learning_rate * gradient[0]
                    gradient[1] = config.learning_rate * gradient[1]
                    gradient[2] = config.learning_rate * gradient[2]
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[0])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_I_after_learning_rate_scaled", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[1])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_A_after_learning_rate_scaled", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[2])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_B_after_learning_rate_scaled", summarize = config.assert_summarize)
                    # print("layer.C.init_kernel before apply grad=", layer.C.init_kernel)
                    optimizer.apply_gradients(zip(gradient, [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel]))
                    # print("layer.C.init_kernel after apply grad =", layer.C.init_kernel)

                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.init_kernel)),       [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel], name = "I_kernel_after_apply_grads", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.transition_kernel)), [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel], name = "A_kernel_after_apply_grads", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.emission_kernel)),   [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel], name = "B_kernel_after_apply_grads", summarize = config.assert_summarize)
        exit()


################################################################################
    else:
        if num_gpu > 1 and not config.dont_use_mirrored_strategy:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model, cgp_hmm_layer = make_model(config)
                model.summary()

                # compile model
                start = time.perf_counter()
                run_id = randint(0,100)
                append_time_ram_stamp_to_file(start, f"Training:model.compile() start {run_id}", config.bench_path)
                model.compile(optimizer = optimizer, run_eagerly = config.run_eagerly)
                append_time_ram_stamp_to_file(start, f"Training:model.compile() end   {run_id}", config.bench_path)

                # fit model
                start = time.perf_counter()
                run_id = randint(0,100)
                append_time_ram_stamp_to_file(start, f"Training:model.fit() start {run_id}", config.bench_path)
                history = model.fit(data_set, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
                append_time_ram_stamp_to_file(start, f"Training:model.fit() end   {run_id}", config.bench_path)
        else:
             model, cgp_hmm_layer = make_model(config)
             model.summary()

             # compile model
             start = time.perf_counter()
             run_id = randint(0,100)
             append_time_ram_stamp_to_file(start, f"Training:model.compile() start {run_id}", config.bench_path)
             model.compile(optimizer = optimizer, run_eagerly = config.run_eagerly)
             append_time_ram_stamp_to_file(start, f"Training:model.compile() end   {run_id}", config.bench_path)

             # git model
             start = time.perf_counter()
             run_id = randint(0,100)
             append_time_ram_stamp_to_file(start, f"Training:model.fit() start {run_id}", config.bench_path)
             print("optimizer.iterations should be 0:", optimizer.iterations)
             history = model.fit(data_set, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
             print("optimizer.iterations should be larger 0:", optimizer.iterations)
             append_time_ram_stamp_to_file(start, f"Training:model.fit() end   {run_id}", config.bench_path)

    return model, history
