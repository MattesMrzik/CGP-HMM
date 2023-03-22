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
from Utility import append_time_ram_stamp_to_file

from CgpHmmLayer import CgpHmmLayer
from ReadData import read_data_with_order
# from ReadData import read_data_one_hot_with_Ns_spread
from ReadData import read_data_one_hot_with_Ns_spread_str
from ReadData import convert_data_one_hot_with_Ns_spread_str_to_numbers
from tensorflow.python.client import device_lib

import time

def make_model(config):
    start = time.perf_counter()
    run_id = randint(0,100)

    # TODO: https://www.tensorflow.org/guide/keras/masking_and_padding

    append_time_ram_stamp_to_file(f"Traning.make_model() start {run_id}", config.bench_path, start)

    # another None added automatically for yet unkown batch_size
    sequences = tf.keras.Input(shape = (None, config.model.number_of_emissions), name = "sequences", dtype = config.dtype)

    cgp_hmm_layer = CgpHmmLayer(config) # init of layer



    loglik = cgp_hmm_layer(sequences) # layer is build, then called. it seems i cant call build before to avoid building it here again

    # print(tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik))

    model = tf.keras.Model(inputs = sequences, outputs = [tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik)]) #  the output of the model is the value that is computed by a final layer that picks the loglike of the [alpha, loglik, count]

    append_time_ram_stamp_to_file(f"Traning.make_model() end   {run_id}", config.bench_path, start)
    return model, cgp_hmm_layer


def make_dataset(config):
    start = time.perf_counter()
    run_id = randint(0,100)
    append_time_ram_stamp_to_file(f"Training.make_dataset() start {run_id}", config.bench_path, start)

    if config.generate_new_seqs:
        if config.use_simple_seq_gen:
            from generate_seqs import generate_simple
            generate_simple()
        else:
            command = f"python3 {config.src_path}/useMSAgen.py -c {config.nCodons} \
                          {'-n 100'} \
                          {'-l' + str(config.seq_len)} \
                          {'-cd ' + str(config.coding_dist) if config.coding_dist else ''} \
                          {'-ncd ' + str(config.noncoding_dist) if config.noncoding_dist else ''}\
                          {'--dont_strip_flanks' if config.dont_strip_flanks else ''} \
                          {'-p ' + config.src_path if config.src_path else ''} \
                          {'--insertions ' if config.simulate_insertions else ''} \
                          {'--deletions ' if config.simulate_deletions else ''}"

            command = re.sub("\s+", " ", command)
            run(command)


    def print_data_set(ds, name):
        dsout = open(name,"w")
        for batch_id, batch in enumerate(ds):
            for seq_id, seq in enumerate(batch):
                for id_in_seq, one_hot_entry in enumerate(seq):
                    # tf.print(one_hot_entry.numpy(), summarize = -1)
                    dsout.write(f"name = {name}, batch_id = {batch_id}, seq_id = {seq_id}, id in seq = {id_in_seq}\n")
                    dsout.write(config.model.emission_id_to_str(np.argmax(one_hot_entry.numpy())))
                    dsout.write("\n")
                    json.dump([float(x) for x in one_hot_entry.numpy()], dsout)
                    dsout.write("\n")

    use_old_read_seqs = 0

    # if use_old_read_seqs:
    index_of_terminal = config.model.emission_tuple_to_id("X")
    if use_old_read_seqs:
        seqs = read_data_with_order(config, add_one_terminal_symbol = True)
        ds = tf.data.Dataset.from_generator(lambda: seqs,
                                             tf.as_dtype(tf.int32), # has to be int, bc one_hot doesnt work for floats
                                             tf.TensorShape([None]))

        ds = ds.padded_batch(config.batch_size, padding_values = index_of_terminal)

        def to_one_hot(seq): # shoud this rather be called seqs? (plural)
            return tf.cast(tf.one_hot(seq, config.model.number_of_emissions), dtype=config.dtype)

        ds = ds.map(to_one_hot)

        # print_data_set(ds,"ds")

    if not use_old_read_seqs:
        # when mapping
        # Use tf.py_function, which allows you to write arbitrary Python code but will generally result in worse performance than 1). For example:
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
        seqs = read_data_one_hot_with_Ns_spread_str(config, add_one_terminal_symbol = True)

        dataset = tf.data.Dataset.from_generator(
            lambda: seqs, tf.string, output_shapes=[None])

        bucket_boundaries = [1000] * (len(seqs)//config.batch_size)
        bucket_batch_sizes = [config.batch_size] * (len(seqs)//config.batch_size) + [len(seqs) - len(seqs)//config.batch_size]
        # print("(len(seqs)//config.batch_size)", (len(seqs)//config.batch_size))

        # dataset = dataset.bucket_by_sequence_length(
        #     element_length_func = lambda elem: tf.shape(elem)[0],
        #     bucket_boundaries = bucket_boundaries,
        #     padded_shapes = tf.TensorShape(None),  # Each sequence has 2 values
        #     padding_values = f"0{'_0'*(config.model.number_of_emissions-2)}_1",
        #     bucket_batch_sizes = bucket_batch_sizes)
        dataset = dataset.padded_batch(config.batch_size, None, "_".join(["1.0" if i == index_of_terminal else "0.0" for i in range(config.model.number_of_emissions)]))
        dataset = dataset.map(lambda x: tf.strings.to_number(tf.strings.split(x,'_')))
        dataset = dataset.map(lambda x: x.to_tensor()) # bc it is ragged
        np.set_printoptions(linewidth=200)

        print_data_set(dataset, "ds_str")

    # TODO: shuffle dataset?
    dataset = dataset.repeat()

    if config.viterbi:
        seqs_out = convert_data_one_hot_with_Ns_spread_str_to_numbers(seqs)
        with open(f"{config.fasta_path}.json", "w") as out_file:
            json.dump(seqs_out, out_file)

    append_time_ram_stamp_to_file(f"Training.make_dataset() end   {run_id}", config.bench_path, start)
    return dataset, seqs

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

            dy_dx = tape.gradient(y,  [layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel])

            for g, name in zip(dy_dx, "IAB"):
                tf.print(f"gradient for {name}", g)
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(g)), [g], name = name, summarize = -1)


        exit()
################################################################################
    elif config.get_gradient_of_first_batch:
        print("get_gradient_of_first_batch is decrepated")
        exit()
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
            tape.watch([layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel])
            y = layer(first_batch)
            dy_dx = tape.gradient(y,  [layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel])
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
                    print("mean(loglike += log(sum_q(alpha)) =", tf.reduce_mean(loglike).numpy())
                    print("mean(log(sum_q(alpha * prod_zi))) =", tf.reduce_mean(tf.math.log(tf.reduce_sum(alpha * prod_zi, axis=1))).numpy())
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
                    print("mean(loglike += log(sum_q(alpha)) + log(sum_q(alpha)) =", tf.reduce_mean(loglike + tf.math.log(tf.reduce_sum(alpha, axis = 1))).numpy())
                    print("mean(log(sum_q(alpha * prod_zi))) =", tf.reduce_mean(tf.math.log(tf.reduce_sum(alpha * prod_zi, axis=1))).numpy())


                    # manual_forward
                    alpha = tf.matmul(batch[:,0,:], cell.B) * cell.I
                    for i in range(1, max_len_seq_in_batch + 1):
                        # E * R
                        alpha =  tf.matmul(batch[:,i,:], cell.B) * tf.matmul(alpha, cell.A)

                    loglike = tf.math.log(tf.reduce_sum(alpha, axis=1))
                    mean_loglike = tf.reduce_mean(loglike)

                    print("\n=========> alpha dp <===========")
                    print("mean(log(sum_q(alpha))) =", mean_loglike.numpy())
                    print()

                    # logsumexp
                    alpha = tf.math.log(tf.matmul(batch[:,0,:], cell.B)) + tf.math.log(cell.I)
                    for i in range(1, max_len_seq_in_batch + 1):
                        m_alpha = tf.reduce_max(alpha, axis = 1, keepdims = True)
                        alpha =  tf.math.log(tf.matmul(batch[:,i,:], cell.B)) + tf.math.log(tf.matmul(tf.math.exp(alpha - m_alpha), cell.A)) + m_alpha

                    loglike = tf.math.reduce_logsumexp(alpha, axis=1)
                    mean_loglike = tf.reduce_mean(loglike)

                    print("\n=========> alpha logsumexp <===========")
                    print("logsumexp(alpha) =", mean_loglike.numpy())
                    print()



            exit()

    elif config.manual_training_loop:
        assert config.return_seqs, "if manual training loop then return seqs must be true"
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
                batch = tf.one_hot(batch, config.model.number_of_emissions)
                # tf.print("batch = ", batch, summarize = -1)

                with tf.GradientTape() as tape:

                    tape.watch([layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel])
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
                        gradient = tape.gradient(-1*tf.math.reduce_mean(y),  [layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel])
                    elif config.alpha_i_gradient == -2: # ie. differentiate alpha(n-1), this should be the same as when y is differentiated
                        gradient = tape.gradient(-1*tf.math.reduce_mean(loglikes[-1]),  [layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel])
                    else:
                        gradient =  tape.gradient(-1*tf.math.reduce_mean(loglikes[config.alpha_i_gradient]),  [layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel])
                    # tf.print("gradient =", gradient, summarize = gradient_summarize)


                    # ik = [float(x) for x in gradient[0]]
                    ak = [float(x) for x in gradient[1]]
                    bk = [float(x) for x in gradient[2]]

                    # with open(f"{config.src_path}/output/{config.nCodons}codons/gradient_ik_for_alpha{config.alpha_i_gradient}.txt","w") as file:
                    #     json.dump(ik, file)
                    with open(f"{config.src_path}/output/{config.nCodons}codons/gradient_ak_for_alpha{config.alpha_i_gradient}.txt","w") as file:
                        json.dump(ak, file)
                    with open(f"{config.src_path}/output/{config.nCodons}codons/gradient_bk_for_alpha{config.alpha_i_gradient}.txt","w") as file:
                        json.dump(bk, file)

                    # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[0])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_I", summarize = config.assert_summarize)
                    # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[1])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_A", summarize = config.assert_summarize)
                    # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[2])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_B", summarize = config.assert_summarize)
                    # gradient[0] = config.learning_rate * gradient[0]
                    # gradient[1] = config.learning_rate * gradient[1]
                    # gradient[2] = config.learning_rate * gradient[2]
                    # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[0])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_I_after_learning_rate_scaled", summarize = config.assert_summarize)
                    # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[1])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_A_after_learning_rate_scaled", summarize = config.assert_summarize)
                    # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[2])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_B_after_learning_rate_scaled", summarize = config.assert_summarize)
                    # # print("layer.C.I_kernel before apply grad=", layer.C.I_kernel)
                    # optimizer.apply_gradients(zip(gradient, [layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel]))
                    # # print("layer.C.I_kernel after apply grad =", layer.C.I_kernel)
                    #
                    # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.I_kernel)),       [layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel], name = "I_kernel_after_apply_grads", summarize = config.assert_summarize)
                    # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.A_kernel)), [layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel], name = "I_kernel_after_apply_grads", summarize = config.assert_summarize)
                    # tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.B_kernel)),   [layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel], name = "B_kernel_after_apply_grads", summarize = config.assert_summarize)

                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[1])), [gradient[1], gradient[2]], name = "gradient_for_A", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[2])), [gradient[1], gradient[2]], name = "gradient_for_B", summarize = config.assert_summarize)
                    gradient[1] = config.learning_rate * gradient[1]
                    gradient[2] = config.learning_rate * gradient[2]
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[1])), [gradient[1], gradient[2]], name = "gradient_for_A_after_learning_rate_scaled", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[2])), [gradient[1], gradient[2]], name = "gradient_for_B_after_learning_rate_scaled", summarize = config.assert_summarize)
                    # print("layer.C.I_kernel before apply grad=", layer.C.I_kernel)
                    optimizer.apply_gradients(zip(gradient, [layer.C.I_kernel, layer.C.A_kernel, layer.C.B_kernel]))
                    # print("layer.C.I_kernel after apply grad =", layer.C.I_kernel)

                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.A_kernel)), [layer.C.A_kernel, layer.C.B_kernel], name = "I_kernel_after_apply_grads", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.B_kernel)),   [layer.C.A_kernel, layer.C.B_kernel], name = "B_kernel_after_apply_grads", summarize = config.assert_summarize)
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
                append_time_ram_stamp_to_file(f"Training:model.compile() start {run_id}", config.bench_path, start)
                model.compile(optimizer = optimizer, run_eagerly = config.run_eagerly)
                append_time_ram_stamp_to_file(f"Training:model.compile() end   {run_id}", config.bench_path, start)

                # fit model
                start = time.perf_counter()
                run_id = randint(0,100)
                append_time_ram_stamp_to_file(f"Training:model.fit() start {run_id}", config.bench_path, start)
                history = model.fit(data_set, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
                append_time_ram_stamp_to_file(f"Training:model.fit() end   {run_id}", config.bench_path, start)
        else:
             model, cgp_hmm_layer = make_model(config)
             model.summary()

             # compile model
             start = time.perf_counter()
             run_id = randint(0,100)
             append_time_ram_stamp_to_file(f"Training:model.compile() start {run_id}", config.bench_path, start)
             model.compile(optimizer = optimizer, run_eagerly = config.run_eagerly)
             append_time_ram_stamp_to_file(f"Training:model.compile() end   {run_id}", config.bench_path, start)

             # git model
             start = time.perf_counter()
             run_id = randint(0,100)
             append_time_ram_stamp_to_file(f"Training:model.fit() start {run_id}", config.bench_path, start)
             print("optimizer.iterations should be 0:", optimizer.iterations)
             history = model.fit(data_set, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
             print("optimizer.iterations should be larger 0:", optimizer.iterations)
             append_time_ram_stamp_to_file(f"Training:model.fit() end   {run_id}", config.bench_path, start)

    return model, history
