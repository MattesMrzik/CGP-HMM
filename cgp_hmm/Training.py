#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint
import os
import json
import re

from Utility import run
from CallBacks import get_call_backs
from Utility import transform_verbose_txt_to_csv
from Utility import append_time_ram_stamp_to_file


from CgpHmmLayer import CgpHmmLayer
from ReadData import read_data_one_hot
from ReadData import read_data
from ReadData import read_data_with_order
from tensorflow.python.client import device_lib

import time

def prRed(skk): print(f"Training\033[96m {skk}\033[00m")

np.set_printoptions(linewidth=400)

def make_model(config):
    start = time.perf_counter()
    run_id = randint(0,100)
    append_time_ram_stamp_to_file(start, f"Traning.make_model() start {run_id}", config.bench_path)

    alphabet_size = 4

    if config.order_transformed_input:
        #                                                                                        terminal
        sequences = tf.keras.Input(shape = (None, (alphabet_size + 1) ** (config.order + 1) + 1), name = "sequences", dtype = config.dtype)
    else:
        sequences = tf.keras.Input(shape = (None, alphabet_size + 2), name = "sequences", dtype = config.dtype)

    # another None added automatically for yet unkown batch_size

    cgp_hmm_layer = CgpHmmLayer(config) # init of layer

    loglik = cgp_hmm_layer(sequences) # layer is build, then called. it seems i cant call build before to avoid building it here again
    # "[tf.keras.layers.Lambda(lambda x:x, name = \"loglik\")(loglik)] =", [
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

    if config.generate_new_seqs:
        if config.use_simple_seq_gen:
            num_seqs = 100
            seqs = {}
            with open(config.fasta_path, "w") as file:
                genlen = 3 * config.nCodons # ATG and STOP are not on gene
                seqlen = genlen * config.l
                seqlen += 6 # start and stop codon
                seqlen += 2 # ig states
                max_flanklen = (seqlen - genlen )//2
                low = max_flanklen - 1 if config.dont_strip_flanks else 1

                for seq_id in range(num_seqs):

                    ig5 = "".join(np.random.choice(["A","C","G","T"], np.random.randint(low, max_flanklen))) # TODO: also check if low = 2
                    atg = "ATG"
                    # coding = "".join(np.random.choice(["A","C","G","T"], config["nCodons"] * 3))
                    coding = "".join(np.random.choice(codons, config.nCodons))
                    stop = np.random.choice(["TAA","TGA","TAG"])
                    ig3 = "".join(np.random.choice(["A","C","G","T"], np.random.randint(low, max_flanklen)))

                    seqs[f">use_simple_seq_gen_{seq_id}"] = ig5 + atg + coding + stop + ig3
                for key, value in seqs.items():
                    file.write(key + "\n")
                    file.write(value + "\n")
        else:
            command = f"python3 {config.src_path}/useMSAgen.py -c {config.nCodons} \
                          {'-n 4'} \
                          {'-l' + str(config.l)} \
                          {'-cd ' + str(config.coding_dist) if config.coding_dist else ''} \
                          {'-ncd ' + str(config.noncoding_dist) if config.noncoding_dist else ''}\
                          {'--dont_strip_flanks' if config.dont_strip_flanks else ''} \
                          {'-p ' + config.src_path if config.src_path else ''}"
            command = re.sub("\s+", " ", command)
            run(command)


    if config.order_transformed_input:
        seqs = read_data_with_order(config.fasta_path, config.order, add_one_terminal_symbol = True)
    else:
        seqs = read_data(config.fasta_path)

    ds = tf.data.Dataset.from_generator(lambda: seqs,
                                         tf.as_dtype(tf.int32), # has to be int, bc one_hot doesnt work for floats
                                         tf.TensorShape([None]))
    if config.order_transformed_input:
        ds = ds.padded_batch(32, padding_values = (4 + 1)**(config.order + 1))

        def to_one_hot(seq):
            return tf.cast(tf.one_hot(seq, (4 + 1)**(config.order + 1) + 1), dtype=config.dtype)
    else:
        ds = ds.padded_batch(32, padding_values = 5) # 5 is terminal symbol, 4 is "padded left flank"

        def to_one_hot(seq):
            return tf.cast(tf.one_hot(seq, 4 + 1 + 1), dtype=config.dtype)

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
    # model, cgp_hmm_layer = make_model(config)

    num_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
    print("Using", num_gpu, "GPUs. device_lib.list_local_devices()")

    num_gpu = len(tf.config.experimental.list_logical_devices('GPU'))
    print("Using", num_gpu, "GPUs.tf.config.experimental.list_logical_devices('GPU')")

    if config.optimizer == "Adam":
        if config.clip_gradient_by_value:
            # optimizer = tf.optimizers.Adam(config["learning_rate"], clipvalue = config["clip_gradient_by_value"])
            optimizer = tf.optimizers.Adam(config.learning_rate, clipvalue = config.clip_gradient_by_value)
            config.optimizer = f"Adam with learning_rate {config.learning_rate} and clipvalue {config.clip_gradient_by_value}"
        else:
            # optimizer = tf.optimizers.Adam(config["learning_rate"])
            optimizer = tf.optimizers.Adam(config.learning_rate)
            config.optimizer = f"Adam with learning_rate {config.learning_rate} and no clipping"

    elif config.optimizer == "SGD":
        optimizer = tf.optimizers.SGD(config.learning_rate)
        config.optimizer = f"SGD with learning_rate {config.learning_rate}"
    else:
        print("setting optimizer didnt work")
        exit(1)

     # manual call to forward algo

    # _, seqs = make_dataset()# first return value is data_set
    # model(seqs)
    data_set, seqs = make_dataset(config)

    output_path = f"bench/{config.nCodons}codons"



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
            y = layer(input) # eventuell wird hier die cell nochmal gebaut und das weight setzen davor bringt nichts
            dy_dx = tape.gradient(y,  [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])

            for g, name in zip(dy_dx, "IAB"):
                tf.print(f"gradient for {name}", g)
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(g)), [g], name = name, summarize = -1)

            # TODO: this doesnt work
            # gradient_of_optimizer = optimizer.compute_gradients(y, [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel], tape) # todo: this might not have "meta" parameters that change over time, but i dont even know if any exist
            # for g, name in zip(gradient_of_optimizer, ["i_grad_optimizer", "a_grad_optimizer", "b_grad_optimizer"]):
            #     tf.print(f"gradient for {name}", g)
            #     tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(g)), [g], name = name, summarize = -1)

        exit()
################################################################################
    elif config.get_gradient_of_first_batch:
            layer = CgpHmmLayer(config)
            layer.build(None)
            layer.C.build(None)

            first_batch = seqs[:32] # not one hot, not padded
            # pad seqs:
            max_len_seq_in_batch = max([len(seq) for seq in first_batch])
            # print("max_len_seq_in_batch =", max_len_seq_in_batch)
            # for seq in first_batch:
            #     print((max_len_seq_in_batch - len(seq)))
            #     print(seq + [126] * (max_len_seq_in_batch - len(seq)))
            first_batch = [seq + [125] * (max_len_seq_in_batch - len(seq)) for seq in first_batch]

            # print("first_batch =", "\n".join([str(seq) for seq in first_batch]))

            # one_hot:
            first_batch = tf.one_hot(first_batch, 126)

            # print("first_batch =", "\n".join([str(seq) for seq in first_batch]))

            with tf.GradientTape() as tape:
                tape.watch([layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
                y = layer(first_batch)
                dy_dx = tape.gradient(y,  [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
                if not dy_dx:
                    print("list dy_dx =", round(dy_dx,3))
                print("::::::::::::::::::::::::::::::::::::::::::::::")
                for g in dy_dx:
                    print("dy_dx =", g)
                    # print("dy_dx.numpy() =", g.numpy())
                    print()
            exit()

################################################################################
    elif config.manual_traning_loop:
        #  get the very first init weights of a run that resulted in nan
        # maybe this is not necessary, since i can run with --dont_generate_new_seqs flag, and even though the kernels are always initialized differently nan always occur

        layer = CgpHmmLayer(config)
        layer.build(None)
        layer.C.build(None)


        for epoch in range(config.epochs):
            for step in range(4):
                minimum = step*32
                maximum = min((step+1)*32, 100)
                batch = seqs[minimum : maximum]
                max_len_seq_in_batch = max([len(seq) for seq in batch])
                batch = [seq + [125] * (max_len_seq_in_batch - len(seq)) for seq in batch]
                batch = tf.one_hot(batch, 126)

                with tf.GradientTape() as tape:

                    tape.watch([layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
                    y = layer(batch)

                    print(f"epoch({epoch}), step({step}) the loss is:\n{tf.math.reduce_mean(y)}")
                    gradient = tape.gradient(-1*y,  [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
                    # print("gradient =", gradient)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[0])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_I", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[1])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_A", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(gradient[2])), [gradient[0], gradient[1], gradient[2]], name = "gradient_for_B", summarize = config.assert_summarize)

                    optimizer.apply_gradients(zip(gradient, [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel]))

                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.init_kernel)),       [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel], name = "I_kernel_after_apply_grads", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.transition_kernel)), [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel], name = "A_kernel_after_apply_grads", summarize = config.assert_summarize)
                    tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(layer.C.emission_kernel)),   [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel], name = "B_kernel_after_apply_grads", summarize = config.assert_summarize)
        exit()


################################################################################
    else:
        if num_gpu > 1 and not config.dont_use_gpu:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model, cgp_hmm_layer = make_model(config)
                model.summary()
                start = time.perf_counter()
                run_id = randint(0,100)
                append_time_ram_stamp_to_file(start, f"Training:model.compile() start {run_id}", config.bench_path)
                model.compile(optimizer = optimizer)
                append_time_ram_stamp_to_file(start, f"Training:model.compile() end   {run_id}", config.bench_path)

                start = time.perf_counter()
                run_id = randint(0,100)
                append_time_ram_stamp_to_file(start, f"Training:model.fit() start {run_id}", config.bench_path)
                history = model.fit(data_set, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
                append_time_ram_stamp_to_file(start, f"Training:model.fit() end   {run_id}", config.bench_path)
        else:
             model, cgp_hmm_layer = make_model(config)
             model.summary()
             start = time.perf_counter()
             run_id = randint(0,100)
             append_time_ram_stamp_to_file(start, f"Training:model.compile() start {run_id}", config.bench_path)
             model.compile(optimizer = optimizer)
             append_time_ram_stamp_to_file(start, f"Training:model.compile() end   {run_id}", config.bench_path)

             start = time.perf_counter()
             run_id = randint(0,100)
             append_time_ram_stamp_to_file(start, f"Training:model.fit() start {run_id}", config.bench_path)
             print("optimizer.iterations should be 0:", optimizer.iterations)
             history = model.fit(data_set, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
             print("optimizer.iterations should be larger 0:", optimizer.iterations)
             append_time_ram_stamp_to_file(start, f"Training:model.fit() end   {run_id}", config.bench_path)


    return model, history
