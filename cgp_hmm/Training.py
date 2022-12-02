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
    append_time_ram_stamp_to_file(start, f"Traning.make_model() start {run_id}", config["bench_path"])

    alphabet_size = 4

    if config["order_transformed_input"]:
        #                                                                                        terminal
        sequences = tf.keras.Input(shape = (None, (alphabet_size + 1) ** (config["order"] + 1) + 1), name = "sequences", dtype = config["dtype"])
    else:
        sequences = tf.keras.Input(shape = (None, alphabet_size + 2), name = "sequences", dtype = config["dtype"])

    # another None added automatically for yet unkown batch_size

    cgp_hmm_layer = CgpHmmLayer(config) # init of layer

    loglik = cgp_hmm_layer(sequences) # layer is build, then called. it seems i cant call build before to avoid building it here again
    # "[tf.keras.layers.Lambda(lambda x:x, name = \"loglik\")(loglik)] =", [
    print(tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik))

    model = tf.keras.Model(inputs = sequences, outputs = [tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik)]) #  the output of the model is the value that is computed by a final layer that picks the loglike of the [alpha, loglik, count]

    append_time_ram_stamp_to_file(start, f"Traning.make_model() end   {run_id}", config["bench_path"])
    return model, cgp_hmm_layer


def make_dataset(config):
    start = time.perf_counter()
    run_id = randint(0,100)
    append_time_ram_stamp_to_file(start, f"Training.make_dataset() start {run_id}", config["bench_path"])

    if config["order_transformed_input"]:
        seqs = read_data_with_order(config["fasta_path"], config["order"])
    else:
        seqs = read_data(config["fasta_path"])

    ds = tf.data.Dataset.from_generator(lambda: seqs,
                                         tf.as_dtype(tf.int32), # has to be int, bc one_hot doesnt work for floats
                                         tf.TensorShape([None]))
    if config["order_transformed_input"]:
        ds = ds.padded_batch(32, padding_values = (4 + 1)**(config["order"] + 1))

        def to_one_hot(seq):
            return tf.cast(tf.one_hot(seq, (4 + 1)**(config["order"] + 1) + 1), dtype=config["dtype"])
    else:
        ds = ds.padded_batch(32, padding_values = 5) # 5 is terminal symbol, 4 is "padded left flank"

        def to_one_hot(seq):
            return tf.cast(tf.one_hot(seq, 4 + 1 + 1), dtype=config["dtype"])

    ds = ds.map(to_one_hot)
    ds = ds.repeat()

    append_time_ram_stamp_to_file(start, f"Training.make_dataset() end   {run_id}", config["bench_path"])
    return ds, seqs

# from memory_profiler import profile
# @profile
def fit_model(config):
    nCodons = config["nCodons"]


    # model, cgp_hmm_layer = make_model(config)

    num_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
    print("Using", num_gpu, "GPUs. device_lib.list_local_devices()")

    num_gpu = len(tf.config.experimental.list_logical_devices('GPU'))
    print("Using", num_gpu, "GPUs.tf.config.experimental.list_logical_devices('GPU')")

    if "clip_gradient_by_value" in config:
        optimizer = tf.optimizers.Adam(config["learning_rate"], clipvalue = config["clip_gradient_by_value"])
    else:
        optimizer = tf.optimizers.Adam(config["learning_rate"])
     # manual call to forward algo

    # _, seqs = make_dataset()# first return value is data_set
    # model(seqs)

    data_set, seqs = make_dataset(config)

    output_path = f"bench/{nCodons}codons"





################################################################################
    def get_batch_input_from_file():
        input = []
        with open(f"{config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_inputs.txt", "r") as file:
            seq = []
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    input.append(seq)
                    seq = []
                else:
                    line = re.sub("[\[\]]","", line)
                    line = line.split(" ")
                    line = [float(x) for x in line]
                    seq.append(line)
            if len(seq) != 0:
                input.append(seq)
        return tf.constant(input, dtype = tf.float32)

                    # use this instead to get easy access to gradients
                    # but then i have to manually do data management ie splitting into batches

                    # optimizer = tf.optimizers.Adam()
                    # def optimize(x, y):
                    #     with tf.GradientTape() as tape:
                    #         predictions = network(x, is_training=True)
                    #         loss = cross_entropy_loss(predictions, y)
                    #     gradients = tape.gradient(loss, model.trainable_variables)
                    #     gradients = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in gradients]
                    #     optimizer.apply_gradients(zip(gradients,     model.trainable_variables))

################################################################################
    if config["get_gradient_for_current_txt"] or config["get_gradient_from_saved_model_weights"]:

        if config["get_gradient_from_saved_model_weights"]:
            model, cgp_hmm_layer = make_model(config)
            model.load_weights(f"{config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_weights")
            config["model"] = model
            print('config["model"]', config["model"])

        layer = CgpHmmLayer(config)
        layer.build(None)
        layer.C.build(None)

        # assuming that inputs are formatted in shape batch, seq_len, one_hot_dim = 32, l, 126
        input = get_batch_input_from_file()
        with tf.GradientTape() as tape:
            y = layer(input) # eventuell wird hier die cell nochmal gebaut und das weight setzen davor bringt nichts
            dy_dx = tape.gradient(y,  [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
            for g, name in zip(dy_dx, "IAB"):
                tf.print(f"gradient for {name}", g)
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(g)), [g], name = name, summarize = -1)

        exit()
################################################################################
    elif config["get_gradient_of_first_batch"]:
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
    else:
        if num_gpu > 1:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model, cgp_hmm_layer = make_model(config)
                model.summary()
                start = time.perf_counter()
                run_id = randint(0,100)
                append_time_ram_stamp_to_file(start, f"Training:model.compile() start {run_id}", config["bench_path"])
                model.compile(optimizer = optimizer)
                append_time_ram_stamp_to_file(start, f"Training:model.compile() end   {run_id}", config["bench_path"])

                start = time.perf_counter()
                run_id = randint(0,100)
                append_time_ram_stamp_to_file(start, f"Training:model.fit() start {run_id}", config["bench_path"])
                history = model.fit(data_set, epochs=5, steps_per_epoch=15, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
                append_time_ram_stamp_to_file(start, f"Training:model.fit() end   {run_id}", config["bench_path"])
        else:
             model, cgp_hmm_layer = make_model(config)
             model.summary()
             start = time.perf_counter()
             run_id = randint(0,100)
             append_time_ram_stamp_to_file(start, f"Training:model.compile() start {run_id}", config["bench_path"])
             model.compile(optimizer = optimizer)
             append_time_ram_stamp_to_file(start, f"Training:model.compile() end   {run_id}", config["bench_path"])

             start = time.perf_counter()
             run_id = randint(0,100)
             append_time_ram_stamp_to_file(start, f"Training:model.fit() start {run_id}", config["bench_path"])
             history = model.fit(data_set, epochs=5, steps_per_epoch=15, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
             append_time_ram_stamp_to_file(start, f"Training:model.fit() end   {run_id}", config["bench_path"])


    return model, history
