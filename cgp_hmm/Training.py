#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint
import os

from Utility import run
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

    learning_rate = .1

    optimizer = tf.optimizers.Adam(learning_rate)

     # manual call to forward algo

    # _, seqs = make_dataset()# first return value is data_set
    # model(seqs)

    data_set = make_dataset(config)[0] # [1] is data tensor

    output_path = f"bench/{nCodons}codons"


    # class my_callback(tf.keras.callbacks.Callback):
    #     def on_epoch_begin(self, epoch, logs = None):
    #         print("model.weights")
    #         print("A =", tf.nn.softmax(model.get_weights()[0]))


    class write_time_epoch_start_callback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs = None):
            with open(f"{output_path}/callbackoutput_time_start.txt", "a") as file:
                file.write(f"{time.time()}\n")
    class write_time_epoch_end_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = None):
            with open(f"{output_path}/callbackoutput_time_end.txt", "a") as file:
                file.write(f"{time.time()}\n")

    # import os, psutil
    # process = psutil.Process(os.getpid())

    # todo: oder nicht epoch sondern batch
    # on_train_batch_begin
    class write_time_ram_epoch_start_callback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs = None):
            # with open(f"{output_path}/callbackoutput_ram_start.txt", "a") as file:
            #     file.write(f"{process.memory_info().rss}\n")
            append_time_ram_stamp_to_file(0, "epoch_begin", config["bench_path"])

    class write_time_ram_epoch_end_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = None):
            # with open(f"{output_path}/callbackoutput_ram_end.txt", "a") as file:
            #     file.write(f"{process.memory_info().rss}\n")
                #                              oder vms     Virtual Memory Size
            append_time_ram_stamp_to_file(0, "epoch_end", config["bench_path"])

    class exit_after_first_batch(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs = None):
            exit(1)

    class exit_after_loglik_is_nan(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs = None):
            if tf.math.reduce_any(tf.math.is_nan(logs["loglik"])):
                print("loglik_mean contained nan")
                exit(1)

    class remove_verbose_at_batch_begin(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            os.system(f"rm {config['src_path']}/verbose/{nCodons}codons.txt")


    # checkpoint_path = "training_1/cp.ckpt"
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)


    # callbacks = [tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: print("A =", tf.nn.softmax(model.get_weights()[0])))]
    callbacks = []
    callbacks = [write_time_ram_epoch_start_callback(),
                 write_time_ram_epoch_end_callback()]

    if "exit_after_first_batch" in config and config["exit_after_first_batch"]:
        callbacks += [exit_after_first_batch()]
    if "exit_after_loglik_is_nan" in config and config["exit_after_loglik_is_nan"]:
        callbacks += [exit_after_loglik_is_nan()]
    if "only_keep_verbose_of_last_batch" in config and config["only_keep_verbose_of_last_batch"]:
        callbacks += [remove_verbose_at_batch_begin()]


    if num_gpu > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model, cgp_hmm_layer = make_model(config)
            start = time.perf_counter()
            run_id = randint(0,100)
            append_time_ram_stamp_to_file(start, f"Training:model.compile() start {run_id}", config["bench_path"])
            model.compile(optimizer = optimizer)
            append_time_ram_stamp_to_file(start, f"Training:model.compile() end   {run_id}", config["bench_path"])

            start = time.perf_counter()
            run_id = randint(0,100)
            append_time_ram_stamp_to_file(start, f"Training:model.fit() start {run_id}", config["bench_path"])
            history = model.fit(data_set, epochs=5, steps_per_epoch=15, callbacks = callbacks) # with callbacks it is way slower
            append_time_ram_stamp_to_file(start, f"Training:model.fit() end   {run_id}", config["bench_path"])
    else:
         model, cgp_hmm_layer = make_model(config)
         start = time.perf_counter()
         run_id = randint(0,100)
         append_time_ram_stamp_to_file(start, f"Training:model.compile() start {run_id}", config["bench_path"])
         model.compile(optimizer = optimizer)
         append_time_ram_stamp_to_file(start, f"Training:model.compile() end   {run_id}", config["bench_path"])

         start = time.perf_counter()
         run_id = randint(0,100)
         append_time_ram_stamp_to_file(start, f"Training:model.fit() start {run_id}", config["bench_path"])
         history = model.fit(data_set, epochs=5, steps_per_epoch=15, callbacks = callbacks) # with callbacks it is way slower
         append_time_ram_stamp_to_file(start, f"Training:model.fit() end   {run_id}", config["bench_path"])



    return model, history
