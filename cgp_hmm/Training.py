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
from Utility import append_time_ram_stamp_to_file
from Utility import transform_verbose_txt_to_csv

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
            # das vielleicht rein ins callback, da ja exit und der code hier dann ja gar nicht mehr erreicht wird
            if config["verbose"] and config["exit_after_first_batch"]:
                transform_verbose_txt_to_csv(f"{config['src_path']}/verbose/{nCodons}codons.txt", nCodons)
            exit(1)

    class exit_after_loglik_is_nan(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs = None):
            if tf.math.reduce_any(tf.math.is_nan(logs["loglik"])):
                print("loglik_mean contained nan")
                exit(1)

    class remove_verbose_at_batch_begin(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            os.system(f"rm {config['src_path']}/verbose/{nCodons}codons.txt")

    class write_weights_to_file_and_exit_when_nan(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            ik, ak, bk = model.get_weights()
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(ik)), [ak], name = "I_is_nan", summarize = -1)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(ak)), [ak], name = "A_is_nan", summarize = -1)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(bk)), [ak], name = "B_is_nan", summarize = -1)

            os.system(f"rm {config['src_path']}/output/{config['nCodons']}codons/current_I.json")
            os.system(f"rm {config['src_path']}/output/{config['nCodons']}codons/current_A.json")
            os.system(f"rm {config['src_path']}/output/{config['nCodons']}codons/current_B.json")

            # in layer, inputs is written to file
            os.system(f"rm {config['src_path']}/output/{config['nCodons']}codons/current_inputs.txt")

            ik = [float(x) for x in ik]
            ak = [float(x) for x in ak]
            bk = [float(x) for x in bk]

            with open(f"{config['src_path']}/output/{config['nCodons']}codons/current_I.json", "w") as file:
                json.dump(ik, file)
            with open(f"{config['src_path']}/output/{config['nCodons']}codons/current_A.json", "w") as file:
                json.dump(ak, file)
            with open(f"{config['src_path']}/output/{config['nCodons']}codons/current_B.json", "w") as file:
                json.dump(bk, file)


    class get_the_gradient(tf.keras.callbacks.Callback):

        # def get_weight_grad(model, inputs, outputs):
        #     """ Gets gradient of model for given inputs and outputs for all weights"""
        #     grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
        #     symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        #     f = K.function(symb_inputs, grads)
        #     x, y, sample_weight = model._standardize_user_data(inputs, outputs)
        #     output_grad = f(x + y + sample_weight)
        #     return output_grad

        # also print the gradient on batch begin
        def on_train_batch_begin(self, batch, logs = None):
            pass

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
    if "most_recent_weights_and_inputs_to_file" in config and config["most_recent_weights_and_inputs_to_file"]:
        callbacks += [write_weights_to_file_and_exit_when_nan()]

    callbacks += [get_the_gradient()]

    if config["get_gradient_for_current_txt"]:
        layer = CgpHmmLayer(config)
        layer.build(None)
        layer.C.build(None)

        # getting weights from files, which where written the last time the main_programm was run
        with open(f"{config['src_path']}/output/{config['nCodons']}codons/current_I.json") as file:
            weights_I = np.array(json.load(file))
        with open(f"{config['src_path']}/output/{config['nCodons']}codons/current_A.json") as file:
            weights_A = np.array(json.load(file))
        with open(f"{config['src_path']}/output/{config['nCodons']}codons/current_B.json") as file:
            weights_B = np.array(json.load(file))
        weights = [weights_I, weights_A, weights_B]
        layer.C.set_weights(weights)

        # assuming that inputs are formatted in shape batch, seq_len, one_hot_dim = 32, l, 126
        input = []
        with open(f"{config['src_path']}/output/{config['nCodons']}codons/current_inputs.txt", "r") as file:
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
        input = tf.constant(input, dtype = tf.float32)

        with tf.GradientTape() as tape:
            tape.watch([layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
            y = layer(input) # heir wird viel geprintet
            dy_dx = tape.gradient(y,  [layer.C.init_kernel, layer.C.transition_kernel, layer.C.emission_kernel])
            if not dy_dx:
                print("not dy_dx")
                print("list dy_dx =", round(dy_dx,3))
            print("::::::::::::::::::::::::::::::::::::::::::::::")
            for g, name in zip(dy_dx, "IAB"):
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(g)), [g], name = name, summarize = -1)

        exit()

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
                history = model.fit(data_set, epochs=5, steps_per_epoch=15, callbacks = callbacks) # with callbacks it is way slower
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
             history = model.fit(data_set, epochs=5, steps_per_epoch=15, callbacks = callbacks) # with callbacks it is way slower
             append_time_ram_stamp_to_file(start, f"Training:model.fit() end   {run_id}", config["bench_path"])


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

    return model, history
