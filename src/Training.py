#!/usr/bin/env python3
from Utility import append_time_ram_stamp_to_file
from ReadData import convert_data_one_hot_with_Ns_spread_str_to_numbers, read_data_one_hot_with_Ns_spread_str
from CgpHmmLayer import CgpHmmLayer
from CallBacks import get_call_backs
import tensorflow as tf
import numpy as np
import time
import os
import json
import numpy as np
import time
from tensorflow.python.client import device_lib

def make_model(config, current_epoch = None):
    start = time.perf_counter()

    append_time_ram_stamp_to_file(f"Traning.make_model() start ", config.bench_path, start)

    # another None added automatically for yet unkown batch_size
    sequences = tf.keras.Input(shape = (None, config.model.number_of_emissions), name = "sequences", dtype = config.dtype)

    masked_values = tf.keras.layers.Masking(mask_value=0.)(sequences)

    cgp_hmm_layer = CgpHmmLayer(config, current_epoch = current_epoch) # init of layer

    loglik = cgp_hmm_layer(masked_values) # layer is build, then called. it seems i cant call build before to avoid building it here again


    model = tf.keras.Model(inputs = sequences, outputs = [tf.keras.layers.Lambda(lambda x:x, name = "loglik")(loglik)]) #  the output of the model is the value that is computed by a final layer that picks the loglike of the [alpha, loglik, count]

    append_time_ram_stamp_to_file(f"Traning.make_model() end ", config.bench_path, start)
    return model, cgp_hmm_layer


def make_dataset(config):
    start = time.perf_counter()
    append_time_ram_stamp_to_file(f"Training.make_dataset() start ", config.bench_path, start)

    def print_data_set(ds, name):
        dsout = open(name,"w")
        for batch_id, batch in enumerate(ds):
            for seq_id, seq in enumerate(batch):
                for id_in_seq, one_hot_entry in enumerate(seq):
                    dsout.write(f"name = {name}, batch_id = {batch_id}, seq_id = {seq_id}, id in seq = {id_in_seq}\n")
                    dsout.write(config.model.emission_id_to_str(np.argmax(one_hot_entry.numpy())))
                    dsout.write("\n")
                    json.dump([float(x) for x in one_hot_entry.numpy()], dsout)
                    dsout.write("\n")


    index_of_terminal = config.model.emission_tuple_to_id("X")

    seqs = read_data_one_hot_with_Ns_spread_str(config, add_one_terminal_symbol = True)
    config.nSeqs = len(seqs)

    if not config.dont_shuffle_seqs:
        np.random.shuffle(seqs)

    def get_initial_data_set():
        dataset = tf.data.Dataset.from_generator(lambda: seqs, \
                                                    tf.string, \
                                                    output_shapes=[None])
        return dataset

    # since I can only pad with scalar values and I spread the Ns, I have to encode a multi hot emission in a string.
    # then pad the terminal symbol in its string representation. When making a batch I can convert the strings
    # to actual multi hot encodings
    padding_value = "_".join(["1.0" if i == index_of_terminal else "0.0" for i in range(config.model.number_of_emissions)])

    if config.bucket_by_seq_len:

        def batch_sizes_are_unequal_one(dataset):
            for batch_id, batch in enumerate(dataset):
                if len(batch) < 2:
                    return False
            return True

        def bucket_seqs_of_dataset(dataset, batch_size):
            sorted_seq_lens = sorted([len(seq) for seq in seqs])
            print("sorted_seq_lens", sorted_seq_lens)
            bucket_boundaries = [length +1 for i, length in enumerate(sorted_seq_lens) if (i +1) % batch_size == 0]

            # sort bucket_boundries, such that the ones closest to the median seqlen come first,
            # and buckets that differ much from the median are trained at the end
            median_seq_len = np.median(sorted_seq_lens)
            bucket_boundaries = sorted(bucket_boundaries, key = lambda x: abs(x - median_seq_len))

            config.nBatches = len(bucket_boundaries) + 1
            bucket_batch_sizes = [batch_size] * config.nBatches
            print("bucket_boundaries", bucket_boundaries)
            print("bucket_batch_sizes", bucket_batch_sizes)

            if not config.dont_shuffle_seqs:
                dataset = dataset.shuffle(buffer_size = config.nSeqs, reshuffle_each_iteration = True)

            dataset = dataset.bucket_by_sequence_length(
                element_length_func = lambda elem: tf.shape(elem)[0],
                bucket_boundaries = bucket_boundaries,
                padded_shapes = tf.TensorShape(None),  # Each sequence has 2 values
                padding_values = padding_value,
                bucket_batch_sizes = bucket_batch_sizes)

            return dataset

        adjusted_batch_size = config.batch_size

        # since it causes problem if a bucket batch is only a single seq
        # i increase the batch size until this is not the case anymore
        dataset = get_initial_data_set()
        dataset = bucket_seqs_of_dataset(dataset, adjusted_batch_size)

        while not batch_sizes_are_unequal_one(dataset):
            adjusted_batch_size += 1
            dataset = get_initial_data_set()
            dataset = bucket_seqs_of_dataset(dataset, adjusted_batch_size)
        print(f"Batch_size was adjusted from {config.batch_size} to {adjusted_batch_size} to avoid a bucket batch beeing only a single sequence.")

    if not config.bucket_by_seq_len:

        def get_adjusted_batch_size(nSeqs, initial_batch_size):
            adjusted_batch_size = initial_batch_size
            while nSeqs % adjusted_batch_size == 1:
                adjusted_batch_size += 1
            return adjusted_batch_size
        adjusted_batch_size = get_adjusted_batch_size(config.nSeqs, config.batch_size)
        print(f"Batch_size was adjusted from {config.batch_size} to {adjusted_batch_size} to avoid a batch beeing only a single sequence.")
        config.nBatches = config.nSeqs // adjusted_batch_size + 1
        dataset = get_initial_data_set()
        dataset = dataset.shuffle(buffer_size = config.nSeqs, reshuffle_each_iteration = True)
        dataset = dataset.padded_batch(adjusted_batch_size, None, padding_value)

    if config.steps_per_epoch == 0:
        config.steps_per_epoch = config.nBatches
        print("setting steps_per_epoch to", config.steps_per_epoch)

    dataset = dataset.map(lambda x: tf.strings.to_number(tf.strings.split(x,'_')))
    dataset = dataset.map(lambda x: x.to_tensor()) # bc it is ragged
    dataset = dataset.repeat()

    append_time_ram_stamp_to_file(f"Training.make_dataset() end ", config.bench_path, start)
    return dataset, seqs

################################################################################
################################################################################
################################################################################
def fit_model(config):

    num_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
    print("Using", num_gpu, "GPUs. device_lib.list_local_devices()")

    num_gpu_logical = len(tf.config.experimental.list_logical_devices('GPU'))
    print("Using", num_gpu_logical, "GPUs.tf.config.experimental.list_logical_devices('GPU')")

    # setting the optimizer
    if config.optimizer.lower() == "adam":
        if config.clip_gradient_by_value:
            optimizer = tf.optimizers.Adam(config.learning_rate, clipvalue = config.clip_gradient_by_value)
            config.optimizer = f"Adam with learning_rate {config.learning_rate} and clipvalue {config.clip_gradient_by_value}"
        else:
            optimizer = tf.optimizers.Adam(config.learning_rate)
            config.optimizer = f"Adam with learning_rate {config.learning_rate} and no clipping"
        print("config.optimizer =", config.optimizer)

    elif config.optimizer.lower() == "sgd":
        optimizer = tf.optimizers.SGD(config.learning_rate)
        config.optimizer = f"SGD with learning_rate {config.learning_rate}"
        print("config.optimizer =", config.optimizer)
    else:
        print("setting optimizer didnt work")
        exit(1)

    data_set, seqs = make_dataset(config)

    index_of_terminal = config.model.emission_tuple_to_id("X")

################################################################################
    if config.manual_forward:
        layer = CgpHmmLayer(config)
        layer.build(None)
        layer.C.build(None)
        cell = layer.C
        batchsize = config.batch_size
        low = 0
        seqs = convert_data_one_hot_with_Ns_spread_str_to_numbers(seqs)
        n = len(seqs)
        high = min(batchsize, n)
        for epoch in range(config.epochs):
            for step in range(config.steps_per_epoch):
                print(f"low = {low}, high = {high}")
                batch = seqs[low : high]

                low = low + batchsize if low + batchsize < n else 0
                if high == n:
                    high = min(batchsize, n)
                elif high + batchsize > n:
                    high = n
                else:
                    high += batchsize

                min_len_seq_in_batch = min([len(seq) for seq in batch])

                if min_len_seq_in_batch > 100:
                    batch = [seq[:100] for seq in batch]

                max_len_seq_in_batch = max([len(seq) for seq in batch])

                # print("max_len_seq_in_batch =", max_len_seq_in_batch)
                padding_value = [1.0 if i == index_of_terminal else 0.0 for i in range(config.model.number_of_emissions)]
                batch = [seq + [padding_value] * (max_len_seq_in_batch - len(seq) + 1) for seq in batch]

                batch = tf.convert_to_tensor(batch, dtype = config.dtype)

                # felix
                alpha = tf.matmul(batch[:,0,:], cell.B) * cell.I
                prod_zi =  1
                loglike = tf.math.log(tf.reduce_sum(alpha, axis = 1))
                for i in range(1, max_len_seq_in_batch + 1):
                    z_i_minus_1 = tf.reduce_sum(alpha, axis = 1, keepdims = True)
                    prod_zi *= z_i_minus_1
                    alpha =  tf.matmul(batch[:,i,:], cell.B) * tf.matmul(alpha, cell.A)
                    alpha = tf.math.divide(alpha, z_i_minus_1)
                    loglike += tf.math.log(tf.reduce_sum(alpha, axis = 1))
                print("\n=========> felix version to scale <===========")
                print("mean(loglike += log(sum_q(alpha)) =", tf.reduce_mean(loglike).numpy())

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

                print("\n=========> my old evrsion to scale for wich no gradient could be calculated <===========")
                print("mean(loglike += log(sum_q(alpha)) + log(sum_q(alpha)) =", tf.reduce_mean(loglike + tf.math.log(tf.reduce_sum(alpha, axis = 1))).numpy())

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
################################################################################
    else:
        if num_gpu > 1 and not config.dont_use_mirrored_strategy:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model, cgp_hmm_layer = make_model(config)
                model.summary()

                # compile model
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Training:model.compile() start ", config.bench_path, start)
                model.compile(optimizer = optimizer, run_eagerly = config.eager_execution)
                append_time_ram_stamp_to_file(f"Training:model.compile() end   ", config.bench_path, start)

                # fit model
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Training:model.fit() start ", config.bench_path, start)
                history = model.fit(data_set, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
                append_time_ram_stamp_to_file(f"Training:model.fit() end   ", config.bench_path, start)
        else:
            if not config.ll_growth_factor:
                model, cgp_hmm_layer = make_model(config)
                model.summary()

                # compile model
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Training:model.compile() start ", config.bench_path, start)
                model.compile(optimizer = optimizer, run_eagerly = config.eager_execution)
                append_time_ram_stamp_to_file(f"Training:model.compile() end ", config.bench_path, start)

                # fit model
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Training:model.fit() start ", config.bench_path, start)
                history = model.fit(data_set, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
                append_time_ram_stamp_to_file(f"Training:model.fit() end ", config.bench_path, start)

            if config.ll_growth_factor:
                dir_path = f"{config.current_run_dir}/after_fit_para"
                if not os.path.exists(dir_path):
                        os.system(f"mkdir -p {dir_path}")
                for current_epoch in range(config.epochs):
                    if config.ll_growth_factor * (current_epoch + 1) >= 1:
                        break
                    if current_epoch != 0:
                        config.init_weights_from = dir_path
                    model, cgp_hmm_layer = make_model(config, current_epoch = current_epoch +1)
                    if current_epoch == 0:
                        model.summary()

                    skipeed_data_set = data_set.skip(current_epoch)

                    # compile model
                    start = time.perf_counter()
                    append_time_ram_stamp_to_file(f"Training:model.compile() start ", config.bench_path, start)
                    model.compile(optimizer = optimizer, run_eagerly = config.eager_execution)
                    append_time_ram_stamp_to_file(f"Training:model.compile() end ", config.bench_path, start)

                    # fit model
                    start = time.perf_counter()
                    append_time_ram_stamp_to_file(f"Training:model.fit() start ", config.bench_path, start)
                    history = model.fit(skipeed_data_set, epochs = 1, steps_per_epoch=config.steps_per_epoch, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
                    append_time_ram_stamp_to_file(f"Training:model.fit() end ", config.bench_path, start)

                    model.get_layer(f"cgp_hmm_layer{'_' + str(current_epoch) if current_epoch > 0 else ''}").C.write_weights_to_file(dir_path)
                # likelihood growth reached a vaule of 1 so now
                # training is done with the full likelihood as per usual

                config.init_weights_from = dir_path
                model, cgp_hmm_layer = make_model(config, current_epoch)

                skipeed_data_set = data_set.skip(current_epoch)

                # compile model
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Training:model.compile() start ", config.bench_path, start)
                model.compile(optimizer = optimizer, run_eagerly = config.eager_execution)
                append_time_ram_stamp_to_file(f"Training:model.compile() end ", config.bench_path, start)

                # fit model
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Training:model.fit() start ", config.bench_path, start)
                history = model.fit(skipeed_data_set, epochs = config.epochs - current_epoch, steps_per_epoch=config.steps_per_epoch, callbacks = get_call_backs(config, model)) # with callbacks it is way slower
                append_time_ram_stamp_to_file(f"Training:model.fit() end ", config.bench_path, start)

                model.get_layer(f"cgp_hmm_layer{'_' + str(current_epoch) if current_epoch > 0 else ''}").C.write_weights_to_file(dir_path)

    return model, history
