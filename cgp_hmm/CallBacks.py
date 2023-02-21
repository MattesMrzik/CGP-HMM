import Utility
from CgpHmmCell import CgpHmmCell
import tensorflow as tf
import json
import time
import numpy as np
import os

def get_call_backs(config, model):

    class write_time_ram_epoch_start_callback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs = None):
            Utility.append_time_ram_stamp_to_file("epoch_begin_callback", config.bench_path)

    class write_time_ram_epoch_end_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = None):
            # with open(f"{output_path}/callbackoutput_ram_end.txt", "a") as file:
            #     file.write(f"{process.memory_info().rss}\n")
                #                              oder vms     Virtual Memory Size
            Utility.append_time_ram_stamp_to_file("epoch_end_callback", config.bench_path)

    class exit_after_first_batch(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs = None):
            # das vielleicht rein ins callback, da ja exit und der code hier dann ja gar nicht mehr erreicht wird
            # if config.verbose and config.exit_after_first_batch:
            #     Utility.transform_verbose_txt_to_csv(f"{config.src_path}/verbose/{config.nCodons}codons.txt", config.nCodons)
            exit(1)

    class exit_after_loglik_is_nan(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            try:
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(logs["loglik"])), [logs["loglik"]], name = "logs['loglik']_batch_begin", summarize = config.assert_summarize)
            except:
                print("key error in callback: exit_after_loglik_is_nan")
        def on_train_batch_end(self, batch, logs = None):
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(logs["loglik"])), [logs["loglik"]], name = "logs['loglik']_batch_end", summarize = config.assert_summarize)

    class remove_verbose_at_batch_begin(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            os.system(f"rm {config.src_path}/verbose/{config.nCodons}codons.txt")

    class assert_kernels_at_batch_begin(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            ik, ak, bk = model.get_weights()

            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(ik)), [ik,ak,bk], name = "I_kernel_is_nan", summarize = config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(ak)), [ik,ak,bk], name = "A_kernel_is_nan", summarize = config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(bk)), [ik,ak,bk], name = "B_kernel_is_nan", summarize = config.assert_summarize)

    class batch_id_at_begin(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            tf.print("on_train_batch_begin_batch_id (1 based index) =", batch + 1)

    class write_initial_weights_to_file(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs = None):
            if epoch == 0:
                path = f"{config.src_path}/output/{config.nCodons}codons/initial_weights_from_callback/"
                model.get_layer("cgp_hmm_layer").C.write_weights_to_file(path)


    class batch_begin_write_weights__layer_call_write_inputs(tf.keras.callbacks.Callback):
        # layer call write inputs -> the code is located in layer.py
        # and is activated by same flag as this callback
        def on_train_batch_begin(self, batch, logs = None):
            # TODO: why can i access config here?

            if batch == 0:
                path = f"{config.src_path}/output/{config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/"

                model.get_layer("cgp_hmm_layer").C.write_weights_to_file(path)

            if config.check_for_zeros:
                Utility.find_indices_in_sparse_A_that_are_zero(config = config, \
                                                               I_dense = I, \
                                                               A_dense = A, \
                                                               B_dense = B)

            model.save_weights(f"{config.src_path}/output/{config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/current_weights", overwrite=True, save_format="h5") #todo also try tf as save format


    class get_the_gradient(tf.keras.callbacks.Callback):

        # coulndt get this to work

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
            pass # not used

        # checkpoint_path = "training_1/cp.ckpt"
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                                  save_weights_only=True,
        #                                                  verbose=1)
    import datetime
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    callbacks = []
    callbacks = [write_time_ram_epoch_start_callback(),
                 write_time_ram_epoch_end_callback()]

    if config.print_batch_id:
        callbacks += [batch_id_at_begin()]
    if config.exit_after_first_batch:
        callbacks += [exit_after_first_batch()]
    if config.exit_after_loglik_is_nan:
        callbacks += [exit_after_loglik_is_nan()]
    if config.check_assert:
        callbacks += [assert_kernels_at_batch_begin()]
    if config.remove_verbose_at_batch_begin:
        callbacks += [remove_verbose_at_batch_begin()]
    if config.batch_begin_write_weights__layer_call_write_inputs:
        callbacks += [batch_begin_write_weights__layer_call_write_inputs()]

    # callbacks += [tensorboard_callback]

    # callbacks += [get_the_gradient()]

    return callbacks
