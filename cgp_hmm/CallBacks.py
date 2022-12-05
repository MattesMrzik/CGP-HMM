import Utility
from CgpHmmCell import CgpHmmCell
import tensorflow as tf
import json
import time
import numpy as np
import os

def get_call_backs(config, model):

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
            Utility.append_time_ram_stamp_to_file(0, "epoch_begin", config["bench_path"])

    class write_time_ram_epoch_end_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = None):
            # with open(f"{output_path}/callbackoutput_ram_end.txt", "a") as file:
            #     file.write(f"{process.memory_info().rss}\n")
                #                              oder vms     Virtual Memory Size
            Utility.append_time_ram_stamp_to_file(0, "epoch_end", config["bench_path"])

    class exit_after_first_batch(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs = None):
            # das vielleicht rein ins callback, da ja exit und der code hier dann ja gar nicht mehr erreicht wird
            if config["verbose"] and config["exit_after_first_batch"]:
                Utility.transform_verbose_txt_to_csv(f"{config['src_path']}/verbose/{config['nCodons']}codons.txt", config['nCodons'])
            exit(1)

    class exit_after_loglik_is_nan(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            try:
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(logs["loglik"])), [logs["loglik"]], name = "logs['loglik']_batch_begin", summarize = -1)
            except:
                print("key error in callback: exit_after_loglik_is_nan")
        def on_train_batch_end(self, batch, logs = None):
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(logs["loglik"])), [logs["loglik"]], name = "logs['loglik']_batch_end", summarize = -1)

    class remove_verbose_at_batch_begin(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            os.system(f"rm {config['src_path']}/verbose/{config['nCodons']}codons.txt")

    class batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            ik, ak, bk = model.get_weights()

            if config["type"] != 4:
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(ik)), [ik,ak,bk], name = "I_kernel_is_nan", summarize = -1)
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(ak)), [ik,ak,bk], name = "A_kernel_is_nan", summarize = -1)
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(bk)), [ik,ak,bk], name = "B_kernel_is_nan", summarize = -1)

                Utility.run(f"mkdir -p {config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/")
                os.system(f"rm {config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/*")

                ik = [float(x) for x in ik]
                ak = [float(x) for x in ak]
                bk = [float(x) for x in bk]

                with open(f"{config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_I.json", "w") as file:
                    json.dump(ik, file)
                with open(f"{config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_A.json", "w") as file:
                    json.dump(ak, file)
                with open(f"{config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_B.json", "w") as file:
                    json.dump(bk, file)

                with open(f"{config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_I_dense.json", "w") as file:
                    json.dump(model.get_layer("cgp_hmm_layer").C.I_dense.numpy().tolist(), file)
                with open(f"{config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_A_dense.json", "w") as file:
                    json.dump(model.get_layer("cgp_hmm_layer").C.A_dense.numpy().tolist(), file)
                with open(f"{config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_B_dense.json", "w") as file:
                    json.dump(model.get_layer("cgp_hmm_layer").C.B_dense.numpy().tolist(), file)

                model.save_weights(f"{config['src_path']}/output/{config['nCodons']}codons/batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs/current_weights", overwrite=True, save_format="h5") #todo also try tf as save format

            if config["tpye"] == 4:
                pass


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
            pass # not used

        # checkpoint_path = "training_1/cp.ckpt"
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                                  save_weights_only=True,
        #                                                  verbose=1)

    callbacks = []
    callbacks = [write_time_ram_epoch_start_callback(),
                 write_time_ram_epoch_end_callback()]

    if "exit_after_first_batch" in config and config["exit_after_first_batch"]:
        callbacks += [exit_after_first_batch()]
    if "exit_after_loglik_is_nan" in config and config["exit_after_loglik_is_nan"]:
        callbacks += [exit_after_loglik_is_nan()]
    if "only_keep_verbose_of_last_batch" in config and config["only_keep_verbose_of_last_batch"]:
        callbacks += [remove_verbose_at_batch_begin()]
    if "batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs" in config and config["batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs"]:
        callbacks += [batch_begin_exit_when_nan_and_write_weights__layer_call_write_inputs()]

    callbacks += [get_the_gradient()]

        # class my_callback(tf.keras.callbacks.Callback):
        #     def on_epoch_begin(self, epoch, logs = None):
        #         print("model.weights")
        #         print("A =", tf.nn.softmax(model.get_weights()[0]))


        # callbacks = [tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: print("A =", tf.nn.softmax(model.get_weights()[0])))]
    return callbacks
