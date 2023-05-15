import Utility
import tensorflow as tf
import time
import os
from Utility import append_time_ram_stamp_to_file
import time

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
            #     Utility.transform_verbose_txt_to_csv(f"{config.out_path}/verbose/{config.nCodons}codons.txt", config.nCodons)
            exit(1)

    class exit_after_loglik_is_nan(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            tf.print("logs", logs)
            try:
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(logs["loss"])), [logs["loss"]], name = "logs['loss']_batch_begin", summarize = config.assert_summarize)
            except:
                print("key error in callback begin: exit_after_loglik_is_nan")
        def on_train_batch_end(self, batch, logs = None):
            try:
                tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(logs["loss"])), [logs["loss"]], name = "logs['loss']_batch_end", summarize = config.assert_summarize)
            except:
                print("key error in callback end: exit_after_loglik_is_nan")
    class remove_verbose_at_batch_begin(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            os.system(f"rm {config.out_path}/verbose/{config.nCodons}codons.txt")

    class assert_kernels_at_batch_begin(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            ik, ak, bk = model.get_weights()

            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(ik)), [ik,ak,bk], name = "I_kernel_is_nan", summarize = config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(ak)), [ik,ak,bk], name = "A_kernel_is_nan", summarize = config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(bk)), [ik,ak,bk], name = "B_kernel_is_nan", summarize = config.assert_summarize)

    class batch_id_at_begin(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs = None):
            tf.print("on_train_batch_begin_batch_id (1 based index) =", batch + 1)

    class write_inital_parameters_to_file(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 0:
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Callbacks.write_initial_para start", config.bench_path, start)

                path = f"{config.current_run_dir}/before_fit_para/"
                cell = model.get_layer("cgp_hmm_layer").C
                cell.write_weights_to_file(path)

                append_time_ram_stamp_to_file(f"Callbacks.write_initial_para end", config.bench_path, start)



    class write_initial_matrices_to_file(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs = None):
            if epoch == 0:
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Callbacks.write_initial_matrices start", config.bench_path, start)
                # weights
                path = f"{config.current_run_dir}/before_fit_para/"
                cell = model.get_layer("cgp_hmm_layer").C

                # json format dense matrices
                config.model.I_as_dense_to_json_file(f"{path}/I.json", cell.I_kernel)
                config.model.A_as_dense_to_json_file(f"{path}/A.json", cell.A_kernel)
                config.model.B_as_dense_to_json_file(f"{path}/B.json", cell.B_kernel)

                append_time_ram_stamp_to_file(f"Callbacks.write_initial_matrices end", config.bench_path, start)

    class init_png(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs = None):
            if epoch == 0:
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Callbacks.init_png start", config.bench_path, start)
                cell = model.get_layer("cgp_hmm_layer").C
                config.model.export_to_dot_and_png(cell.A_kernel, cell.B_kernel, name = "before_fit", to_png = config.nCodons < 10)
                append_time_ram_stamp_to_file(f"Callbacks.init_png end", config.bench_path, start)

    # class batch_begin_write_weights__layer_call_write_inputs(tf.keras.callbacks.Callback):
    #     # layer call write inputs -> the code is located in layer.py
    #     # and is activated by same flag as this callback
    #     def on_train_batch_begin(self, batch, logs = None):
    #         # TODO: why can i access config here?

    #         if batch == 0:
    #             path = f"{config.out_path}/output/{config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/"

    #             model.get_layer("cgp_hmm_layer").C.write_weights_to_file(path)

    #         # if config.check_for_zeros:
    #         #     Utility.find_indices_in_sparse_A_that_are_zero(config = config, \
    #         #                                                    I_dense = I, \
    #         #                                                    A_dense = A, \
    #         #                                                    B_dense = B)

    #         model.save_weights(f"{config.out_path}/output/{config.nCodons}codons/batch_begin_write_weights__layer_call_write_inputs/weights.h5", overwrite=True, save_format="h5") #todo also try tf as save format


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


    class RelativeEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self, monitor = 'loss', min_delta_prop = 0.0001, patience = 5):
            super().__init__()
            self.monitor = monitor
            self.min_delta_prop = min_delta_prop
            self.patience = patience
            self.best_value = float('inf')
            self.wait = 0

        def on_epoch_end(self, epoch, logs=None):
            current_value = logs.get(self.monitor)
            if current_value is None:
                return

            if current_value < self.best_value * (1 - self.min_delta_prop):
                self.best_value = current_value
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    print(f"\nEpoch {epoch}: RelativeEarlyStopping triggered!")


    callbacks = []
    callbacks = [write_time_ram_epoch_start_callback(),
                 write_time_ram_epoch_end_callback()]

    if config.print_batch_id:
        callbacks += [batch_id_at_begin()]
    if config.exit_after_first_batch:
        callbacks += [exit_after_first_batch()]
    if config.exit_after_loglik_is_nan :
        # callbacks += [exit_after_loglik_is_nan()]
        callbacks  += [tf.keras.callbacks.TerminateOnNaN()]

    if config.check_assert:
        callbacks += [assert_kernels_at_batch_begin()]
    if config.remove_verbose_at_batch_begin:
        callbacks += [remove_verbose_at_batch_begin()]
    # if config.batch_begin_write_weights__layer_call_write_inputs:
    #     callbacks += [batch_begin_write_weights__layer_call_write_inputs()]
    if config.write_initial_matrices_to_file:
        callbacks += [write_initial_matrices_to_file()]
    if config.init_png:
        callbacks += [init_png()]

    callbacks += [write_inital_parameters_to_file()]
    callbacks += [RelativeEarlyStopping()]

    # callbacks += [tensorboard_callback]

    # callbacks += [get_the_gradient()]

    return callbacks
