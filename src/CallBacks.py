from Utility import append_time_ram_stamp_to_file
import tensorflow as tf
import time
import os
import time

def get_call_backs(config, model):

    class write_time_ram_epoch_start_callback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs = None):
            append_time_ram_stamp_to_file("epoch_begin_callback", config.bench_path)

    class write_time_ram_epoch_end_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = None):
            append_time_ram_stamp_to_file("epoch_end_callback", config.bench_path)

    class exit_after_first_batch(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs = None):
            exit(1)

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
        def on_train_batch_end(self, batch, logs = None):
            tf.print("on_train_batch_end_batch_id (1 based index) =", batch + 1)

    class write_inital_parameters_to_file(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 0:
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Callbacks.write_initial_para start", config.bench_path, start)

                path = f"{config.current_run_dir}/before_fit_para/"
                try:
                    cell = model.get_layer("cgp_hmm_layer").C
                    cell.write_weights_to_file(path)
                except:
                    pass

                append_time_ram_stamp_to_file(f"Callbacks.write_initial_para end", config.bench_path, start)

    class save_best_weights(tf.keras.callbacks.Callback):
        def __init__(self):
            super(save_best_weights, self).__init__()
            self.best_weights = None
            self.lowest_loss = float('inf')

        def on_epoch_end(self, epoch, logs=None):
            current_loss = logs.get('loss')
            if current_loss < self.lowest_loss:
                self.lowest_loss = current_loss
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Callbacks.write_best_weights start", config.bench_path, start)

                path = f"{config.current_run_dir}/after_fit_para/"
                try:
                    cell = model.get_layer("cgp_hmm_layer").C
                    cell.write_weights_to_file(path)
                except:
                    pass

                append_time_ram_stamp_to_file(f"Callbacks.write_best_weights end", config.bench_path, start)

    class write_initial_matrices_to_file(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs = None):
            if epoch == 0:
                start = time.perf_counter()
                append_time_ram_stamp_to_file(f"Callbacks.write_initial_matrices start", config.bench_path, start)
                # weights
                path = f"{config.current_run_dir}/before_fit_para/"
                cell = model.get_layer("cgp_hmm_layer").C

                # json format dense matrices
                config.model.I_as_dense_to_json_file(f"{path}/I.json")
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
            # if ll_growth_factor != 0 then dont use early stopping
            # if the factor of the likelihood is still growing
            if config.ll_growth_factor * (epoch + 1) < 1:
                return
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


    callbacks = [write_time_ram_epoch_start_callback(),
                 write_time_ram_epoch_end_callback()]

    if config.print_batch_id:
        callbacks += [batch_id_at_begin()]
    if config.exit_after_first_batch:
        callbacks += [exit_after_first_batch()]
    if config.exit_after_loglik_is_nan :
        callbacks  += [tf.keras.callbacks.TerminateOnNaN()]

    if config.check_assert:
        callbacks += [assert_kernels_at_batch_begin()]
    if config.remove_verbose_at_batch_begin:
        callbacks += [remove_verbose_at_batch_begin()]
    if config.init_png:
        callbacks += [init_png()]

    callbacks += [write_inital_parameters_to_file()]
    callbacks += [RelativeEarlyStopping()]
    callbacks += [save_best_weights()]

    # callbacks += [tensorboard_callback]

    # callbacks += [get_the_gradient()]

    return callbacks
