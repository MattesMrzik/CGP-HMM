#!/usr/bin/env python3
import tensorflow as tf
import time
from random import randint


from Utility import append_time_ram_stamp_to_file

from CgpHmmCell import CgpHmmCell

class CgpHmmLayer(tf.keras.layers.Layer):
    def __init__(self, config, current_epoch = None):
        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(f"Layer.init() start {run_id}", config.bench_path, start)
        super(CgpHmmLayer, self).__init__()
        self.nCodons = config.nCodons
        self.config = config
        if current_epoch is not None:
            self.current_epoch = current_epoch

        append_time_ram_stamp_to_file(f"Layer.init() end  {run_id}", self.config.bench_path, start)

    def build(self, inputs):

        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(f"Layer.build() start {run_id}", self.config.bench_path, start)

        self.C = CgpHmmCell(self.config) # init

        self.F = tf.keras.layers.RNN(self.C, return_state = True) # F = forward ie the chain of cells C
        append_time_ram_stamp_to_file(f"Layer.build() end   {run_id}", self.config.bench_path, start)

    def call(self, inputs, training = False): # shape of inputs is None = batch, None = seqlen, 126 = emissions_size
        # try:
        #     tf.print("hash values of inputs, A, B")
        #     for tensor in [inputs, self.C.A_kernel, self.C.B_kernel]:
        #         tensor_serialized = tf.io.serialize_tensor(tensor)
        #         hash_value = tf.strings.to_hash_bucket_fast(tensor_serialized, num_buckets=1000)
        #         tf.print(hash_value, end = ", ")
        #     tf.print("")
        # except:
        #     pass

        # tf.print(f"inputs {inputs}")
        if self.config.trace_verbose:
            print("layer.call()")
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer call")
        # tf.print("~~~~~~~~~~~~~~~~~~~~~~~~~ layer call: tf")

        start = time.perf_counter()
        run_id = randint(0,100)
        append_time_ram_stamp_to_file(f"Layer.call() start {run_id}", self.config.bench_path, start)

        # todo: felix macht auch nochmal a und b

        # result = self.F(inputs, initial_state = self.C.get_initial_state(batch_size=tf.shape(inputs)[0])) #  build and call of CgpHmmCell are called
        initial_state = self.C.get_initial_state(batch_size = tf.shape(inputs)[0])
        # print("initial_state =", initial_state)
        _, result_first_init_call = self.C(inputs[:,0], initial_state, init = True, training = training)

        inputs_with_out_first_emission = inputs[:,1:]

        result = self.F(inputs_with_out_first_emission, initial_state = result_first_init_call, training = training)
        alpha_state = result[0]
        loglik_state = result[1]

        # squeeze removes dimensions of size 1, ie shape (1,3,2,1) -> (3,2)
        loglik_mean = tf.reduce_mean(loglik_state)



        if self.config.check_assert:
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik_state)), [loglik_state],              name = "loglik_state_is_finite", summarize = self.config.assert_summarize)
            tf.debugging.Assert(tf.math.reduce_all(tf.math.is_finite(loglik_mean)),  [loglik_mean, loglik_state], name = "loglik_mean_is_finite",  summarize = self.config.assert_summarize)



        if self.config.priorB or self.config.priorA:
            if not self.config.prior_only:
                self.add_metric(loglik_mean, "loglik_mean")
                # normed likelihood by batchsize
                if self.config.ll_growth_factor and self.config.ll_growth_factor != 1:
                    self.add_metric(self.current_epoch, "self.current_epoch")
                    loglik_mean = loglik_mean * min(1, self.config.ll_growth_factor * self.current_epoch)
                    self.add_metric(loglik_mean, "scaled_loglik_mean")
                loss = - loglik_mean
            else:
                loss = 0


            m = self.config.nSeqs
            self.add_metric(m, "m")
            if self.config.priorA:
                A_prior = self.config.model.get_A_log_prior(self.C.A_kernel) / m
                self.add_metric(A_prior, "A_prior")
                loss -= A_prior
            if self.config.priorB:

                B_prior = self.config.model.get_B_log_prior(self.C.B_kernel) / m
                self.add_metric(B_prior, "B_prior")
                loss -= B_prior

            self.add_loss(tf.squeeze(loss))
        else:
            self.add_loss(tf.squeeze(-loglik_mean))

        if training:
            self.add_metric(tf.math.reduce_max(self.C.A_kernel),"tk_max")
            self.add_metric(tf.math.reduce_min(self.C.A_kernel),"tk_min")

            self.add_metric(tf.math.reduce_max(self.C.B_kernel),"ek_max")
            self.add_metric(tf.math.reduce_min(self.C.B_kernel),"ek_min")


        append_time_ram_stamp_to_file(f"Layer.call() end   {run_id}", self.config.bench_path, start)
        return loglik_state
