import argparse
import Utility
import re
from Utility import run

class Config():

    def __init__(self, for_which_program):
        self.parser = argparse.ArgumentParser(description='Config module description')
        self.manuall_arg_lists = {"small_bench" : [], "main_programm" : []}
        if for_which_program == "small_bench":
            self.add_small_bench()
        if for_which_program == "main_programm":
            self.add_main_programm()

        self.parsed_args = self.parser.parse_args()

        self.add_attribtes()
        self.prepare_before_main_programm()
        self.determine_attributes()

        self.apply_args()

    def apply_args(self):
        import tensorflow as tf

        # dtype
        if self.dtype == tf.float64:
            policy = tf.keras.mixed_precision.Policy("float64")
            tf.keras.mixed_precision.set_global_policy(policy)
        # cpu vs gpu logging
        if self.cpu_gpu:
            tf.debugging.set_log_device_placement(True) # shows whether cpu or gpu is used

        # virtual gpus
        self.num_physical_gpus = len(tf.config.list_physical_devices('GPU'))
        assert self.num_physical_gpus == len(tf.config.experimental.list_physical_devices("GPU")), "different number of GPUs determined"
        if self.num_physical_gpus and self.split_gpu:
            tf.config.experimental.set_virtual_device_configuration(
                tf.config.experimental.list_physical_devices("GPU")[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512),
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            num_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

            print("Config: Using", num_gpu, "GPUs. device_lib.list_local_devices()")

            print("Config: printing local devices")
            for i, x in  enumerate(device_lib.list_local_devices()):
                print(i, x.name)

    def add_attribtes(self):
        import tensorflow as tf

        self.alphabet_size = 4
        self.write_return_sequnces = False

        self.bench_path = f"{self.src_path}/bench/{self.nCodons}codons/{self.call_type}call_type.log"
        self.fasta_path = f"{self.src_path}/output/{self.nCodons}codons/out.seqs.{self.nCodons}codons.fa"

        self.check_assert = not self.dont_check_assert
        self.generate_new_seqs = not self.dont_generate_new_seqs
        self.dtype = tf.float64 if self.dtype64 else tf.float32
        self.learning_rate = self.learning_rate if not self.no_learning else 0

        self.gen_len = 3 * self.nCodons

        self.seq_len = self.parsed_args.seq_len if self.parsed_args.seq_len else (self.nCodons * 3 + 6 + 2) * 2

        #                                     start and stop, i want at least one ig 3' and 5'
        assert self.seq_len >= self.gen_len + 6               + 2, f"self.seq_len ({self.seq_len}) < self.gen_len ({self.gen_len}) + 6 + 2"

    def prepare_before_main_programm(self):
        run(f"mkdir -p {self.src_path}/output/{self.nCodons}codons/")

        run(f"mkdir -p {self.src_path}/verbose")
        run(f"rm       {self.src_path}/verbose/{self.nCodons}codons.txt")

        run(f"rm {self.src_path}/{self.bench_path}")

    def determine_attributes(self):
        from Utility import get_state_id_description_list
        self.state_id_description_list = get_state_id_description_list(self.nCodons)

        from Utility import get_indices_for_config
        Utility.get_indices_for_config(self)
        # print("self.indices_for_B in determine_attributes in config =", self.indices_for_B)


    def add_arg_small_bench(self, *kwargs, type = None, help ="help", default = None, action = None, nargs = None):
        arg_name = kwargs[-1].strip("-")
        if action:
            self.parser.add_argument(*kwargs, action = action, help = help)
        else:
            self.parser.add_argument(*kwargs, type = type, default = default, help = help, nargs = nargs)
        self.manuall_arg_lists["small_bench"].append(arg_name)

    def add_arg_main(self, *kwargs, type = None, help ="help", default = None, action = None):
        arg_name = kwargs[-1].strip("-") , re.match("(-*)", kwargs[-1]).group(1)
        if action:
            self.parser.add_argument(*kwargs, action = action, help = help)
        else:
            self.parser.add_argument(*kwargs, type = type, default = default, help = help)
        self.manuall_arg_lists["main_programm"].append(arg_name)


    def print(self):
        s = "==========> config <==========\n"
        max_len = max([len(k[0]) for l in self.manuall_arg_lists.values() for k in l ])
        keys = [k for l in self.manuall_arg_lists.values() for k in l ]
        keys = sorted(keys)
        for i, key in enumerate(keys):
            s += key[0]
            if i % 5 == 0:
                s += "-" * (max_len - len(key[0]))
            else:
                s += " " * (max_len - len(key[0]))
            s += " = "
            s += str(self.__dict__[key[0]] if key[0] in self.__dict__ else self.parsed_args.__dict__[key[0]]) + "\n"
        s += "==========> config <==========\n"
        for key, value in self.__dict__.items():
            s += f"{key} = {str(value)[:50]}"
        s += "==========> config full <=========="

        print(s)


    def __getattr__(self, name):
        return self.parsed_args.__dict__[name]

    def add_small_bench(self):
        self.add_arg_small_bench('-r', '--range_codon', nargs='+', help='usage: < -r 1 10 > to run 1 2 3 4 5 6 7 8 9 10 codons')
        self.add_arg_small_bench('-tl', '--typesList', nargs="+", help='types of cell.call() that should be used')
        self.add_arg_small_bench('-cl', '--nCodonsList', nargs="+", help ='usage: < -c 10 20 50 > to run 10 20 50 codons')
        self.add_arg_small_bench('--repeat', type = int, default = 1, help ='repeat the main programm [repeat] times')
        self.add_arg_small_bench('--exit_on_nan', action='store_true', help ="exit_on_nan")

        self.add_main_programm()

    def add_main_programm(self):
        self.add_arg_main('-c', '--nCodons', type = int, default = 1, help='number of codons')
        self.add_arg_main('-t', '--call_type', type = int, default = 3, help='type of cell.call():  0:A;B sparse, 1:A dense, 2:B dense, 3:A;B dense, 4:fullmodel')
        self.add_arg_main('--order', type = int, default = 2, help = '[order] many preceeding emissions before the current one')
        self.add_arg_main('-p', '--src_path', default = ".", help='path to src')
        self.add_arg_main('--optimizer', default = "SGD", help = 'Adam, Adadelta, Adagrad, Adamax, Ftrl , Nadam, RMSprop, SGD [SDG]')
        self.add_arg_main('--epochs', default = 2, type = int, help = 'how many epochs [2]')
        self.add_arg_main('--steps_per_epoch', default = 4, type = int, help = 'how many steps (i think batches) per epoch [4] (bc #seqs=100 batch_size=32 -> every seq is used)')
        self.add_arg_main('--run_viterbi', action='store_true', help ="run_viterbi")


        # fine tune algo
        self.add_arg_main('--use_weights_for_consts', action='store_true', help ="use weights for transitions that become 1 after softmax")
        self.add_arg_main('-d', '--dtype64', action='store_true', help='using dytpe tf.float64')
        self.add_arg_main('--clip_gradient_by_value', help ="clip_gradient_by_values", type = float)
        self.add_arg_main('--learning_rate', help ="learning_rate", type = float, default = 0.01)
        self.add_arg_main('--no_learning', help ="learning_rate is set to 0", action='store_true')
        self.add_arg_main('--seq_len', type = int, help = 'lenght of output seqs before the optional stripping of flanks')
        self.add_arg_main('--use_simple_seq_gen', action='store_true', help ="use_simple_seq_gen (just random seqs) and not MSAgen")
        self.add_arg_main('-cd', '--coding_dist', type = float, default = 0.2, help='coding_dist for MSAgen')
        self.add_arg_main('-ncd', '--noncoding_dist', type = float, default = 0.4, help='noncoding_dist for MSAgen')
        self.add_arg_main('--dont_strip_flanks', action='store_true', help ="dont_strip_flanks ie all seqs have the same length")

        # hardware
        self.add_arg_main('--split_gpu', action='store_true', help ="split gpu into 2 logical devices")
        self.add_arg_main('--dont_use_gpu', action='store_true', help ="dont_use_gpu")

        # verbose
        self.add_arg_main('-v', '--verbose', default = 0, type = int, help ="verbose E,R, alpha, A, B to file, pass 1 for shapes, 2 for shapes and values")
        self.add_arg_main('-o', '--remove_verbose_at_batch_begin', action='store_true', help ="only_keep_verbose_of_last_batch")
        self.add_arg_main('-s', '--verbose_to_stdout', action='store_true', help ="verbose to stdout instead of to file")
        self.add_arg_main('--cpu_gpu', action='store_true', help ="print whether gpu or cpu is used")
        self.add_arg_main('--batch_begin_write_weights__layer_call_write_inputs', action='store_true', help ="batch_begin_write_weights__layer_call_write_inputs")
        self.add_arg_main('--get_gradient_of_first_batch', action='store_true', help ="get_gradient_of_first_batch")
        self.add_arg_main('--get_gradient_for_current_txt', action='store_true', help ="get_gradient_for_current_txt, previous run wrote IAB and inputbatch to file (via --batch_begin_write_weights__layer_call_write_inputs flag)-> get respective gradient")
        self.add_arg_main('--get_gradient_from_saved_model_weights', action='store_true', help ="get_gradient_from_saved_model_weights, previous run saved weights when passing --batch_begin_write_weights__layer_call_write_inputs")
        self.add_arg_main('--assert_summarize', type = int, default = -1, help = 'assert_summarize [-1]')

        # debugging
        self.add_arg_main('-b', '--exit_after_first_batch', action = 'store_true', help ="exit after first batch, you may use this when verbose is True in cell.call()")
        self.add_arg_main('-n', '--exit_after_loglik_is_nan', action='store_true', help ="exit_after_loglik_is_nan, you may use this when verbose is True in cell.call()")
        self.add_arg_main('--dont_generate_new_seqs', action='store_true', help ="dont_generate_new_seqs, but use the ones that were created before")
        self.add_arg_main('--manual_traning_loop', action='store_true', help ="manual_traning_loop")
        self.add_arg_main('--dont_check_assert', action='store_true', help ="dont_check_assert")
        self.add_arg_main('--run_eagerly', action='store_true', help ='run model.fit in eager execution')

    def get_args_as_str(self, for_what): # for_what \in {"small_bench", "main_programm"}
        s = ""
        for key in self.manuall_arg_lists[for_what]:
            value = self.__dict__[key[0]] if key[0] in self.__dict__ else self.parsed_args.__dict__[key[0]]
            if key[0] == "call_type":
                print("value =", value, type(value))
            if type(value) == bool:
                s += key[1] + key[0] + " " if value else ""
            else:
                s += key[1] + key[0]  + " " + str(value) + " " if value != None else ""
        return(s)
