import time
import argparse
import re
import json
import os
import numpy as np



def get_exon_from_fasta_path(fasta_path) -> str:
    try:
        return re.search("exon_(.+?_\d+_\d+)", fasta_path).group(1)
    except:
        return "no-exon-name-found"

def get_date_str() -> str:
    from datetime import datetime
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M")
    return date_string

def get_dir_path_from_fasta_nCodons_and_out_path(out_path, nCodons, fasta_path = None) -> str:
    random_id = "".join(np.random.choice(list("qwertzuioplkjhgfdsayxcvbm0123456789"), size=4))
    if fasta_path == None or fasta_path == -1:
        return os.path.join(out_path, f"{get_date_str()}_{random_id}_generated_{nCodons}")
    return os.path.join(out_path,f"{get_date_str()}_{random_id}_{get_exon_from_fasta_path(fasta_path)}_{nCodons}")


class Config():

    def __init__(self):
        self.args = {}
        self.det_args = {}


    def init_for_matrix_diff(self):
        self.get_args_for_add_str_to_matrices()
        self.add_args_from_parser()

        self.load_training_args()
        self.set_current_run_dir(use_existing = True) # from viterbi.args
        self.determine_attributes_that_only_depend_on_args(dont_multiply_model_size_factor = True)

        self.get_model(only_prepare = True)

    def init_for_convert_kernel(self):
        self.get_args_for_add_str_to_matrices()
        self.add_args_from_parser()

        self.load_training_args()
        self.set_current_run_dir(use_existing = True) # from viterbi.args
        self.determine_attributes_that_only_depend_on_args(dont_multiply_model_size_factor = True)

        self.get_model()

    def init_for_add_str_to_matrices(self):
        self.get_args_for_add_str_to_matrices()
        self.add_args_from_parser()

        self.load_training_args()
        self.set_current_run_dir(use_existing = True)
        self.determine_attributes_that_only_depend_on_args(dont_multiply_model_size_factor = True)

        self.get_model(only_prepare = True)

    def init_for_get_dot_and_png(self):
        self.get_args_for_get_dot_and_png()
        self.add_args_from_parser()

        self.load_training_args()
        self.set_current_run_dir(use_existing = True)
        self.determine_attributes_that_only_depend_on_args(dont_multiply_model_size_factor = True)
        self.asserts()

        self.get_model()

    def init_for_viterbi(self):
        self.get_args_for_viterbi()
        self.add_args_from_parser()

        self.load_training_args()
        self.set_current_run_dir(use_existing = True)
        self.determine_attributes_that_only_depend_on_args(dont_multiply_model_size_factor = True)

        self.asserts()
        self.get_model()


    def init_for_training(self):
        self.get_args_for_training()
        self.add_args_from_parser()
        self.set_current_run_dir()
        self.determine_attributes_that_only_depend_on_args()
        self.write_passed_args_to_file()
        self.asserts()

        if not os.path.exists(self.current_run_dir): # this is set in add_attributes()
            os.system(f"mkdir -p {self.current_run_dir}")

        self.apply_args()
        self.get_model()


    def add_args_from_parser(self):
        parsed_args = self.parser.parse_args()
        self.args = parsed_args.__dict__

    def load_training_args(self):
        args_json_path = f"{self.parent_input_dir}/passed_args.json"
        with open(args_json_path, "r") as file:
            loaded_training_args = json.load(file)
        for key, value in loaded_training_args.items():
            if key not in self.args:
                self.args[key] = value

################################################################################
    def get_current_run_dir(self, use_existing = False) -> str:
        if use_existing:
            return self.parent_input_dir

        if self.passed_current_run_dir: # if main is called from multi_run
            return self.passed_current_run_dir

        return get_dir_path_from_fasta_nCodons_and_out_path(self.out_path, self.nCodons, self.fasta_path)


    def set_current_run_dir(self, use_existing = False):
        # should be run before determine_attributes_that_only_depend_on_args
        self.det_args["current_run_dir"] = self.get_current_run_dir(use_existing)
################################################################################
    def determine_attributes_that_only_depend_on_args(self, dont_multiply_model_size_factor = False):
        if not dont_multiply_model_size_factor:
            self.args["nCodons"] = int(self.nCodons * self.model_size_factor)

        self.det_args["alphabet_size"] = 4
        self.det_args["order"] = 2
        self.det_args["write_return_sequnces"] = False

        self.det_args["bench_path"] = f"{self.current_run_dir}/bench.log"

        self.det_args["dtype"] = np.float64 if self.dtype64 else np.float32
        self.det_args["learning_rate"] = self.learning_rate if not self.no_learning else 0

        self.det_args["A_is_dense"] = self.AB[0] == "d"
        self.det_args["A_is_sparse"] = not self.A_is_dense
        self.det_args["B_is_dense"] = self.AB[1] == "d"
        self.det_args["B_is_sparse"] = not self.B_is_dense

        if self.global_log_epsilon:
            self.det_args["E_epsilon"] = self.global_log_epsilon
            self.det_args["R_epsilon"] = self.global_log_epsilon
            self.det_args["l_epsilon"] = self.global_log_epsilon
            self.det_args["log_prior_epsilon"] = self.global_log_epsilon

################################################################################
################################################################################
    def get_model(self, only_prepare = False) -> None:
        from Utility import append_time_ram_stamp_to_file
        # split into prepare model and build
        start = time.perf_counter()
        append_time_ram_stamp_to_file(f"Config.get_model() start", self.bench_path, start)

        from CGP_HMM import CGP_HMM
        self.model = CGP_HMM(self)

        self.model.prepare_model()
        if not only_prepare:
            self.model.make_model()

        append_time_ram_stamp_to_file(f"Config.get_model() end", self.bench_path, start)
################################################################################
################################################################################
    def write_passed_args_to_file(self, dir_path = None):

        try:
            assert self.determine_attributes_that_only_depend_on_args_was_run, "determine_attributes_that_only_depend_on_args_was_not_run_before_write_args_to_file"
        except:
            print("determine_attributes_that_only_depend_on_args_was_not_run_before_write_args_to_file")

        if dir_path is None:
            dir_path = self.current_run_dir
        if not os.path.exists(dir_path):
            os.system(f"mkdir -p {dir_path}")

        out_path = f"{dir_path}/passed_args.json"
        print("writing cfg to file:", out_path)
        with open(out_path, "w") as file:
            json.dump(self.args, file)


    def __getattr__(self, name):
        # det_args have higher priority
        try:
            return self.det_args[name]
        except:
            try:
                return self.args[name]
            except:
                print(f"config.{name} not defined")
                exit(1)

    def print(self):
        print("==============> printing args <================================")
        print("args:")
        for key, value in sorted(self.args.items()):
            print(key, value)

        print()
        print("det_args:")
        for key, value in sorted(self.det_args.items()):
            print(key, value)

        print("==============> printing args done <===========================")

################################################################################
################################################################################
################################################################################
    def apply_args(self):
        import tensorflow as tf
        if self.dtype == np.float64:
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

        import logging
        log_file = f"{self.current_run_dir}/tensorflow_infos_and_warning.log"
        file_handler = logging.FileHandler(log_file)

        # Create a filter that only captures WARNING and INFO log records
        class TensorFlowFilter(logging.Filter):
            def filter(self, record):
                return 'tensorflow' in record.name.lower() and \
                    (record.levelno == logging.WARNING or record.levelno == logging.INFO)
        filter = TensorFlowFilter()
        file_handler.addFilter(filter)

        # Create a formatter for the file handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the TensorFlow logger
        logger = tf.get_logger()
        logger.addHandler(file_handler)

################################################################################
################################################################################
################################################################################
    def asserts(self):

        if not os.path.exists(self.fasta_path):
            print("fasta not found at", self.fasta_path)
            exit(1)

        if not os.path.exists(self.viterbi_exe):
            print("viterbi_exe not found at", self.viterbi_exe)
            exit(1)

        # try to run viterbi and check if permission is not denied
        if not os.access(self.viterbi_exe, os.X_OK):
            print("viterbi_exe is not executable for user. Use 'chmod u+x Viterbi' to make it executable")
            exit(1)

        if not os.path.exists(self.fasta_path):
            print("fasta not found at", self.fasta_path)
            exit(1)

        if os.path.isdir(self.fasta_path):
            print("fasta_path is a dir, but should be a fasta file")
            exit(1)


        assert self.batch_size > 0, "batch_size must be greater than 0"
        assert self.AB in ["dd", "ds", "sd", "ss"], "-AB must be in ['dd', 'ds', 'sd', 'ss']"


        if self.manual_forward:
            assert self.AB == "dd", "manula forward only works with dense matrices, so pass -AB dd"

        if (self.priorA or self.priorB):
            assert self.nCodons > 1, "when using prior and internal model you must use more than 1 codon since for 1 codon there are no priors for the transition matrix"


        assert self.flatten_B_init >= 0 and self.flatten_B_init <= 1, f"self.flatten_B_init must be in [0,1] but is {self.flatten_B_init}"

################################################################################
################################################################################
################################################################################

    def get_args_for_training(self):
        self.parser = argparse.ArgumentParser(description='args for cgphmm')

        self.parser.add_argument('-c', '--nCodons', type = int, default = 2, help='Model size n, ie number of codons.')
        self.parser.add_argument('-AB', default = 'sd', help = '[sd (default), ds, sd, ss] specify the sparse or denseness of A and B.')

        self.parser.add_argument('--fasta_path', required=1, help = 'Path to fasta file where the training seqs are')
        self.parser.add_argument('--out_path', default = "../../cgp_output", help='Path to otput dir')
        self.parser.add_argument('--passed_current_run_dir', help='If nothing is passed, then on is generated.')
        self.parser.add_argument('--primates_path', type = str, default = "../../data/phylogenetic_data/primates.lst", help = 'Path to primates.list.')
        self.parser.add_argument('--viterbi_exe', default = '../viterbi_cc/Viterbi', help = 'Path to c++ Viterbi executable.')
        self.parser.add_argument('--prior_path', default = "../data/human_priors/", help = 'Path to the directory containing exon and intron.pbl.')
        self.parser.add_argument('--diverse_trees_path', default = "../../data/phylogenetic_data/max_diverse_subtrees", help = 'Path to directory containin the files that contain the leaves of diverse species.')

        # learning
        self.parser.add_argument('--optimizer', default = "Adam", help = 'Adam, SGD [Adam]')
        self.parser.add_argument('--epochs', default = 2, type = int, help = 'How many epochs [2]')
        self.parser.add_argument('--steps_per_epoch', default = 0, type = int, help = 'If zero, the number of steps is set such that all sequences are seen once per epoch.')
        self.parser.add_argument('-d', '--dtype64', action='store_true', help='Switch to using dytpe float64')
        self.parser.add_argument('--batch_size', type = int, default = 16, help = 'Batch size [16]')
        self.parser.add_argument('--no_learning', action='store_true', help ="Learning rate is set to 0")
        self.parser.add_argument('--learning_rate', type = float, default = 0.05, help ="Learning rate [0.05]")
        self.parser.add_argument('--clip_gradient_by_value', type = float, help ="Clip gradient by value [0]")
        self.parser.add_argument('--bucket_by_seq_len', action = 'store_true', help = 'Bucket seqs by their lengths.')
        self.parser.add_argument('--ll_growth_factor', type = float, default = 0, help = 'If 1, likelihood is used as usual, but if for example 0.1. then likelihood in first epoch is scaled by .1 then in second by 0.2 ... until 1.0')
        self.parser.add_argument('--model_size_factor', type = float, default = 1, help = 'Change model length by this factor.')
        self.parser.add_argument('--dont_shuffle_seqs', action = 'store_true', help = 'Dont shuffle sequences before creating batches')
        self.parser.add_argument('--global_log_epsilon', type = float, default = 1e-20, help = 'Sets all epsilon values that are used in the Farward algorithm to this to this value [1e-20]')
        self.parser.add_argument('--log_prior_epsilon', type = float, default = 0, help = 'Epsilon for prior calculation with log [0]')
        self.parser.add_argument('--l_epsilon', type = float, default = 0, help = 'Epsilon for the likelihod calculation with log [0]')
        self.parser.add_argument('--R_epsilon', type = float, default = 0, help = 'Epsilon for R calculation with log [0]')
        self.parser.add_argument('--E_epsilon', type = float, default = 0, help = 'Epsilon for E calculation with log [0]')

        self.parser.add_argument('--only_primates', action = "store_true", help = 'Only use primates to train.')
        self.parser.add_argument('--only_human', action = "store_true", help = 'Only use human to train.')
        self.parser.add_argument('--only_diverse', action = "store_true", help = 'Only use max_diverse_set_same_size_as_primates to train.')

        # model
        self.parser.add_argument('--akzeptor_pattern_len', type = int, default = 5, help = 'akzeptor_pattern_len before AG')
        self.parser.add_argument('--donor_pattern_len', type = int, default = 5, help = 'donor_pattern_len after GT')

        self.parser.add_argument('--left_intron_const', type = int, default = 0, help = 'Dont learn transition upstream intron loop.')
        self.parser.add_argument('--right_intron_const', type = int, default = 0, help = 'Dont learn transition downstream intron loop.')
        self.parser.add_argument('--exon_skip_const', action = 'store_true', help = 'Dont learn transition from upstream intron to downstream intron.')
        # prior
        self.parser.add_argument('--priorA', type = float, default = 0, help = 'Use prior for A and scale the alphas (= concentration).')
        self.parser.add_argument('--priorB', type = float, default = 0, help = 'Use prior for B and scale the alphas (= concentration).')
        self.parser.add_argument('--ass_start', type = int, default = 7, help = 'Length of prior pattern before AG ASS splice site, needs to be adjusted to the input .pbl files.')
        self.parser.add_argument('--ass_end', type = int, default = 2, help = 'Length of prior pattern after AG ASS splice site, needs to be adjusted to the input .pbl files.')
        self.parser.add_argument('--dss_start', type = int, default = 1, help = 'Length of prior pattern before GT DSS splice site, needs to be adjusted to the input .pbl files.')
        self.parser.add_argument('--dss_end', type = int, default = 5, help = 'Length of prior pattern after GT DSS splice site, needs to be adjusted to the input .pbl files.')

        # prior and initial weights
        self.parser.add_argument('--use_thesis_weights', type = bool, default = True, help = 'Initialize weights like described in my thesis.')
        self.parser.add_argument('--single_high_prob_kernel', type = float, default = 3, help = 'This sets th weight for the transitions to the next codon, and leaven a codon insertion. In my thesis the default [3] was used.')
        self.parser.add_argument('--diminishing_factor', type = float, default = 4, help = 'Make delets more expensive [4].')
        self.parser.add_argument('--left_intron_init_weight', type = float, default = 4.35, help = 'Initial weight for the transition upstream intron loop.')
        self.parser.add_argument('--right_intron_init_weight', type = float, default = 4, help = 'Initial weight for the transition downstream intron loop.')
        self.parser.add_argument('--exon_skip_init_weight', type = float, default = -1, help = 'Initial weight for the transition from upstream intron to downstream intron.')
        self.parser.add_argument('--flatten_B_init', type = float, default = 0, help = 'Flatten the initial parameters of B, ie let c = flatten_B_init: priorB * c + uniform * (1-c).')
        self.parser.add_argument('--use_constant_initializer', action='store_true', help = 'Initialize weights with all ones.')
        self.parser.add_argument('--init_weights_from', help = 'Initialize weights from directory that contains I/A/B_kernel.json. (I is currently just []).')

        # hardware
        self.parser.add_argument('--split_gpu', action='store_true', help ="Split GPU into 2 logical devices.")
        self.parser.add_argument('--dont_use_mirrored_strategy', action='store_true', help ="Dont use mirrored strategy.")

        # verbose
        self.parser.add_argument('--verbose', default = 0, type = int, help ="Verbose E, R, alpha, A, B. Pass 1 for shapes, 2 for shapes and values.")
        self.parser.add_argument('--remove_verbose_at_batch_begin', action='store_true', help ="Only keeps the verbose at the latest batch.")
        self.parser.add_argument('--print_to_file', action='store_true', help = 'Print verbose to file instead of stdout.')
        self.parser.add_argument('--cpu_gpu', action='store_true', help ="Print whether GPU or CPU is used")
        self.parser.add_argument('--check_assert', action='store_true', help ="Check asserts in Forward loop.")
        self.parser.add_argument('--assert_summarize', type = int, default = 5, help = 'Sets the summarize argument [5].')
        self.parser.add_argument('--print_batch_id', action='store_true', help = 'Prints the batch id via on_train_batch_begin callback.')
        self.parser.add_argument('--init_png', action='store_true', help = 'Create .dot and .png (if nCodons < 10) for initial parameters.')
        self.parser.add_argument('--after_fit_png', action='store_true', help = 'Create .dot and .png (if nCodons < 10) for initial parameters.')
        self.parser.add_argument('--trace_verbose', action = 'store_true', help = 'Actiave some print() calls to see if functions get retraced.')

        # debugging
        self.parser.add_argument('-b', '--exit_after_first_batch', action = 'store_true', help ="Exit after first batch.")
        self.parser.add_argument('-n', '--exit_after_loglik_is_nan', action='store_true', help ="Exit after loglik is NaN.")
        # self.parser.add_argument('--manual_training_loop', action='store_true', help ="manual_training_loop")
        # self.parser.add_argument('--alpha_i_gradient', type = int, default = -1, help = 'if --manual_training_loop is passed, then the gradient for alpha_i wrt the kernels is computed, if -2 is passed, i is set to n - 1, where n is the length of th seq')
        # self.parser.add_argument('--check_for_zeros', action='store_true', help = 'must be passed together with --batch, checks for zeros in parameters')
        # self.parser.add_argument('--autograph_verbose', action = 'store_true', help = 'set tf.autograph.set_verbosity(3, True)')
        self.parser.add_argument('--eager_execution', action='store_true', help ='Run model.fit in eager execution.')
        self.parser.add_argument('--manual_forward', action = 'store_true', help = 'gets mean likelihood of with manual loop')



    def get_args_for_viterbi(self):
        if self.parser == -1:
            self.parser = argparse.ArgumentParser(description='args for Viterbi.py')

        self.parser.add_argument('--only_first_seq', action = 'store_true', help = 'Run Viterbi only for the first sequence in the fasta file.')
        self.parser.add_argument('--parent_input_dir', help = 'Path to directory containing the passed_args.json of the training run and paratemeter directories used for Viterbi.')
        self.parser.add_argument('--in_viterbi_path', help = 'If Viterbi is already calculated, path to Viterbi file which is then written to the alignment.clw file.')
        self.parser.add_argument('--viterbi_threads', type = int, default = 1, help = 'How many threads for viterbi.cc.')
        self.parser.add_argument('--after_or_before', default = "a", help = 'Use matrices of after/before training.')
        self.parser.add_argument('--out_file_path', help = 'Path of outfile.')


    def get_args_for_get_dot_and_png(self):
        if self.parser == -1:
            self.parser = argparse.ArgumentParser(description='args for script get_args_for_get_dot_and_png')

        self.parser.add_argument('--parent_input_dir', help = 'Path to directory containing the passed_args.json of the training run and paratemeter directories used for matrix creation.')
        self.parser.add_argument('--for_initial_weights', action = 'store_true', help = 'Use the before training weights.')
        self.parser.add_argument('--png', action = 'store_true', help = 'Render .png from .dot.')

    def get_args_for_add_str_to_matrices(self):
        if self.parser == -1:
            self.parser = argparse.ArgumentParser(description='args for script get_args_for_add_str_to_matrices')

        self.parser.add_argument('--parent_input_dir', required = True, help = 'Path to directory containing the passed_args.json of the training run and paratemeter directories used for matrix creation.')