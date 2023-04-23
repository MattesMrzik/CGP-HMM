import argparse
import re
import json
import os
import tensorflow as tf

class Config():

    def __init__(self):
        self.args = {}
        self.det_args = {}

    def init_for_viterbi(self):

        self.get_args_for_viterbi()
        self.add_args_from_parser()

        self.load_training_args()
        self.get_current_run_dir(for_viterbi = True) # from viterbi.args
        self.determine_attributes_that_only_depend_on_args()

        self.asserts()

        self.get_model(only_prepare = True)


    def init_for_training(self):
        self.get_args_for_training()
        self.add_args_from_parser()
        self.print()
        self.get_current_run_dir()
        self.determine_attributes_that_only_depend_on_args()
        self.write_passed_args_to_file()
        # TODO do i also want to write determined args to seperate file?
        # this might only be necessray if determine_attributes_that_only_depend_on_args()
        # changes during development
        self.asserts()

        if not os.path.exists(self.current_run_dir): # this is set in add_attributes()
            os.system(f"mkdir -p {self.current_run_dir}")

        self.apply_args()
        self.get_model()


    def add_args_from_parser(self):
        parsed_args = self.parser.parse_args()
        self.args = parsed_args.__dict__

    def load_training_args(self): # this is only when Viterbi is run on its own
        '''load most recent run dir if no viterbi in path is specified'''

        print("self.viterbi_parent_input_dir", self.viterbi_parent_input_dir)
        print("self.path_to_dir_where_most_recent_dir_is_selected", self.path_to_dir_where_most_recent_dir_is_selected)

        assert self.viterbi_parent_input_dir or self.path_to_dir_where_most_recent_dir_is_selected, "if run viterbi, args are loaded from file, so path must be specified"

        if self.viterbi_parent_input_dir is None:
            print("viterbi_parent_input_dir was not passed so try to get the most recent dir")
            # viterbi_parent_input_dir was not set, so try to retrieve the most recent output dir
            p = self.path_to_dir_where_most_recent_dir_is_selected
            regex = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})"
            subdirs = [os.path.join(p, subdir) for subdir in os.listdir(p) if (os.path.isdir(os.path.join(p, subdir)) and re.search(regex, subdir))]
            print("selecting most recent run dir from:", subdirs)
            # Define a regular expression to match the date and time in the directory name
            # Sort the subdirectories by their datetime
            sorted_subdirs = sorted(subdirs, key=lambda subdir: re.search(regex, subdir).group(1), reverse=True)
            # Get the path to the most recent subdirectory
            most_recent_subdir = sorted_subdirs[0]
            print("most_recent_subdir", most_recent_subdir)
            self.viterbi_parent_input_dir = most_recent_subdir

        args_json_path = f"{self.viterbi_parent_input_dir}/passed_args.json"
        with open(args_json_path, "r") as file:
            loaded_training_args = json.load(file)
        print("loaded_training_args", loaded_training_args)
        for key, value in loaded_training_args.items():
            self.args[key] = value

    def get_current_run_dir(self, for_viterbi = False):
        if for_viterbi:
            self.det_args["current_run_dir"] = self.viterbi_parent_input_dir
            return

        from datetime import datetime

        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d_%H-%M")
        fasta_name = "generated"
        if self.fasta_path:
            try:
                fasta_name = re.search("exon_().*?\d+_\d+)").group(1)
            except:
                fasta_name ="no-exon-name-found"

        self.det_args["current_run_dir"] = f"{self.out_path}/{date_string}_{fasta_name}_{self.nCodons}"

    def determine_attributes_that_only_depend_on_args(self):
        self.det_args["alphabet_size"] = 4
        self.det_args["write_return_sequnces"] = False

        self.det_args["bench_path"] = f"{self.current_run_dir}/bench.log"
        if not self.fasta_path:
            self.det_args["manual_passed_fasta"] = False
            self.det_args["fasta_path"] = f"{self.current_run_dir}/seqs.fa"
        else:
            self.det_args["manual_passed_fasta"] = True
            self.det_args["generate_new_seqs"] = False

        self.det_args["generate_new_seqs"] = not self.dont_generate_new_seqs
        self.det_args["dtype"] = tf.float64 if self.dtype64 else tf.float32
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


        self.det_args["gen_len"] = 3 * self.nCodons

        self.det_args["seq_len"] = self.seq_len if self.seq_len else ((self.nCodons * 3 + 6 + 2) * 2)

        print('self.det_args["seq_len"]', self.det_args["seq_len"])
        print('self.seq_len',  self.seq_len)

        #                                     start and stop, i want at least one ig 3' and 5'
        assert self.seq_len >= self.gen_len + 6               + 2, f"self.seq_len ({self.seq_len}) < self.gen_len ({self.gen_len}) + 6 + 2"


################################################################################
################################################################################
################################################################################

    def get_model(self, only_prepare = False):

        # split into prepare model and build
        print("start getting model")

        # import
        if self.intron_model:
            from My_Model_with_introns import My_Model
            self.model = My_Model(self)
        elif self.internal_exon_model:
            from My_internal_exon_model import My_Model
            self.model = My_Model(self)
        elif self.msa_model:
            from Felix_hard_coded_model import My_Model
            self.model = My_Model(self)
        else:
            from My_Model import My_Model
            self.model = My_Model(self)


        self.model.prepare_model()
        if not only_prepare:
            self.model.make_model()

        print("done getting model")


    # def print(self):
    #     s = "==========> config <==========\n"
    #     max_len = max([len(k[0]) for l in self.manuall_arg_lists.values() for k in l ])
    #     keys = [k for l in self.manuall_arg_lists.values() for k in l ]
    #     keys = sorted(keys)
    #     for i, key in enumerate(keys):
    #         s += key[0]
    #         if i % 5 == 0:
    #             s += "-" * (max_len - len(key[0]))
    #         else:
    #             s += " " * (max_len - len(key[0]))
    #         s += " = "
    #         s += str(self.__dict__[key[0]] if key[0] in self.__dict__ else self.parsed_args.__dict__[key[0]]) + "\n"

    #     # the attributes including parser
    #     # s += "==========> config <==========\n"
    #     # for key, value in self.__dict__.items():
    #     #     s += f"{key} = {str(value)[:50]}"
    #     s += "==========> config full <=========="

    #     print(s)

################################################################################
################################################################################
################################################################################

    def write_passed_args_to_file(self, dir_path = None):
        # TODO: auch viterbi args to file, wenn nicht im main gecalled?
        if dir_path is None:
            dir_path = self.current_run_dir
        if not os.path.exists(dir_path):
            os.system(f"mkdir -p {dir_path}")

        # common_keys = set(self.args.keys()) & set(self.det_args.keys())
        # # assert len(common_keys) == 0, f"merging args and det args failed, bc keys had intersection: {common_keys}"
        # print(f"merging args and det args keys had intersection: {common_keys}")
        # att = {**self.args, **self.det_args} # if key occured in both, the value from the second dir is used
        # att = dict(sorted(att.items()))
        out_path = f"{dir_path}/passed_args.json"
        print("writing cfg to file:", out_path)
        with open(out_path, "w") as file:
            json.dump(self.args, file)


    def __getattr__(self, name, exit_when_not_found = False):
        '''det_args have higher priority'''
        try:
            return self.det_args[name]
        except:
            try:
                return self.args[name]
            except:
                if exit_when_not_found:
                    print(f"config.[{name}] not defined")
                    exit(1)
                else:
                    return -1

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

    # def get_args_as_str(self, for_what): # for_what \in {"small_bench", "main_programm"}
    #     s = ""
    #     for key in self.manuall_arg_lists[for_what]:
    #         value = self.__dict__[key[0]] if key[0] in self.__dict__ else self.parsed_args.__dict__[key[0]]
    #         if type(value) == bool:
    #             s += key[1] + key[0] + " " if value else ""
    #         else:
    #             s += key[1] + key[0]  + " " + str(value) + " " if value != None else ""
    #     return(s)

################################################################################
################################################################################
################################################################################

    def apply_args(self):

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

################################################################################
################################################################################
################################################################################

    def asserts(self):
        if self.check_for_zeros:
            assert self.batch_begin_write_weights__layer_call_write_inputs, "if check_for_zeros also pass --batch"

        if self.get_gradient_of_first_batch:
            assert self.batch_size == 32, "if you pass get_gradient_of_first_batch, then batch_size must be 32 (=default)"
        if self.manual_training_loop:
            assert self.batch_size == 32, "if you pass manual_training_loop, then batch_size must be 32 (=default)"
        assert self.batch_size > 0, "batch_size must be greater than 0"
        assert self.AB in ["dd", "ds", "sd", "ss"], "-AB must be in ['dd', 'ds', 'sd', 'ss']"

        if self.scale_with_conditional_const:
            assert self.scale_with_const == 0, "if you pass --scale_with_conditional_const, then --scale_with_const must be 0"
            assert not self.felix, "not felix"

        if self.scale_with_const:
            assert not self.scale_with_conditional_const, "not scale_with_const"
            assert not self.felix, "not felix"

        if self.felix:
            assert not self.scale_with_const, "felix is on, so not scale with const"
            assert not self.scale_with_conditional_const, "felix is on, so not scale with conditional_const"

        # if self.ig5_const_transition:
        #     assert not self.use_weights_for_consts, "if --ig5_const_transition then --use_weights_for_consts cant be used"

        if self.simulate_insertions or self.simulate_deletions:
            assert not self.use_simple_seq_gen, "indels only work with MSAgen"
            assert not self.dont_generate_new_seqs, "simulate indels option isnt applied if you dont generate new seqs"

        if self.E_epsilon or self.l_epsilon or self.R_epsilon:
            assert self.logsumexp, "you passed E_epsilon or l_epsilon or R_epsilon, so you must also pass --log"

        if self.conditional_epsilon:
            assert self.scale_with_conditional_const, "you passed conditional_epsilon, so you must also pass scale_with_conditional_const"

        if self.my_scale_log_epsilon or self.my_scale_alpha_epsilon:
            assert not self.felix, "my_scale_epsilon was passed, so you must not use --felix"
            assert not self.scale_with_const, "my_scale_epsilon was passed, so you must not use --scale_with_const"
            assert not self.scale_with_conditional_const, "my_scale_epsilon was passed, so you must not use --scale_with_conditional_const"

        if self.manual_forward:
            assert self.AB == "dd", "manula forward only works with dense matrices, so pass -AB dd"

        if self.fasta_path:
            assert not self.dont_generate_new_seqs, "using fasta path, so nothig should get generated"
            assert not self.use_simple_seq_gen, "using fasta path, so nothig should get generated"

        assert self.akzeptor_pattern_len < 10, "akzeptor_pattern_len >= 10, setting priors uses the str rep of a state and only work with single diget number, this might not be the only place where this is required"
        assert self.donor_pattern_len < 10, "donor_pattern_len >= 10, setting priors uses the str rep of a state and only work with single diget number, this might not be the only place where this is required"


        if (self.priorA or self.priorB) and self.internal_exon_model:
            assert self.nCodons > 1, "when using prior and internal model you must use more than 1 codon since for 1 codon there are no priors for the transition matrix"


################################################################################
################################################################################
################################################################################

    def get_args_for_training(self):
        self.parser = argparse.ArgumentParser(description='Config module description')

        self.parser.add_argument('-c', '--nCodons', type = int, default = 1, help='number of codons')
        self.parser.add_argument('-AB', default = 'sd', help = '[sd (default), ds, sd, ss] specify the sparse or denseness of A and B')
        self.parser.add_argument('--order', type = int, default = 2, help = '[order] many preceeding emissions before the current one')
        self.parser.add_argument('-p', '--out_path', default = "../../cgp_data", help='path to paranet output dir')
        self.parser.add_argument('--path_to_MSAgen_dir', default= "../MSAgen", help = 'path to MSAgen_dir')
        self.parser.add_argument('--fasta_path', help = 'path to fasta file where the traning seqs are')

        # learning
        self.parser.add_argument('--optimizer', default = "SGD", help = 'Adam, Adadelta, Adagrad, Adamax, Ftrl , Nadam, RMSprop, SGD [SDG]')
        self.parser.add_argument('--epochs', default = 2, type = int, help = 'how many epochs [2]')
        self.parser.add_argument('--steps_per_epoch', default = 4, type = int, help = 'how many steps (i think batches) per epoch [4] (bc #seqs=100 batch_size=32 -> every seq is used)')
        self.parser.add_argument('-d', '--dtype64', action='store_true', help='using dytpe tf.float64')
        self.parser.add_argument('--batch_size', type = int, default = 32, help = 'the batch_size, default si 32')
        self.parser.add_argument('--no_learning', help ="learning_rate is set to 0", action='store_true')
        self.parser.add_argument('--learning_rate', help ="learning_rate", type = float, default = 0.05)
        self.parser.add_argument('--clip_gradient_by_value', help ="clip_gradient_by_values", type = float)
        self.parser.add_argument('--use_weights_for_consts', action='store_true', help ="use weights for transitions that become 1 after softmax")

        # what model
        self.parser.add_argument('--intron_model', action='store_true', help = 'use my model that includes introns')

        self.parser.add_argument('--msa_model', action = "store_true", help = "use a hard coded felix msa model with nucleodite emission to check if i can produce NaN, bc felix doesnt get NaN even for long seqs")

        self.parser.add_argument('--internal_exon_model', action = 'store_true', help = 'finde ein exon welches von zwei introns begrenzt ist')
        self.parser.add_argument('--inserts_at_intron_borders', action = 'store_true', help = 'inserts can come right after and before intron')
        self.parser.add_argument('--akzeptor_pattern_len', type = int, default = 3, help = 'akzeptor_pattern_len before AG')
        self.parser.add_argument('--donor_pattern_len', type = int, default = 3, help = 'donor_pattern_len after GT')
        self.parser.add_argument('--left_intron_const', action = 'store_true', help = 'uses const transition left_intron loop')
        self.parser.add_argument('--right_intron_const', action = 'store_true', help = 'uses const transition right_intron loop')
        self.parser.add_argument('--deletes_after_intron_to_codon', action = 'store_true', help = 'light green: deletes_after_intron_to_codon')
        self.parser.add_argument('--deletes_after_codon_to_intron', action = 'store_true', help = 'dark green: deletes_after_codon_to_intron')
        self.parser.add_argument('--deletes_after_insert_to_codon', action = 'store_true', help = 'red: deletes_after_insert_to_codon')
        self.parser.add_argument('--deletes_after_codon_to_insert', action = 'store_true', help = 'pink: deletes_after_codon_to_insert')
        self.parser.add_argument('--pattern_length_before_intron_loop', type = int, default = 2, help = 'number of states before intron loop')
        self.parser.add_argument('--pattern_length_after_intron_loop', type = int, default = 2, help = 'number of states after intron loop')
        self.parser.add_argument('--deletions_and_insertions_not_only_between_codons', action = 'store_true', help = 'deletions_and_insertions_not_only_between_codons. ie not after insertion or intron')
        self.parser.add_argument('--exon_skip_const', action = 'store_true', help = 'transition from left intron to rigth intron is not learend')

        # prior
        self.parser.add_argument('--priorB', type = float, default = 0, help = 'use prior for B and scale the alphas')
        self.parser.add_argument('--priorA', type = float, default = 0, help = 'use prior for A and scale the alphas')
        self.parser.add_argument('--scale_prior', type = float, default = 1, help = "scale the prior loss with this. bc it is not yet scaled by 1/m")
        self.parser.add_argument('--prior_path', default = "/home/mattes/Documents/cgp_data/priors/human/", help = ' path to the dir containing exon and intron .pbl')
        self.parser.add_argument('--ass_start', type = int, default = 5, help = 'len of prior pattern before AG ASS splice site')
        self.parser.add_argument('--ass_end', type = int, default = 2, help = 'len of prior pattern after AG ASS splice site')
        self.parser.add_argument('--dss_start', type = int, default = 5, help = 'len of prior pattern before GT DSS splice site')
        self.parser.add_argument('--dss_end', type = int, default = 2, help = 'len of prior pattern after GT DSS splice site')
        self.parser.add_argument('--log_prior_epsilon', type = float, default = 0, help = '[0] log_prior = tf.math.log(B(B_kernel) + prior_log_epsilon)')

        # prior and initial weights
        self.parser.add_argument('--my_initial_guess_for_parameters', action='store_true', help = 'init A weights with my initial guess and B with priors')
        self.parser.add_argument('--single_high_prob_kernel', type = float, default = 3, help = 'if my_initial_guess_for_parameters, this value is for high prob transitions, all other transitions get kernel weight 1')
        self.parser.add_argument('--diminishing_factor', type = float, default = 4, help = 'deletes get initialized with [[-(to_codon - from_codon)/config.diminishing_factor]]')
        self.parser.add_argument('--add_noise_to_initial_weights', action = 'store_true', help = 'add noise to my initial guess for weights ')
        self.parser.add_argument('--left_intron_init_weight', type = float, default = 4, help = 'weight for left -> left, the para for leaving left is 0')
        self.parser.add_argument('--right_intron_init_weight', type = float, default = 4, help = 'weight for right -> right, the para for leaving right is 0')
        self.parser.add_argument('--exon_skip_init_weight', type = float, default = -1, help = 'initparameter for exon strip')

        s = "else case to no model passed, ie the ATG CCC CCC STP model without introns"
        self.parser.add_argument('--ig5_const_transition', type = float, default = 0, help = "uses const transition from ig5 -> ig5 (weight = --ig5) and ig5 -> startA (weight = 1) and softmax applied")
        self.parser.add_argument('--ig3_const_transition', type = float, default = 0, help = "uses const transition from ig3 -> ig3 (weight = --ig3) and ig3 -> terminal (weight = 1) and softmax applied")
        self.parser.add_argument('--regularize', action= 'store_true', help = 'regularize the parameters')
        self.parser.add_argument('--inserts_punish_factor', type = float, default = 1, help = 'inserts_punish_factor')
        self.parser.add_argument('--deletes_punish_factor', type = float, default = 1, help = 'deletes_punish_factor')


        # what forward
        self.parser.add_argument('--felix', action='store_true',  help = 'use felix forward version')

        self.parser.add_argument('--logsumexp', action = "store_true", help = "logsumexp")
        self.parser.add_argument('--global_log_epsilon', type = float, default = 0, help = 'set l_, R_, E_ and prior_log_epsilon to this value')
        self.parser.add_argument('--l_epsilon', type = float, default = 0, help = '[0] loglik = tf.math.log(tf.reduce_sum(tf.math.exp(scaled_alpha - m_alpha) + config.epsilon_l, axis = 1, keepdims = True)) + m_alpha')
        self.parser.add_argument('--R_epsilon', type = float, default = 0, help = '[0] R = tf.math.log(mul(tf.math.exp(old_forward - m_alpha) + config.epsilon_R, A)) + m_alpha')
        self.parser.add_argument('--E_epsilon', type = float, default = 0, help = '[0] unscaled_alpha = tf.math.log(E + config.epsilon_E) + R')

        self.parser.add_argument('--scale_with_const', type = float, default = 0, help = 'scale the forward variables with constant float')

        self.parser.add_argument('--scale_with_conditional_const', action = "store_true", help = 'scale the forward variables with constant if they are too small')
        self.parser.add_argument('--conditional_epsilon', type = float, default = 0, help = '[0] loglik = tf.math.log(tf.reduce_sum(scaled_alpha, axis = 1, keepdims = True) + config.epsilon_conditional) - scale_helper * tf.math.log(10.0)')

        s = "my forward algo is default case if no other args is specified"
        self.parser.add_argument('--my_scale_log_epsilon', type = float, default = 0, help = '[0] loglik = tf.math.add(old_loglik, tf.math.log(scale_helper + config.epsilon_my_scale), name = "loglik")')
        self.parser.add_argument('--my_scale_alpha_epsilon', type = float, default = 0, help = '[0] scaled_alpha = unscaled_alpha / (scale_helper config.epsilon_my_scale_alpha)')

        # seq gen
        self.parser.add_argument('--seq_len', type = int, help = 'lenght of output seqs before the optional stripping of flanks, must be at least 3*nCodons + 8')
        self.parser.add_argument('--use_simple_seq_gen', action='store_true', help ="use_simple_seq_gen (just random seqs) and not MSAgen")
        self.parser.add_argument('-cd', '--coding_dist', type = float, default = 0.2, help='coding_dist for MSAgen')
        self.parser.add_argument('-ncd', '--noncoding_dist', type = float, default = 0.4, help='noncoding_dist for MSAgen')
        self.parser.add_argument('--dont_strip_flanks', action='store_true', help ="dont_strip_flanks ie all seqs have the same length")
        self.parser.add_argument('--dont_generate_new_seqs', action='store_true', help ="dont_generate_new_seqs, but use the ones that were created before")
        self.parser.add_argument('--simulate_insertions', action='store_true', help = 'simulate insertion when using MSAgen')
        self.parser.add_argument('--simulate_deletions', action='store_true', help = 'simulate deletions when using MSAgen')

        # hardware
        self.parser.add_argument('--split_gpu', action='store_true', help ="split gpu into 2 logical devices")
        self.parser.add_argument('--dont_use_mirrored_strategy', action='store_true', help ="dont_use_mirrored_strategy")

        # verbose
        self.parser.add_argument('-v', '--verbose', default = 0, type = int, help ="verbose E,R, alpha, A, B to file, pass 1 for shapes, 2 for shapes and values")
        self.parser.add_argument('-o', '--remove_verbose_at_batch_begin', action='store_true', help ="only_keep_verbose_of_last_batch")
        self.parser.add_argument('-s', '--verbose_to_stdout', action='store_true', help ="verbose to stdout instead of to file")
        self.parser.add_argument('--cpu_gpu', action='store_true', help ="print whether gpu or cpu is used")
        self.parser.add_argument('--batch_begin_write_weights__layer_call_write_inputs', action='store_true', help ="batch_begin_write_weights__layer_call_write_inputs")
        self.parser.add_argument('--get_gradient_of_first_batch', action='store_true', help ="get_gradient_of_first_batch")
        self.parser.add_argument('--get_gradient_for_current_txt', action='store_true', help ="get_gradient_for_current_txt, previous run wrote IAB and inputbatch to file (via --batch_begin_write_weights__layer_call_write_inputs flag)-> get respective gradient")
        self.parser.add_argument('--get_gradient_from_saved_model_weights', action='store_true', help ="get_gradient_from_saved_model_weights, previous run saved weights when passing --batch_begin_write_weights__layer_call_write_inputs")
        self.parser.add_argument('--assert_summarize', type = int, default = 5, help = 'assert_summarize [5]')
        self.parser.add_argument('--print_batch_id', action='store_true', help = 'prints the batch id via on_train_batch_begin callback')
        self.parser.add_argument('--write_matrices_after_fit', action ='store_true', help ='after fit write matrices to file')
        self.parser.add_argument('--write_parameters_after_fit', action = 'store_true', help = 'after fit write kernels to file')
        self.parser.add_argument('--write_initial_weights_and_matrices_to_file', action='store_true', help = 'before fit write kernels to file')

        # debugging
        self.parser.add_argument('-b', '--exit_after_first_batch', action = 'store_true', help ="exit after first batch, you may use this when verbose is True in cell.call()")
        self.parser.add_argument('-n', '--exit_after_loglik_is_nan', action='store_true', help ="exit_after_loglik_is_nan, you may use this when verbose is True in cell.call()")
        self.parser.add_argument('--manual_training_loop', action='store_true', help ="manual_training_loop")
        self.parser.add_argument('--check_assert', action='store_true', help ="check_assert")
        self.parser.add_argument('--run_eagerly', action='store_true', help ='run model.fit in eager execution')
        self.parser.add_argument('--alpha_i_gradient', type = int, default = -1, help = 'if --manual_training_loop is passed, then the gradient for alpha_i wrt the kernels is computed, if -2 is passed, i is set to n - 1, where n is the length of th seq')
        self.parser.add_argument('--init_weights_from_before_fit', action='store_true', help = 'if this is passed the cells kernels are initialized with the weights stored in the txt files, which were written on a previous run when --batch_begin_write_weights__layer_call_write_inputs was passed')
        self.parser.add_argument('--init_weights_from_after_fit' , action='store_true', help = 'if this is passed the cells kernels are initialized with the weights stored in the txt files, which were written on a previous run when --write_parameters_after_fit was passed')
        self.parser.add_argument('--no_deletes', action='store_true', help = 'the delete transitions in A are removed')
        self.parser.add_argument('--no_inserts', action='store_true', help = 'the insert transitions in A are removed')
        self.parser.add_argument('--forced_gene_structure', action='store_true', help = 'TGs in igs and ACs in coding, ie the state seq is determinded by emission seq')
        self.parser.add_argument('--check_for_zeros', action='store_true', help = 'must be passed together with --batch, checks for zeros in parameters')
        self.parser.add_argument('--use_constant_initializer', action='store_true', help = 'init weights with all ones')
        self.parser.add_argument('--manual_forward', action = 'store_true', help = 'gets mean likelihood of with manual loop')
        self.parser.add_argument('--autograph_verbose', action = 'store_true', help = 'set tf.autograph.set_verbosity(3, True)')


        self.parser.add_argument('--viterbi', action='store_true', help = 'run viterbi after training')
        self.get_args_for_viterbi()


    def get_args_for_viterbi(self):
        if self.parser == -1:
            self.parser = argparse.ArgumentParser(description='Config module description')

        self.parser.add_argument('--only_first_seq', action = 'store_true', help = 'run viterbi only for the first seq')
        self.parser.add_argument('--viterbi_parent_input_dir', help = 'path to dir containing the config_attr.json and paratemeters dir used for viterbi')
        self.parser.add_argument('--in_viterbi_path', help = 'if viteribi is already calculated, path to viterbi file which is then written to the alignment')
        self.parser.add_argument('--viterbi_threads', type = int, default = 1, help = 'how many threads for viterbi.cc')
        self.parser.add_argument('--path_to_dir_where_most_recent_dir_is_selected', help = 'path_to_dir_where_most_recent_dir_is_selected')

