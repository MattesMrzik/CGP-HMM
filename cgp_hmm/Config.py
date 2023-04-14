import argparse
import Utility
import re
from Utility import run
import os
class Config():

    def __init__(self, for_which_program):
        print("start init of config")
        self.parser = argparse.ArgumentParser(description='Config module description')
        self.manuall_arg_lists = {"small_bench" : [], "main_programm" : [], "without_priors": []}
        if for_which_program == "small_bench":
            self.add_small_bench()
            self.parsed_args = self.parser.parse_args()

        if for_which_program == "main_programm":
            self.add_main_programm()
            self.parsed_args = self.parser.parse_args()

            self.asserts()
            self.add_attribtes()
            self.prepare_before_main_programm()
            self.determine_attributes()
            self.apply_args()

        if for_which_program == "main_programm_dont_interfere":
            self.add_main_programm()
            self.parsed_args = self.parser.parse_args()

            self.asserts()
            self.add_attribtes()
            self.determine_attributes()
            self.apply_args()

        if for_which_program == "without_priors":
            self.add_main_programm()
            self.parsed_args = self.parser.parse_args()

            self.asserts()
            self.without_priors = True
            self.add_attribtes()
            self.determine_attributes()
            self.apply_args()
        print("done init of config")

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

        if self.ig5_const_transition:
            assert not self.use_weights_for_consts, "if --ig5_const_transition then --use_weights_for_consts cant be used"

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


        if self.prior and self.internal_exon_model:
            assert self.nCodons > 1, "when using prior and internal model you must use more than 1 codon since for 1 codon there are no priors for the transition matrix"



    def add_attribtes(self):
        import tensorflow as tf

        self.tf_version = tf.__version__
        # added this, so that it gets printed
        self.manuall_arg_lists["main_programm"].append(("tf_version", self.tf_version))

        self.alphabet_size = 4
        self.write_return_sequnces = False

        self.bench_path = f"{self.out_path}/bench/{self.nCodons}codons/{self.AB}_call_type.log"
        if not self.fasta_path:
            self.manual_passed_fasta = False
            self.fasta_path = f"{self.out_path}/output/{self.nCodons}codons/seqs.fa"
        else:
            self.manual_passed_fasta = True
            self.generate_new_seqs = False

        self.generate_new_seqs = not self.dont_generate_new_seqs
        self.dtype = tf.float64 if self.dtype64 else tf.float32
        self.learning_rate = self.learning_rate if not self.no_learning else 0

        self.A_is_dense = self.AB[0] == "d"
        self.A_is_sparse = not self.A_is_dense
        self.B_is_dense = self.AB[1] == "d"
        self.B_is_sparse = not self.B_is_dense


        if self.global_log_epsilon:
            self.E_epsilon = self.global_log_epsilon
            self.R_epsilon = self.global_log_epsilon
            self.l_epsilon = self.global_log_epsilon
            self.log_prior_epsilon = self.global_log_epsilon


        self.gen_len = 3 * self.nCodons

        self.seq_len = self.parsed_args.seq_len if self.parsed_args.seq_len else ((self.nCodons * 3 + 6 + 2) * 2)

        #                                     start and stop, i want at least one ig 3' and 5'
        assert self.seq_len >= self.gen_len + 6               + 2, f"self.seq_len ({self.seq_len}) < self.gen_len ({self.gen_len}) + 6 + 2"


        print("start getting model")
        if self.intron_model:
            from My_Model_with_introns import My_Model
            my_model = My_Model(self)
            self.model = my_model
        elif self.internal_exon_model:
            from My_internal_exon_model import My_Model
            my_model = My_Model(self)
            self.model = my_model
        elif self.msa_model:
            from Felix_hard_coded_model import My_Model
            my_model = My_Model(self)
            self.model = my_model
        else:
            from My_Model import My_Model
            my_model = My_Model(self)
            self.model = my_model
        print("done getting model")


    def prepare_before_main_programm(self):
        paths = [f"{self.out_path}/output/{self.nCodons}codons/", \
                 f"{self.out_path}/verbose"]
        for path in paths:
            if not os.path.exists(path):
                os.system(f"mkdir -p {path}")
        if self.verbose:
            os.system(f"rm {self.out_path}/verbose/{self.nCodons}codons.txt")
        os.system(f"rm {self.bench_path}")

    def determine_attributes(self):
        pass
        # print("self.indices_for_B in determine_attributes in config =", self.indices_for_B)

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

        # the attributes including parser
        # s += "==========> config <==========\n"
        # for key, value in self.__dict__.items():
        #     s += f"{key} = {str(value)[:50]}"
        s += "==========> config full <=========="

        print(s)


    def __getattr__(self, name):
        return self.parsed_args.__dict__[name]

    def add_small_bench(self):
        self.add_arg_small_bench('-r', '--range_codon', nargs='+', help='usage: < -r 1 10 > to run 1 2 3 4 5 6 7 8 9 10 codons')
        self.add_arg_small_bench('-cl', '--nCodonsList', nargs="+", help ='usage: < -c 10 20 50 > to run 10 20 50 codons')
        # self.add_arg_small_bench('-il', '--alpha_i_gradient_list', nargs='+', help ='is only applied when --manual is passed')
        self.add_arg_small_bench('--repeat', type = int, default = 1, help ='repeat the main programm [repeat] times')
        self.add_arg_small_bench('--exit_on_nan', action='store_true', help ="exit_on_nan")

        self.add_main_programm()

    def add_main_programm(self):
        self.add_arg_main('-c', '--nCodons', type = int, default = 1, help='number of codons')
        self.add_arg_main('-AB', default = 'sd', help = '[sd (default), ds, sd, ss] specify the sparse or denseness of A and B')
        self.add_arg_main('--order', type = int, default = 2, help = '[order] many preceeding emissions before the current one')
        self.add_arg_main('-p', '--out_path', default = "../../cgp_data/", help='path to paranet output dir')
        self.add_arg_main('--path_to_MSAgen_dir', default= "../MSAgen", help = 'path to MSAgen_dir')
        self.add_arg_main('--optimizer', default = "SGD", help = 'Adam, Adadelta, Adagrad, Adamax, Ftrl , Nadam, RMSprop, SGD [SDG]')
        self.add_arg_main('--epochs', default = 2, type = int, help = 'how many epochs [2]')
        self.add_arg_main('--steps_per_epoch', default = 4, type = int, help = 'how many steps (i think batches) per epoch [4] (bc #seqs=100 batch_size=32 -> every seq is used)')
        self.add_arg_main('--viterbi', action='store_true', help = 'viterbi')
        self.add_arg_main('--only_first_seq', action = 'store_true', help = 'run viterbi only for the first seq')
        self.add_arg_main('--intron_model', action='store_true', help = 'use my model that includes introns')
        self.add_arg_main('--fasta_path', help = 'path to fasta file where the traning seqs are')
        self.add_arg_main('--in_viterbi_path', help = 'path to viterbi file which is then written to the alignment')

        self.add_arg_main('--internal_exon_model', action = 'store_true', help = 'finde ein exon welches von zwei introns begrenzt ist')
        self.add_arg_main('--inserts_at_intron_borders', action = 'store_true', help = 'inserts can come right after and before intron')
        self.add_arg_main('--akzeptor_pattern_len', type = int, default = 3, help = 'akzeptor_pattern_len before AG')
        self.add_arg_main('--donor_pattern_len', type = int, default = 3, help = 'donor_pattern_len after GT')
        self.add_arg_main('--left_intron_const',  type = float, default = 0, help = 'uses const transition [float] from left_intron -> left_intron and left_intron -> ak_0 [weight = 1]')
        self.add_arg_main('--right_intron_const',  type = float, default = 0, help = 'uses const transition [float] from right_intron -> right_intron and right_intron -> ter [weight = 1]')
        self.add_arg_main('--deletes_after_intron_to_codon', action = 'store_true', help = 'light green: deletes_after_intron_to_codon')
        self.add_arg_main('--deletes_after_codon_to_intron', action = 'store_true', help = 'dark green: deletes_after_codon_to_intron')
        self.add_arg_main('--deletes_after_insert_to_codon', action = 'store_true', help = 'red: deletes_after_insert_to_codon')
        self.add_arg_main('--deletes_after_codon_to_insert', action = 'store_true', help = 'pink: deletes_after_codon_to_insert')
        self.add_arg_main('--prior', type = float, default = 0, help = 'use prior reg or scale it')
        self.add_arg_main('--prior_path', default = "/home/mattes/Documents/cgp_data/priors/human/", help = ' path to the dir containing exon and intron .pbl')

        # fine tune algo
        self.add_arg_main('--use_weights_for_consts', action='store_true', help ="use weights for transitions that become 1 after softmax")
        self.add_arg_main('-d', '--dtype64', action='store_true', help='using dytpe tf.float64')
        self.add_arg_main('--clip_gradient_by_value', help ="clip_gradient_by_values", type = float)
        self.add_arg_main('--learning_rate', help ="learning_rate", type = float, default = 0.05)
        self.add_arg_main('--no_learning', help ="learning_rate is set to 0", action='store_true')
        self.add_arg_main('--seq_len', type = int, help = 'lenght of output seqs before the optional stripping of flanks, must be at least 3*nCodons + 8')
        self.add_arg_main('--use_simple_seq_gen', action='store_true', help ="use_simple_seq_gen (just random seqs) and not MSAgen")
        self.add_arg_main('-cd', '--coding_dist', type = float, default = 0.2, help='coding_dist for MSAgen')
        self.add_arg_main('-ncd', '--noncoding_dist', type = float, default = 0.4, help='noncoding_dist for MSAgen')
        self.add_arg_main('--dont_strip_flanks', action='store_true', help ="dont_strip_flanks ie all seqs have the same length")
        self.add_arg_main('--batch_size', type = int, default = 32, help = 'the batch_size, default si 32')
        self.add_arg_main('--scale_with_const', type = float, default = 0, help = 'scale the forward variables with constant float')
        self.add_arg_main('--scale_with_conditional_const', action = "store_true", help = 'scale the forward variables with constant if they are too small')
        self.add_arg_main('--felix', action='store_true',  help = 'use felix forward version')
        self.add_arg_main('--logsumexp', action = "store_true", help = "logsumexp")
        self.add_arg_main('--global_log_epsilon', type = float, default = 0, help = 'set l_, R_, E_ and prior_log_epsilon to this value')
        self.add_arg_main('--l_epsilon', type = float, default = 0, help = '[0] loglik = tf.math.log(tf.reduce_sum(tf.math.exp(scaled_alpha - m_alpha) + self.config.epsilon_l, axis = 1, keepdims = True)) + m_alpha')
        self.add_arg_main('--R_epsilon', type = float, default = 0, help = '[0] R = tf.math.log(mul(tf.math.exp(old_forward - m_alpha) + self.config.epsilon_R, self.A)) + m_alpha')
        self.add_arg_main('--E_epsilon', type = float, default = 0, help = '[0] unscaled_alpha = tf.math.log(E + self.config.epsilon_E) + R')
        self.add_arg_main('--log_prior_epsilon', type = float, default = 0, help = '[0] log_prior = tf.math.log(self.B(B_kernel) + prior_log_epsilon)')
        self.add_arg_main('--my_scale_log_epsilon', type = float, default = 0, help = '[0] loglik = tf.math.add(old_loglik, tf.math.log(scale_helper + self.config.epsilon_my_scale), name = "loglik")')
        self.add_arg_main('--my_scale_alpha_epsilon', type = float, default = 0, help = '[0] scaled_alpha = unscaled_alpha / (scale_helper self.config.epsilon_my_scale_alpha)')
        self.add_arg_main('--conditional_epsilon', type = float, default = 0, help = '[0] loglik = tf.math.log(tf.reduce_sum(scaled_alpha, axis = 1, keepdims = True) + self.config.epsilon_conditional) - scale_helper * tf.math.log(10.0)')
        self.add_arg_main('--ig5_const_transition', type = float, default = 0, help = "uses const transition from ig5 -> ig5 (weight = --ig5) and ig5 -> startA (weight = 1) and softmax applied")
        self.add_arg_main('--ig3_const_transition', type = float, default = 0, help = "uses const transition from ig3 -> ig3 (weight = --ig3) and ig3 -> terminal (weight = 1) and softmax applied")
        self.add_arg_main('--regularize', action= 'store_true', help = 'regularize the parameters')
        self.add_arg_main('--inserts_punish_factor', type = float, default = 1, help = 'inserts_punish_factor')
        self.add_arg_main('--deletes_punish_factor', type = float, default = 1, help = 'deletes_punish_factor')
        self.add_arg_main('--simulate_insertions', action='store_true', help = 'simulate insertion when using MSAgen')
        self.add_arg_main('--simulate_deletions', action='store_true', help = 'simulate deletions when using MSAgen')
        self.add_arg_main('--pattern_length_before_intron_loop', type = int, default = 2, help = 'number of states before intron loop')
        self.add_arg_main('--pattern_length_after_intron_loop', type = int, default = 2, help = 'number of states after intron loop')
        self.add_arg_main('--deletions_and_insertions_not_only_between_codons', action = 'store_true', help = 'deletions_and_insertions_not_only_between_codons. ie not after insertion or intron')
        self.add_arg_main('--my_initial_guess_for_parameters', action='store_true', help = 'init A weights with my initial guess')
        self.add_arg_main('--single_high_prob_kernel', type = float, default = 3, help = 'if my_initial_guess_for_parameters, this value is for high prob transitions, all other transitions get kernel weight 1')
        self.add_arg_main('--diminishing_factor', type = float, default = 4, help = 'deletes get initialized with [[-(to_codon - from_codon)/self.config.diminishing_factor]]')
        self.add_arg_main('--ass_start', type = int, default = 5, help = 'len of prior pattern before AG ASS splice site')
        self.add_arg_main('--ass_end', type = int, default = 2, help = 'len of prior pattern after AG ASS splice site')
        self.add_arg_main('--dss_start', type = int, default = 5, help = 'len of prior pattern before GT DSS splice site')
        self.add_arg_main('--dss_end', type = int, default = 2, help = 'len of prior pattern after GT DSS splice site')

        # hardware
        self.add_arg_main('--split_gpu', action='store_true', help ="split gpu into 2 logical devices")
        self.add_arg_main('--dont_use_mirrored_strategy', action='store_true', help ="dont_use_mirrored_strategy")

        # verbose
        self.add_arg_main('-v', '--verbose', default = 0, type = int, help ="verbose E,R, alpha, A, B to file, pass 1 for shapes, 2 for shapes and values")
        self.add_arg_main('-o', '--remove_verbose_at_batch_begin', action='store_true', help ="only_keep_verbose_of_last_batch")
        self.add_arg_main('-s', '--verbose_to_stdout', action='store_true', help ="verbose to stdout instead of to file")
        self.add_arg_main('--cpu_gpu', action='store_true', help ="print whether gpu or cpu is used")
        self.add_arg_main('--batch_begin_write_weights__layer_call_write_inputs', action='store_true', help ="batch_begin_write_weights__layer_call_write_inputs")
        self.add_arg_main('--get_gradient_of_first_batch', action='store_true', help ="get_gradient_of_first_batch")
        self.add_arg_main('--get_gradient_for_current_txt', action='store_true', help ="get_gradient_for_current_txt, previous run wrote IAB and inputbatch to file (via --batch_begin_write_weights__layer_call_write_inputs flag)-> get respective gradient")
        self.add_arg_main('--get_gradient_from_saved_model_weights', action='store_true', help ="get_gradient_from_saved_model_weights, previous run saved weights when passing --batch_begin_write_weights__layer_call_write_inputs")
        self.add_arg_main('--assert_summarize', type = int, default = 5, help = 'assert_summarize [5]')
        self.add_arg_main('--print_batch_id', action='store_true', help = 'prints the batch id via on_train_batch_begin callback')
        self.add_arg_main('--write_matrices_after_fit', action ='store_true', help ='after fit write matrices to file')
        self.add_arg_main('--write_parameters_after_fit', action = 'store_true', help = 'after fit write kernels to file')
        self.add_arg_main('--write_initial_weights_and_matrices_to_file', action='store_true', help = 'before fit write kernels to file')

        # debugging
        self.add_arg_main('-b', '--exit_after_first_batch', action = 'store_true', help ="exit after first batch, you may use this when verbose is True in cell.call()")
        self.add_arg_main('-n', '--exit_after_loglik_is_nan', action='store_true', help ="exit_after_loglik_is_nan, you may use this when verbose is True in cell.call()")
        self.add_arg_main('--dont_generate_new_seqs', action='store_true', help ="dont_generate_new_seqs, but use the ones that were created before")
        self.add_arg_main('--manual_training_loop', action='store_true', help ="manual_training_loop")
        self.add_arg_main('--check_assert', action='store_true', help ="check_assert")
        self.add_arg_main('--run_eagerly', action='store_true', help ='run model.fit in eager execution')
        self.add_arg_main('--alpha_i_gradient', type = int, default = -1, help = 'if --manual_training_loop is passed, then the gradient for alpha_i wrt the kernels is computed, if -2 is passed, i is set to n - 1, where n is the length of th seq')
        self.add_arg_main('--init_weights_from_before_fit', action='store_true', help = 'if this is passed the cells kernels are initialized with the weights stored in the txt files, which were written on a previous run when --batch_begin_write_weights__layer_call_write_inputs was passed')
        self.add_arg_main('--init_weights_from_after_fit' , action='store_true', help = 'if this is passed the cells kernels are initialized with the weights stored in the txt files, which were written on a previous run when --write_parameters_after_fit was passed')
        self.add_arg_main('--add_noise_to_initial_weights', action = 'store_true', help = 'add noise to my initial guess for weights ')
        self.add_arg_main('--no_deletes', action='store_true', help = 'the delete transitions in A are removed')
        self.add_arg_main('--no_inserts', action='store_true', help = 'the insert transitions in A are removed')
        self.add_arg_main('--forced_gene_structure', action='store_true', help = 'TGs in igs and ACs in coding, ie the state seq is determinded by emission seq')
        self.add_arg_main('--check_for_zeros', action='store_true', help = 'must be passed together with --batch, checks for zeros in parameters')
        self.add_arg_main('--use_constant_initializer', action='store_true', help = 'init weights with all ones')
        self.add_arg_main('--manual_forward', action = 'store_true', help = 'gets mean likelihood of with manual loop')
        self.add_arg_main('--autograph_verbose', action = 'store_true', help = 'set tf.autograph.set_verbosity(3, True)')
        self.add_arg_main('--msa_model', action = "store_true", help = "use a hard coded felix msa model with nucleodite emission to check if i can produce NaN, bc felix doesnt get NaN even for long seqs")

    def get_args_as_str(self, for_what): # for_what \in {"small_bench", "main_programm"}
        s = ""
        for key in self.manuall_arg_lists[for_what]:
            value = self.__dict__[key[0]] if key[0] in self.__dict__ else self.parsed_args.__dict__[key[0]]
            if type(value) == bool:
                s += key[1] + key[0] + " " if value else ""
            else:
                s += key[1] + key[0]  + " " + str(value) + " " if value != None else ""
        return(s)
