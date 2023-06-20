#!/usr/bin/env python3
from datetime import datetime
import numpy as np
import math
import os
import json
import pandas as pd
import time
import re
from itertools import product
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from Bio import SeqIO
import argparse

from helpers.add_gene_structure_to_alignment import read_true_alignment_with_out_coords_seq

out_path = "../../cgp_data"
# plt.hist(seqs_df["true_p_nt_exon"][seqs_df["true_p_nt_exon"]<600]); plt.savefig("hist.png")

def get_multi_run_config():

    parser = argparse.ArgumentParser(description='Config module description')
    parser.add_argument('--slurm', action = 'store_true', help='use with slurm')

    parser.add_argument('--train', action = 'store_true', help='train with args specified in the methods in this module and write output to calculated path based on current time')
    parser.add_argument('--mem', type = int, default = 20000, help='mem for slurm train')
    parser.add_argument('--adaptive_ram', action = "store_true", help='adaptive mem for slurm train')
    # parser.add_argument('--mpi', type = int, default = 8, help='mpi for slurm train')
    parser.add_argument('--max_jobs', type = int, default = 8, help='max number of jobs for slurm train')
    parser.add_argument('--partition', default = "snowball", help='partition for slurm train')
    parser.add_argument('--continue_training', help='path to multi_run dir for which to continue training')

    parser.add_argument('--viterbi_path', help='path to multi_run dir for which to run viterbi')
    parser.add_argument('--use_init_weights_for_viterbi', action = 'store_true', help = 'use the initial weights instead of the learned ones')
    parser.add_argument('--eval_viterbi', help='path to multi_run dir for which to evaluation of viterbi')
    # parser.add_argument('--threads_for_viterbi', type = int, default = 1, help='how many threads should be used for viterbi')
    # here only first seq, ie the human one is calculated, an parallel calc of M produced more overhead than it actually helped

    args = parser.parse_args()

    if args.use_init_weights_for_viterbi:
        assert args.viterbi_path, "if you pass --use_init_weights_for_viterbi. you must also pass --viterbi_path"

    # do i want to create bench to run on slurm submission which might be able to run both tf.learning and c++ viterbi
    # or create 2 modes in bench one for training on apphub and one for running viterbi and evaluating


    assert args.train or args.viterbi_path or args.eval_viterbi or args.continue_training, "you must pass either --train or --viterbi_path or --eval_viterbi"


    return args

################################################################################

def get_cfg_without_args_that_are_switched():

    # cfg_without_args_that_are_switched = '''
    # exon_skip_const
    # '''
    #
    # cfg_without_args_that_are_switched = re.split("\s+", cfg_without_args_that_are_switched)[1:-1]

    pass

################################################################################

def get_cfg_with_args():
    cfg_with_args = {}

    # or read files in a passed dir
    fasta_dir_path = "/home/s-mamrzi/cgp_data/train_data_set/2023-06-06_19-15/good_exons_1"
    fasta_dir_path = "/home/s-mamrzi/cgp_data/eval_data_set/2023-06-06_20-01/good_exons_1"
    # exons = ["exon_chr1_8364055_8364255", \
    #         "exon_chr1_33625050_33625254"]

    exons = [dir for dir in os.listdir(fasta_dir_path) if not os.path.isfile(os.path.join(fasta_dir_path, dir)) ]

    # TODO i need to determine the nCodons that should be used for each fasta,
    # are there other parameters that defend on the sequences?

    cfg_with_args["fasta"] = [f"{fasta_dir_path}/{exon}/introns/combined.fasta" for exon in exons]
    cfg_with_args["fasta"] = [f"{fasta_dir_path}/{exon}/missing_human_exon/combined.fasta" for exon in exons]
    cfg_with_args["fasta"] = [f"{fasta_dir_path}/{exon}/combined.fasta" for exon in exons]
    get_exon_len = lambda exon_string: (int(exon_string.split("_")[-1]) - int(exon_string.split("_")[-2]))
    get_exon_codons = lambda exon_string: get_exon_len(exon_string) // 3


    #  i can also pass the same arg to a parameter twice and bind it with another parameter to achieve
    # something like manual defined grid points

    # these need to be named the same as in Config.py
    # by if i only pass a partial str as key here
    # it is completed by Config.py but columns i get from the keys
    # of cfg_with_args arent completet automatically
    # i.e epoch is completed to epochs

    # if i pass more than one parameter to an arg, they have to be non unifrom

    exon_nCodons = [get_exon_codons(exon) for exon in exons]

    cfg_with_args["nCodons"] = exon_nCodons
    cfg_with_args["model_size_factor"] = [0.5,0.75,1,1.25,1.5]
    cfg_with_args["model_size_factor"] = [1]

    # ich habe inserts teurer gemacht um die die ergnisse für zu kurze modelle
    # schlechter zu machen, aber ich könnte auch die deletions billiger machen
    # damit einfach die model länge nicht so stark in die gewichtung eingeht
    # vielleicht leider die ergebnisse nicht zu sehr darunter


    # cfg_with_args["exon_sk    ip_init_weight"] = [-2,-3,-4]
    # cfg_with_args["learning_rate"] = [0.1, 0.01]

    cfg_with_args["priorA"] = [10]
    cfg_with_args["priorB"] = [10]
    # cfg_with_args["likelihood_influence_growth_factor"] = [0.2 ,0]

    cfg_with_args["akzeptor_pattern_len"] = [5]
    cfg_with_args["donor_pattern_len"] = [5]

    cfg_with_args["global_log_epsilon"] = [1e-20]
    cfg_with_args["epochs"] = [40]
    cfg_with_args["learning_rate"] = [0.01]
    cfg_with_args["batch_size"] = [16]
    cfg_with_args["step"] = [16]
    cfg_with_args["clip_gradient_by_value"] = [4]
    # cfg_with_args["exon_skip_init_weight"] = [-2, -4, -10]
    # cfg_with_args["exon_skip_init_weight_factor"] = [1,5] # 0 is ni skip, 1 is cheaper than 5, 0s cost is infinite so 0 and 0.0001 are very different
    cfg_with_args["exon_skip_init_weight"] = [-5]
    cfg_with_args["flatten_B_init"] = [0]
    cfg_with_args["cesar_init"] = [1]
    cfg_with_args["optimizer"] = ["Adam"] # SGD didnt work nice with 0.001 learning rate and 1 clip_gradient_by_value

    cfg_with_args["logsumexp"] = [1]

    cfg_with_args["dataset_identifier"] = ["all", "../../cgp_data/primates.lst", "../../cgp_data/max_diverse_subtrees"]
    cfg_with_args["dataset_identifier"] = ["../../cgp_data/max_diverse_subtrees"]

    cfg_with_args["left_intron_init_weight"] = [4.6] # this is usually 4 but left miss was way more often than right miss in eval data set

    cfg_with_args["manual_delete_insert_init_continue_weights"] = ["\"0.1,-0.3,-1\""]

    cfg_with_args["left_intron_const"] = [0,1]
    cfg_with_args["right_intron_const"] = [0,1]



    return cfg_with_args

################################################################################

def get_bind_args_together(cfg_with_args):
    '''
    cant bind args without parameter
    '''
    bind_args_together = [set([key]) for key in cfg_with_args.keys()]
    bind_args_together += [{"fasta", "nCodons"}]
    # bind_args_together += [{"exon_skip_init_weight", "nCodons"}]
    # bind_args_together += [{"priorA", "priorB"}]
    # bind_args_together += [{"priorA", "likelihood_influence_growth_factor"}]
    bind_args_together += [{"akzeptor_pattern_len", "donor_pattern_len"}]

    # left and right intron const
    bind_args_together += [{"left_intron_const", "right_intron_const"}]

    return bind_args_together

################################################################################

def get_cfg_without_args():
    cfg_without_args = '''
    internal_exon_model
    my_initial_guess_for_parameters
    exit_after_loglik_is_nan
    viterbi
    '''
    # exon_skip_const
    # bucket_by_seq_len


    cfg_without_args = re.split("\s+", cfg_without_args)[1:-1]
    return cfg_without_args


################################################################################

def get_and_make_dir():
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M")

    multi_run_dir = f"{out_path}/multi_run_{date_string}"
    if not os.path.exists(multi_run_dir):
        os.makedirs(multi_run_dir)
    return multi_run_dir

################################################################################

def write_cfgs_to_file(multi_run_dir):
    cfg_with_args_path = f"{multi_run_dir}/cfg_with_args.json"
    with open(cfg_with_args_path, "w") as out_file:
        json.dump(get_cfg_with_args(), out_file)

    cfg_without_args_path = f"{multi_run_dir}/cfg_without_args.json"
    with open(cfg_without_args_path, "w") as out_file:
        json.dump(get_cfg_without_args(), out_file)

################################################################################

def merge_one_step(inp : list[set]):
    for i in range(len(inp)):
        for j in range(i+1, len(inp)):
            if len(inp[i] & inp[j]) > 0:
                inp.append(inp[i] | inp[j])
                break
        else:
            continue
        break
    inp.remove(inp[j])
    inp.remove(inp[i])

################################################################################

def no_overlapp(inp : list[set]) -> bool:
        for i in range(len(inp)):
            for j in range(i+1, len(inp)):
                if len(inp[i] & inp[j]) > 0:
                    return False
        return True

################################################################################

def merge(inp : list[set[str]]) -> list[set[str]]:
    while not no_overlapp(inp):
        merge_one_step(inp)

################################################################################

def zip_args(inp : list[set]) -> list[tuple[set, zip]]:
    zipped_args = []
    for merged_args in inp:
        zipped_args.append((merged_args,(zip(*[get_cfg_with_args()[key] for key in merged_args]))))
    return zipped_args

################################################################################

def get_grid_points(zipped_args : list[tuple[set, zip]]) -> list[list[list]]:
    '''return list of grid points.
    a gridpoint is a list of parameters.
    a single parameter is wrapped in a list.
    binded parameters are in the same list'''
    return list(product(*[arg[-1] for arg in zipped_args]))

################################################################################

def run_training(args):

    binded_arg_names = get_bind_args_together(get_cfg_with_args())

    # [{'global_log_epsilon'}, {'epoch'}, {'step'}, {'clip_gradient_by_value'}, {'prior_path'}, {'exon_skip_init_weight'}, {'nCodons', 'fasta'}, {'priorB', 'priorA'}]
    merge(binded_arg_names)
    zipped_args = zip_args(binded_arg_names)

    arg_names = [single_arg for arg in zipped_args for single_arg in arg[0]]

    grid_points = get_grid_points(zipped_args)

    # every point in grid_points is a list of parameters i want to extract the sublist that contains the fasta and nCodons
    def get_sort_values(point_in_grid):
        for sublist in point_in_grid:
            for item in sublist:
                if type(item) is str and re.search("fasta", item):
                    fasta = item
                elif len(sublist) == 2:
                    nCodons = item
        path = os.path.join(os.path.dirname(fasta), "species_seqs/stripped/Homo_sapiens.fa")
        with open(path, "r") as fasta_file:
            seq = next(SeqIO.parse(fasta_file, "fasta"))
            seq_len = len(seq.seq)
        return seq_len * np.sqrt(nCodons)

    # grid_points = sorted(grid_points, key = get_sort_values, reverse=True)

    print("len get_grid_points", len(grid_points))

    print("do you want to continue enter [y/n]")
    while (x :=input()) not in "yn":
        print("input was no y or n")
    if x == "n":
        print("exiting")
        exit()

    multi_run_dir = get_and_make_dir()
    write_cfgs_to_file(multi_run_dir)
    with open(f"{multi_run_dir}/arg_names.json", "w") as arg_names_file:
        json.dump(arg_names, arg_names_file)

    with open(f"{multi_run_dir}/grid_points.json", "w") as grid_point_file:
        json.dump(grid_points, grid_point_file)

    with open(f"{multi_run_dir}/todo_grid_points.json", "w") as grid_point_file:
        json.dump(grid_points, grid_point_file)

    args.continue_training = multi_run_dir
    continue_training(args)

################################################################################

def continue_training(parsed_args):

    with open(f"{parsed_args.continue_training}/todo_grid_points.json", "r") as grid_point_file:
        grid_points = json.load(grid_point_file)

    with open(f"{parsed_args.continue_training}/arg_names.json", "r") as arg_names_file:
        # ['global_log_epsilon', 'epoch', 'step', 'clip_gradient_by_value', 'prior_path', 'exon_skip_init_weight', 'fasta', 'nCodons', 'priorB', 'priorA']
        arg_names = json.load(arg_names_file)


    for i, point_in_grid in enumerate(grid_points):
        print(f"calculating point in hyperparameter grid {i}/{len(grid_points)}")
        # [1e-20, 20, 8, 5, ' ../../cgp_data/priors/human/', -10, '../../cgp_data/good_exons_1/exon_chr1_1050426_1050591/combined.fasta', 55, 100, 100]

        class BreakException(Exception):
            pass
        try:
            for entry in os.listdir(parsed_args.continue_training):
                full_path = os.path.join(parsed_args.continue_training, entry)
                if not os.path.isdir(full_path):
                    continue
                grid_point_log_file_path = os.path.join(full_path, "grid_point.log")
                with open(grid_point_log_file_path,"r") as grid_point_file:
                    for i, line in enumerate(grid_point_file):
                        assert i == 0, f"found more than one line in file {grid_point_log_file_path}"
                        data  = json.loads(line.strip())
                        if data == point_in_grid:
                            before_fit_para_path = os.path.join(full_path, "before_fit_para")
                            if os.path.exists(before_fit_para_path):
                                print(f"found point that was already calculated, ie before fit para exists in {full_path}")
                                raise BreakException
        except BreakException:
            # Continue the outer loop
            continue


        args = [single for p in point_in_grid for single in p]
        pass_args = " ".join([f"--{arg_name} {arg}" for arg_name, arg in zip(arg_names, args)])
        pass_args += " " + " ".join([f"--{arg}" for arg in get_cfg_without_args()])

        from Config import get_dir_path_from_fasta_nCodons_and_out_path

        nCodons = args[arg_names.index("nCodons")]
        fasta_path = args[arg_names.index("fasta")]
        run_dir = get_dir_path_from_fasta_nCodons_and_out_path(parsed_args.continue_training, nCodons, fasta_path)

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        err_path = f"{run_dir}/err_train.log"
        out_path = f"{run_dir}/out_train.log"

        err_path = out_path

        # nochmal wenn kein before da ist
        # da das heißt es wurde nicht gerechnet und
        # es liegt nicht daran, dass es erst nach langem training fehl geschlagen ist



        command = f"./main_programm.py {pass_args} --passed_current_run_dir {run_dir}"
        print("running", command)

        path_to_command_file = f"{run_dir}/called_command.log"
        with open(path_to_command_file, "w") as out_file:
            out_file.write(command)
            out_file.write("\n")

        path_to_gridpoint_file = f"{run_dir}/grid_point.log"
        with open(path_to_gridpoint_file, "w") as out_file:
            out_file.write(json.dumps(point_in_grid))
            out_file.write("\n")

        if not parsed_args.slurm:
            # running command and directing out steams
            with open(out_path, "w") as out_handle:
                with open(err_path, "w") as err_handle:
                    exit_code = subprocess.call(re.split("\s+", command), stderr = err_handle, stdout = out_handle)
                    if exit_code != 0:
                        print("exit_code:", exit_code)
        if parsed_args.slurm:
            submission_file_name = f"{run_dir}/slurm_train_submission.sh"

            if not parsed_args.adaptive_ram:
                mem = parsed_args.mem
            else:
                max_len = 0
                for record in SeqIO.parse(fasta_path,"fasta"):
                    max_len = max(max_len, len(record.seq))
                mem = ((max_len//2000)+1) * 5000
                # if mem is greater than 20000 then set partition to vision and otherwise to snowball
                # parsed_args.partition = "vision" if mem > 20000 else "snowball"

            with open(submission_file_name, "w") as file:
                file.write("#!/bin/bash\n")
                file.write(f"#SBATCH -J {os.path.basename(run_dir)[17:21]}\n")
                file.write(f"#SBATCH -N 1\n")
                # file.write(f"#SBATCH -n {parsed_args.mpi}\n")
                file.write(f"#SBATCH --mem {mem}\n")
                file.write(f"#SBATCH --partition={parsed_args.partition}\n")
                file.write(f"#SBATCH -o {out_path}\n")
                file.write(f"#SBATCH -e {err_path}\n")
                file.write(f"#SBATCH -t 12:00:00\n")
                file.write(command)
                file.write("\n")

            def get_number_of_running_slurm_jobs() -> int:
                # run the 'squeue' command and capture the output
                output = subprocess.check_output(['squeue', '-u', "s-mamrzi"])
                # count the number of lines in the output
                num_jobs = len(output.strip().split(b'\n')) - 1 # subtract 1 to exclude the header
                return num_jobs

            while get_number_of_running_slurm_jobs() > parsed_args.max_jobs:
                time.sleep(1)

            os.system(f"sbatch {submission_file_name}")


        if not parsed_args.slurm:
            # if learning fails, due to RAM overflow or keyboard interupt, then this
            # file shoulnd exists
            path_of_A_kernel_after_fit = f"{run_dir}/after_fit_paras/A.kernel"
            if os.path.exists(path_of_A_kernel_after_fit):
                with open(f"{parsed_args.continue_training}/todo_grid_points.json", "w") as grid_point_file:
                    json.dump(grid_points[i+1:], grid_point_file)
            else:
                import signal

                # define a handler for the timeout signal
                def timeout_handler(signum, frame):
                    print("Time's up!")
                    raise TimeoutError

                # set the signal handler for the SIGALRM signal
                signal.signal(signal.SIGALRM, timeout_handler)

                # prompt the user for input with a 30-second timeout
                try:
                    signal.alarm(30) # set the timeout to 30 seconds
                    print(f"the file {path_of_A_kernel_after_fit} doesnt exist after training")
                    print(f"should the dir {run_dir} be removed? [y/n], it will be removed automatically in 30sec")
                    user_input = input("Please enter some input within 30 seconds: ")
                    signal.alarm(0) # disable the alarm
                except TimeoutError:
                    print("No input provided within 30 seconds. rm -rf {run_dir}")
                    os.system(f"rm -rf {run_dir}")
                else:
                    while user_input not in "yn":
                        user_input = input("was not in [y/n]")
                    if user_input == "y":
                        os.system(f"rm -rf {run_dir}")

################################################################################

def get_run_sub_dirs(path):
    print(path)
    subdirs = []
    for subdir in os.listdir(path):
        # check if the subdirectory is a directory
        sub_path = os.path.join(path, subdir)
        if not os.path.isdir(sub_path):
            continue
        regex = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})"
        if not re.match(regex, subdir):
            continue
        cfg_path = f"{sub_path}/passed_args.json"
        if not os.path.exists(cfg_path):
            continue
        subdirs.append(sub_path)
    return subdirs

################################################################################

def viterbi(parsed_args):
    run_sub_dirs = get_run_sub_dirs(args.viterbi_path)
    for i, sub_path in enumerate(run_sub_dirs):
        print(f"calculating viterbi {i}/{len(run_sub_dirs)}")

        # Viterbi.py args:
        # self.parser.add_argument('--only_first_seq', action = 'store_true', help = 'run viterbi only for the first seq')
        # self.parser.add_argument('--parent_input_dir', help = 'path to dir containing the config_attr.json and paratemeters dir used for viterbi')
        # self.parser.add_argument('--in_viterbi_path', help = 'if viteribi is already calculated, path to viterbi file which is then written to the alignment')
        # self.parser.add_argument('--viterbi_threads', type = int, default = 1, help = 'how many threads for viterbi.cc')
        # self.parser.add_argument('--path_to_dir_where_most_recent_dir_is_selected', help = 'path_to_dir_where_most_recent_dir_is_selected')

        # fasta path doesnt need to get calculated since it is saved in the cfg in parent_input_dir

        force_overwrite = True

        overwrite = "--force_overwrite" if force_overwrite else ""
        after_or_before = "--after_or_before b" if args.use_init_weights_for_viterbi else "--after_or_before a"
        command = f"./Viterbi.py --only_first_seq \
                   --parent_input_dir {sub_path} \
                   --viterbi_threads 1 \
                   {overwrite} \
                   {after_or_before}"

        command = re.sub("\s+", " ", command)
        print("runnung: ", command)

        submission_file_name = f"{sub_path}/slurm_viterbi_submission.sh"

        slurm_out_path = f"{sub_path}/slurm_viterbi_out/"
        if not os.path.exists(slurm_out_path):
            os.makedirs(slurm_out_path)

        if args.slurm:
            with open(submission_file_name, "w") as file:
                file.write("#!/bin/bash\n")
                file.write(f"#SBATCH -J v_{os.path.basename(sub_path)[-6:]}\n")
                file.write(f"#SBATCH -N 1\n")
                file.write(f"#SBATCH -n 1\n")
                file.write("#SBATCH --mem 2000\n")
                file.write("#SBATCH --partition=snowball\n")
                file.write(f"#SBATCH -o {sub_path}/slurm_viterbi_out/out.%j\n")
                file.write(f"#SBATCH -e {sub_path}/slurm_viterbi_out/err.%j\n")
                file.write("#SBATCH -t 02:00:00\n")
                file.write(command)
                file.write("\n")
            def get_number_of_running_slurm_jobs() -> int:
                # run the 'squeue' command and capture the output
                output = subprocess.check_output(['squeue', '-u', "s-mamrzi"])
                # count the number of lines in the output
                num_jobs = len(output.strip().split(b'\n')) - 1 # subtract 1 to exclude the header
                return num_jobs

            while get_number_of_running_slurm_jobs() > parsed_args.max_jobs:
                time.sleep(1)

            os.system(f"sbatch {submission_file_name}")
        else:
            print("running", command)
            subprocess.call(re.split("\s+", command))

################################################################################

def get_true_alignemnt_path(train_run_dir, after_or_before):
    return f"{train_run_dir}/true_alignment_{after_or_before}.clw"

################################################################################

def get_viterbi_aligned_seqs(train_run_dir, after_or_before, true_alignemnt_path = "None", no_asserts = False):
    if true_alignemnt_path =="None":
        true_alignemnt_path = get_true_alignemnt_path(train_run_dir, after_or_before)

    try:
        true_alignemnt = read_true_alignment_with_out_coords_seq(true_alignemnt_path)
    except:
        print(f"couldnt read_true_alignment_with_out_coords_seq({true_alignemnt_path})")
        return -1
    if not no_asserts:
        assert len(true_alignemnt) == 3, f"{true_alignemnt_path} doesnt contain the viterbi guess"
    aligned_seqs = {} # reference_seq, true_seq, viterbi_guess
    for seq in true_alignemnt:
        aligned_seqs[seq.id] = seq

    if not no_asserts:
        assert len(aligned_seqs) == 3, "some seqs had same id"

    return aligned_seqs

################################################################################

def add_true_and_guessed_exons_coords_to_run_stats(run_stats, aligned_seqs):
    run_stats["start"] = aligned_seqs["true_seq"].seq.index("E") # inclusive
    run_stats["end"] = aligned_seqs["true_seq"].seq.index("r") # exclusive

    for i in range(len(aligned_seqs["viterbi_guess"].seq)):
        if aligned_seqs["viterbi_guess"].seq[i:i+2] == "AG":
            run_stats["v_start"] = i+2
        if aligned_seqs["viterbi_guess"].seq[i:i+2] == "GT":
            run_stats["v_end"] = i
    if "v_start" not in run_stats:
        run_stats["v_start"] = -1
    if "v_end" not in run_stats:
        run_stats["v_end"] = -1

    if run_stats["v_start"] == -1 or run_stats["v_end"] == -1:
        assert run_stats["v_end"] + run_stats["v_start"] == -2, f"if viterbi didnt find start it is assumend that it also didnt find end, bc i dont want to handle this case, true_alignemnt_path = {true_alignemnt_path}"

################################################################################

def add_actual_epochs_to_run_stats(sub_path, run_stats, max_epochs = None):
    # Epoch 6/20
    path_to_std_out = f"{sub_path}/out_train.log"
    if not os.path.exists(path_to_std_out):
        path_to_std_out = f"{sub_path}/out.log"

    max_epoch = 0
    with open(path_to_std_out, "r") as log_file:
        regrex = f"Epoch\s(\d{{1,3}})/{max_epochs}"
        for line in log_file:
            line = line.strip()
                # print(line)
            if x :=re.search(regrex, line):
                max_epoch = max(int(x.group(1)), max_epoch)
    run_stats["actual_epochs"] = max_epoch

################################################################################

def add_time_and_ram_to_run_stats(run_stats, train_run_dir):
    path_to_bench = f"{train_run_dir}/bench.log"
    RAM_in_mb_peak = 0
    min_time = 0
    max_time = 0
    fitting_time = float("inf")
    with open(path_to_bench, "r") as log_file:
        for line in log_file:
            data = json.loads(line.strip())
            # print("data", data)
            RAM_in_mb_peak = max(RAM_in_mb_peak, data["RAM in kb"]/1024)
            min_time = min(min_time, data["time"])
            max_time= max(max_time, data["time"])
            if re.search("Training:model.fit\(\) end", data["description"]):
                fitting_time = data["time since passed start time"]
    total_time = max_time - min_time

    if fitting_time == float("inf"):
        RAM_in_mb_peak =  float("inf")
        total_time = float("inf")

    run_stats["mbRAM"] = RAM_in_mb_peak
    run_stats["total_time"] = total_time
    run_stats["fitting_time"] = fitting_time

################################################################################
def add_history_stats_to_run_stats(run_stats, train_run_dir):
    best_loss = float("inf")
    aprior = float("inf")
    bprior = float("inf")
    path_to_history = f"{train_run_dir}/history.log"
    if not os.path.exists(path_to_history):
        # path to loss.log
        path_to_history = f"{train_run_dir}/loss.log"
        with open(path_to_history, "r") as log_file:
            for i, line in enumerate(log_file):
                loss = float(line.strip().split(" ")[0])
                if loss < best_loss:
                    best_loss = loss
        return

    with open(path_to_history, "r") as log_file:
        data = json.load(log_file)
        for i,loss in enumerate(data["loss"]):
            if loss < best_loss:
                best_loss = loss
                aprior = data["A_prior"][i]
                bprior = data["B_prior"][i]


        run_stats["best_loss"] = best_loss
        run_stats["A_prior"] = aprior
        run_stats["B_prior"] = bprior
################################################################################
def add_inserts(run_stats, aligned_seqs):
    # find the number if i cahrs in the viterbi guess
    inserts = 0
    for i in range(len(aligned_seqs["viterbi_guess"].seq)):
        if aligned_seqs["viterbi_guess"].seq[i] == "i":
            inserts += 1

    # last_index = 0
    # number_of_deletions = 0
    # for i in range(len(aligned_seqs["viterbi_guess"].seq)):
    #     last_index = aligned_seqs["viterbi_guess"].seq[i]
    #         if aligned_seqs["viterbi_guess"].seq[i] == "i":
    #             inserts += 1



    # add inserts to run_stats
    run_stats["inserts"] = inserts

################################################################################
def calc_run_stats(path) -> pd.DataFrame:

    stats_list = []
    sub_dir_and_after_or_before = list(product(get_run_sub_dirs(path), ["after", "before"]))
    print(sub_dir_and_after_or_before)
    for i, (train_run_dir, after_or_before) in enumerate(sub_dir_and_after_or_before):
        if i %1000 == 0:
            print(f"getting stats {i}/{len(sub_dir_and_after_or_before)}")

        run_stats = {}
        if (aligned_seqs := get_viterbi_aligned_seqs(train_run_dir, after_or_before)) == -1:
            continue

        add_true_and_guessed_exons_coords_to_run_stats(run_stats, aligned_seqs)
        add_inserts(run_stats, aligned_seqs)

        run_stats["after_or_before"] = after_or_before
        run_stats["seq_len_from_multi_run"] = len(aligned_seqs["true_seq"].seq)

        training_args = json.load(open(f"{train_run_dir}/passed_args.json"))
        add_actual_epochs_to_run_stats(train_run_dir, run_stats, max_epochs = training_args['epochs'])
        add_time_and_ram_to_run_stats(run_stats, train_run_dir)
        add_history_stats_to_run_stats(run_stats, train_run_dir)

        d = {**training_args, **run_stats}
        stats_list.append(d)

    df = pd.DataFrame(stats_list)

    # remove cols, ie name of parameter for training runs, whos args are const across runs
    cols_to_keep = df.columns[df.nunique() > 1]
    # cols_discared = list(set(df.columns) - set(cols_to_keep))
    # non_unique_args = df.iloc[0,:][cols_discared]
    # print("non_unique_args", non_unique_args)
    # i want to keep run stats even if those results are const across runs
    cols_to_keep = list(set(list(cols_to_keep) + list(run_stats.keys())))
    df = df[cols_to_keep]

    df["fasta"] = df["fasta_path"].apply(os.path.dirname).apply(os.path.basename)
    df["passed_current_run_dir"] = df["passed_current_run_dir"].apply(os.path.basename)

    return df

################################################################################

def load_cfg_with_args(args) -> json:
    path_to_multi_run_dir = f"{args.eval_viterbi}/cfg_with_args.json"
    with open(path_to_multi_run_dir, "r") as file:
        cfg_with_args = json.load(file)
    return cfg_with_args

################################################################################

def get_number_of_total_exons(args):
    cfg_with_args = load_cfg_with_args(args)
    return len(cfg_with_args["fasta"])

################################################################################

def get_cols_to_group_by(args):

    cfg_with_args = load_cfg_with_args(args)

    parameters_with_more_than_one_arg = [name for name, args in cfg_with_args.items() if len(args) > 1]
    parameters_with_less_than_one_arg = [name for name, args in cfg_with_args.items() if len(args) <= 1]

    parameter_that_are_not_in_df = [(name, cfg_with_args[name][0]) for name in parameters_with_less_than_one_arg]

    # these should not be grouped over, since i want the mean over these
    parameters_with_more_than_one_arg.remove("fasta")
    parameters_with_more_than_one_arg.remove("nCodons")

    parameters_with_more_than_one_arg += ["after_or_before"]

    return parameters_with_more_than_one_arg, parameter_that_are_not_in_df


def add_additional_eval_cols(df, args):

    df["p_nt_on_exon"]           = df["end"] -    df["start"]
    df["predicted_p_nt_on_exon"] = df["v_end"] - df["v_start"]


    df["deletes"] = df["nCodons"] - np.ceil(df["predicted_p_nt_on_exon"] / 3) + df["inserts"]
    # df[df["after_or_before"]=="after"][["inserts","deletes","passed_current_run_dir","nCodons", "predicted_p_nt_on_exon"]].sample(20)


    df["n_nt_on_exon"]           = df["seq_len_from_multi_run"] - df["p_nt_on_exon"]
    df["predicted_n_nt_on_exon"] = df["seq_len_from_multi_run"] - df["predicted_p_nt_on_exon"]

    df["len_ratio"] = df["predicted_p_nt_on_exon"] / df["p_nt_on_exon"]
    df["true_left"] = df["start"] == df["v_start"]
    df["true_right"] = df["end"] == df["v_end"]
    # are the coords coorect, ie guessed end and true and are both inclusice or exclusive?

    def overlap(row):
        start_overlap = max(row["start"], row["v_start"])
        end_overlap = min(row["end"], row["v_end"])
        return max(0, end_overlap - start_overlap)
    df['tp_nt_on_exon'] = df.apply(lambda row: overlap(row), axis=1)
    df['fp_nt_on_exon'] = df["predicted_p_nt_on_exon"] - df["tp_nt_on_exon"]

    df['fn_nt_on_exon'] = df["p_nt_on_exon"] - df["tp_nt_on_exon"]
    df['tn_nt_on_exon'] = df["seq_len_from_multi_run"] - df["fn_nt_on_exon"] -df["tp_nt_on_exon"]  - df["fp_nt_on_exon"]

    df['sn_nt_on_exon'] = df["tp_nt_on_exon"] / df["p_nt_on_exon"]
    # if there are way more real negative than positives
    # df['sp_nt_on_exon'] = df["tp_nt_on_exon"] / df["predicted_p_nt_on_exon"]
    df['sp_nt_on_exon'] = df["tp_nt_on_exon"] / (df["tp_nt_on_exon"] + df["fn_nt_on_exon"])

    df['f1_nt_on_exon'] =  2 * df["sn_nt_on_exon"] * df["sp_nt_on_exon"] / (df["sn_nt_on_exon"] + df["sp_nt_on_exon"])
    df['f1_nt_on_exon'] = df['f1_nt_on_exon'].fillna(0)


    cols_to_group_by, _ = get_cols_to_group_by(args)
    print("cols_to_group_by", cols_to_group_by)
    print("df.columns", df.columns)
    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["p_nt_on_exon"])).reset_index(name = "p_nt")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["n_nt_on_exon"])).reset_index(name = "n_nt")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["predicted_p_nt_on_exon"])).reset_index(name = "predicted_p_nt")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["predicted_n_nt_on_exon"])).reset_index(name = "predicted_n_nt")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["tp_nt_on_exon"])).reset_index(name = "tp_nt")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["fp_nt_on_exon"])).reset_index(name = "fp_nt")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["tn_nt_on_exon"])).reset_index(name = "tn_nt")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["fn_nt_on_exon"])).reset_index(name = "fn_nt")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    df["sn_nt"] = df["tp_nt"] / df["p_nt"]
    df["sp_nt"] = df["tp_nt"] / df["predicted_p_nt"]
    df["f1_nt"] = 2 * df["sn_nt"] * df["sp_nt"] / (df["sn_nt"] + df["sp_nt"])

    df["MCC_numerator"] = df["tp_nt"] * df["tn_nt"] - df["fp_nt"] * df["fn_nt"]
    # df["MCC_denominator_no_sqrt"] = (df["tp_nt"] + df["fp_nt"]) * (df["tp_nt"] + df["fn_nt"]) * (df["tn_nt"] + df["fp_nt"]) * (df["tn_nt"] + df["fn_nt"])
    # df["MCC_denominator"] = np.sqrt(df["MCC_denominator_no_sqrt"])
    df["MCC_denominator"] = np.sqrt((df["tp_nt"] + df["fp_nt"])) \
                          * np.sqrt((df["tp_nt"] + df["fn_nt"])) \
                          * np.sqrt((df["tn_nt"] + df["fp_nt"])) \
                          * np.sqrt((df["tn_nt"] + df["fn_nt"]))
    df["MCC"] = df["MCC_numerator"] / df["MCC_denominator"]

    t1 = df["tp_nt"] / (df["tp_nt"] + df["fn_nt"])
    t2 = df["tp_nt"] / (df["tp_nt"] + df["fp_nt"])
    t3 = df["tn_nt"] / (df["tn_nt"] + df["fp_nt"])
    t4 = df["tn_nt"] / (df["tn_nt"] + df["fn_nt"])
    df["ACP"] = (t1 + t2 + t3 + t4) / 4

    # that cehckts and then assert that all values are true
    cnew_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["fn_nt_on_exon"])).reset_index(name = "fn_nt")

    df["skipped_exon"] = df.apply(lambda x: True if x["v_start"] == -1 and x["v_end"] == -1 else False, axis=1)
    new_col = df.groupby(cols_to_group_by)["skipped_exon"].mean().reset_index(name = "skipped_exon_mean")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    df["left_miss"] = (df["v_end"] < df["start"]) & (~ df["skipped_exon"])
    df["right_miss"] = (df["v_start"] > df["end"]) & (~ df["skipped_exon"])
    df["miss"] = df["left_miss"] | df["right_miss"]

    df["correct"] = (df["v_start"] == df["start"]) & (df["v_end"] == df["end"])
    df["wrap"] = (df["v_start"] <= df["start"]) & (df["v_end"] >= df["end"]) & (~ df["correct"])
    df["incomplete"] = (df["v_start"] >= df["start"]) & (df["v_end"] <= df["end"]) & (~ df["correct"])

    df["overlaps_left"] = (df["v_start"] < df["start"]) & (df["v_end"] > df["start"]) & (~ df["wrap"])
    df["overlaps_right"] = (df["v_start"] < df["end"]) & (df["v_end"] > df["end"]) & (~ df["wrap"])

    df["overlaps"] = df["overlaps_left"] | df["overlaps_right"]

    df["WEScore"] = (1 - df["correct"] - df["skipped_exon"]) / (1- df["skipped_exon"])

    df["5prime"] = df["true_left"] / (1- df["skipped_exon"])
    df["3prime"] = df["true_right"] / (1- df["skipped_exon"])

    df["time_per_epoch"] = df["fitting_time"]/df["actual_epochs"]

    df["f1"] = 2*df["correct"] / (2- df["skipped_exon"])

    # MEScore
    # total_number_of_true_exons = get_number_of_total_exons(args)
    # new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["fn_nt_on_exon"])).reset_index(name = "ME")
    # df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    # WEScore

    change_type_to_int = ["tp_nt", "tn_nt", "v_start", "v_end", "start", "end", "actual_epochs","tp_nt_on_exon", "fp_nt_on_exon", "fn_nt_on_exon", "tn_nt_on_exon"]
    for col in change_type_to_int:
        df[col] = df[col].astype(int)

    return df

def sort_columns(df):
    sorted_columns = ["passed_current_run_dir", "actual_epochs", \
                      "start", "end", "v_start", "v_end", \
                      "p_nt_exon", "predicted_p_nt_exon",\
                      "correct", "skipped_exon", "wrap", "incomplete", "overlaps", "miss", \
                      "tp_nt_on_exon", "sn_nt_on_exon", "sn_nt", \
                      "after_or_before", "priorA", "priorB", "exon_skip_init_weight"]

    # bc sometimes i dont have multiple values for a parameter, so it is removed
    # from df in get_run_stats()
    for key in sorted_columns[::-1]:
        if key not in df.columns:
            sorted_columns.remove(key)
    remaining_columns = list(set(df.columns) - set(sorted_columns))
    df = df[sorted_columns + remaining_columns]
    return df

# def rename_cols(df):
#     columns = {"start": "start",
#                "end": "end",
#                "v_start": "v_start",
#                "v_end":"v_end"}
#     df = df.rename(columns = columns)
#     return df

# def remove_cols(df):

#     cols = ["fasta_path"]
#     for col in cols:
#         df = df.drop(col, axis = 1)

    return df
def eval_viterbi(args):
    path = args.eval_viterbi
    df = load_or_calc_eval_df(path)
    loaded_cols = df.columns

    # print(df.groupby(["priorA", "priorB", "exon_skip_init_weight", "fasta"]).apply(np.std))
    # print(df.groupby(["priorA", "priorB", "exon_skip_init_weight", "fasta"]).size())

    # print(df.groupby(get_cols_to_group_by(args)).mean()[["correct", "sn", "sp", "f1"]].sort_values("f1").to_string(max_rows = None, max_cols = None))

    # df[df["predicted_p_nt_exon"] != 0].groupby(["priorA", "model_size_factor"]).apply(lambda x: x["predicted_p_nt_exon"] / x["true_p_nt_exon"]).mean(level=["priorA", "model_size_factor"])

    df = add_additional_eval_cols(df, args)
    df = sort_columns(df)
    # df = rename_cols(df)
    # df = remove_cols(df)
    added_cols = list(set(df.columns) - set(loaded_cols))

    #TODO also rename grouped?

    _, parameters_with_less_than_one_arg = get_cols_to_group_by(args)

    cols_to_group_by, _ = get_cols_to_group_by(args)
    eval_cols = ["sn_nt", "sp_nt", "f1_nt", "MCC", "ACP", "correct", "true_left", "true_right", "incomplete", "wrap",  "overlaps", "miss", "skipped_exon", "5prime", "3prime", "WEScore"]
    grouped = df.groupby(cols_to_group_by).mean()[eval_cols].reset_index().sort_values("f1_nt")


    # Create a new column 'lens_group' by assigning the respective bin to each lens value
    # df['total_len'] = pd.cut(df['seq_len_from_multi_run '], bins=bins, labels=False)

    # add a new column right intron len
    df["right_intron_len"] = df["seq_len_from_multi_run"] - df["end"]
    # add a new column left intron len
    df["left_intron_len"] = df["start"]

    # for heat map
    grouped["after_or_before"] = (grouped["after_or_before"] == "after").astype(int)


    g1 = df.groupby(cols_to_group_by).mean()[eval_cols].reset_index()
    print('g1[(g1["priorA"] == 0) & (g1["epochs"] == 20) & (g1["exon_skip_init_weight"] == -4)]')


    # anova_for_one_hyper(df, "priorA", cols_to_group_by)

    return df, grouped, parameters_with_less_than_one_arg, eval_cols, loaded_cols, added_cols, cols_to_group_by

def anova_for_one_hyper(df, group, hyper_grid_para, predicted_column = "f1_nt_on_exon"):


    # maybe if paras are binded like prior a and prior b i must pass a list ["priorA", "priorB"] to group


    '''
    group is for example priorA with levels 0, 5, 20
    hypergird are the other hyperparameters including group
    '''

    print(f"\n\nANOVA for {group}\n\n")
    from scipy import stats
    missing_one_hyper_para_col = list(set(hyper_grid_para) - set([group]))
    # print("missing_one_hyper_para_col", missing_one_hyper_para_col)
    grouped_data = df.groupby(missing_one_hyper_para_col)
    # print("grouped_data.mean", grouped_data.mean(), sep = "\n")

    for group_name, group_df in grouped_data:
        print("Group:", list(zip(missing_one_hyper_para_col, group_name)))
        # print("group", group)
        # print("group_df", group_df.reset_index()[group].unique(), sep="\n")
        group_values = []
        for level_name, level_df in group_df.groupby(group):
            # print("level_name", level_name)
            # print("level_df", level_df)
            # print("level_df[predicted_column]", level_df[predicted_column].values)
            group_values.append(level_df[predicted_column].values)

        # print("group_values", group_values)
        gro = df.groupby(missing_one_hyper_para_col + [group]).mean()["f1_nt"].reset_index()
        # print(gro)
        mask = pd.Series(True, index=gro.index)

        for column, value in list(zip(missing_one_hyper_para_col, group_name)):
            mask &= gro[column] == value
        filtered_df = gro[mask]
        print("this is f1_nt accorss exon. the test is done every exon as its own data point")
        print(filtered_df)

        try:
            # Perform the ANOVA test
            f_statistic, p_value = stats.f_oneway(*group_values)
            # Print the results for each group
            print("F-Statistic:", f_statistic)
            print("p-value:", p_value)
            print("------------------------")
        except Exception:
            print("in except")
            print(traceback.format_exc())

################################################################################

def heatmap_grouped(grouped, figsize = (15,7), angle1 = 90, angle2 = 60, eval_cols = None):

    start = time.perf_counter()
    print(f"staring to make heatmap")
    # heatmap_columns = eval_cols
    # heatmap = sns.heatmap(grouped[heatmap_columns], cmap='YlGnBu')
    # sns.heatmap(grouped, cmap='Blues', annot=True, cbar=False, alpha=0, ax=heatmap)
    # plt.savefig("eval.png")

    fig, axes = plt.subplots(nrows=1, ncols=len(grouped.columns), figsize = figsize)

    # Iterate over each column in the DataFrame and create a heatmap in the corresponding subplot
    for i, column in enumerate(grouped.columns):
        print(f"column {i}/{len(grouped.columns)}")
        sns.heatmap(grouped[[column]], cmap='YlGnBu', annot=True, fmt=".2f", cbar=False, ax=axes[i])
        axes[i].set_title(column)  # Set the title as the column name
        if not eval_cols is None and column in eval_cols:
            axes[i].title.set_rotation(angle2)
        else:
            axes[i].title.set_rotation(angle1)

        axes[i].yaxis.set_visible(False)
        axes[i].xaxis.set_visible(False)
    plt.subplots_adjust(wspace=0)
    # plt.tight_layout()  # Adjust the spacing between subplots
    plt.savefig('heatmap.png', bbox_inches='tight')
    plt.close()

    print(f"made heatmap, it took {np.round(time.perf_counter() - start,3)}")

################################################################################

def load_or_calc_eval_df(path):
    path_to_out_csv = f"{path}/eval.csv"
    if os.path.exists(path_to_out_csv):
        print("an eval df exists. Should it be loaded [l] or be recalculated [r]")
        while (x := input()) not in "lr":
            print("input must be [l/r]")
            pass
        if x == "l":
            df = pd.read_csv(path_to_out_csv, index_col = 0)
        if x == "r":
            df = calc_run_stats(path)
            df.to_csv(path_to_out_csv, header = True)
    else:
        df = calc_run_stats(path)
        df.to_csv(path_to_out_csv, header = True)
    return df

# pd.set_option('display.max_columns', None)
# pd.options.display.width = 0
# pd.set_option("display.max_rows", None)
# df.sort_values(by = "true_p_nt_exon", ascending = 1)

################################################################################
def lin_reg():

    x1 = "nCodons"
    x2 = "seq_len_from_multi_run"
    ys = ["mbRAM", "fitting_time", "time_per_epoch"]

    X = df[[x1, x2]]

    X.loc[:,x1] /= X[x1].max()
    X.loc[:,x2] /= X[x2].max()
    X['intercept'] = 1

    for y in ys:
        print(y)
        ydata = df[y] / df[y].max()

        # Add quadratic terms
        X.loc[:,'x2'] = X[x1] ** 2
        X.loc[:,'y2'] = X[x2] ** 2

        # Add a column of ones to X for the intercept term

        # Compute the regression coefficients using the normal equation (X'X)^-1 X'y
        coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ydata)

        # Retrieve the coefficients and intercept
        intercept = coefficients[-1]
        coefficients = coefficients[:-1]

        # Print the coefficients and intercept
        print(x1,x2,f"{x1}^2", f"{x2}^2")
        print('Coefficients:', coefficients)
        print('Intercept:', intercept)

        print()

################################################################################

def len_groups(df, nbins = 10, mask = [1,1,1,1], eval_cols = None):

    if eval_cols is None:
        eval_cols = ["fn_nt_on_exon", "correct", "miss", "true_left", "true_right", "total_time", "mbRAM"]

    groupby = ["total_len_group", "right_intron_len_group", "left_intron_len_group", "exon_len_group"]
    groupby = [group for i, group in enumerate(groupby) if mask[i]]

    # bins for right intron len, resulting in nbins groups
    bins_right_intron = list(range(0, df["right_intron_len"].max() + 1, int(df["right_intron_len"].max()/nbins)))
    # same for left intron
    bins_left_intron = list(range(0, df["left_intron_len"].max() + 1, int(df["left_intron_len"].max()/nbins)))
    # and the same for total len resulting in nbins groups
    bins_total_len = list(range(0, df["seq_len_from_multi_run"].max() + 1, int(df["seq_len_from_multi_run"].max()/nbins)))
    # add a column exon len
    df["exon_len"] = df["end"] - df["start"]
    # add the same for exon len
    bins_exon_len = list(range(0, df["exon_len"].max() + 1, int(df["exon_len"].max()/nbins)))

    # now cut df into the bins adding a column for each
    df['right_intron_len_group'] = pd.cut(df['right_intron_len'], bins=bins_right_intron, labels=bins_right_intron[:-1])
    df['left_intron_len_group'] = pd.cut(df['left_intron_len'], bins=bins_left_intron, labels=bins_left_intron[:-1])
    df['total_len_group'] = pd.cut(df['seq_len_from_multi_run'], bins=bins_total_len, labels=bins_total_len[:-1] )
    df['exon_len_group'] = pd.cut(df['exon_len'], bins=bins_exon_len, labels=bins_exon_len[:-1])

    # what is the label parameter of df.cut() for?


    # now group by the groups
    len_groups = df.groupby(groupby).mean()[eval_cols].reset_index()

    return len_groups

################################################################################

def plot_model_size_factor_vs_best_loss_f1s(df, figsize = (20,7), angle1 = 90, angle2 = 60, eval_cols = None):
    # as x axis i want model size factor which is categorical
    # then for every model size factor i want 3 box plots, one for best loss one for f1_nt and one for f1
    # i have a pandas df with columns model_size_factor, best_loss, f1_nt, f1

    def scale_0_to_1(column):
        return (column - column.min()) / (column.max() - column.min())

    local_df = df.copy()


    # select only the after or before == after
    local_df = local_df[local_df["after_or_before"] == "after"]

    # add a new column called measure that is the sum of the prior a and prior b but devided by the model size factor



    # add a column called py that is best_loss - a prior - b prior
    local_df["py"] = local_df["best_loss"] - local_df["A_prior"] - local_df["B_prior"]

    columns_to_scale = ["best_loss", "f1_nt_on_exon", "A_prior", "B_prior", "py"]


    make_bigger = 0.01
    for column in columns_to_scale:
        local_df[f"{column}_scaled"] = scale_0_to_1(local_df[column])
        local_df[f"{column}_scaled"] = local_df[f"{column}_scaled"] + np.random.choice([-make_bigger, make_bigger], size=len(local_df))

    # add a measure that is a prior_scaled + b prior_sclade / model size factor
    local_df["measure"] = (local_df["A_prior_scaled"] + local_df["B_prior_scaled"]) / local_df["nCodons"]
    local_df["measure"] = (local_df["inserts"] + local_df["deletes"])
    local_df["inserts"] = local_df["inserts"]
    local_df["deletes"] = local_df["deletes"]

    measure_names = ["measure", "inserts", "deletes"]
    for column in measure_names:
        local_df[f"{column}_scaled"] = scale_0_to_1(local_df[column])


    columns_single_points_in_plot = ["f1", "f1_nt"]
    columns_single_points_in_plot = ["f1"]

    # i want to group by model size factor and add the mean of f1 for every group to a new column
    for column in columns_single_points_in_plot:
        # local_df[f"{column}_scaled"] = local_df.groupby("model_size_factor").mean()[column].reset_index()[column]
        # do similar to the line above but with transform mean
        local_df[f"{column}_scaled"] = local_df.groupby("model_size_factor")[column].transform("mean")
        print(local_df[f"{column}_scaled"])
        print(local_df[f"{column}_scaled"].value_counts())
        local_df[f"{column}_scaled"] = local_df[f"{column}_scaled"] + np.random.choice([-make_bigger, make_bigger], size=len(local_df))


    value_vars = [f"{c}_scaled" for c in columns_to_scale + columns_single_points_in_plot + measure_names]

    for column in value_vars:
        print(column, local_df[column].min(), local_df[column].max())

    df_melt = pd.melt(local_df, id_vars='model_size_factor', value_vars= value_vars)

    plt.figure(figsize=figsize)
    # df_melt_filtered = df_melt[df_melt['variable'].isin(['f1_nt_scaled', 'best_loss_scaled'])]
    sns.boxplot(x='model_size_factor', y='value', hue='variable', data=df_melt)
    # can you add texture to the boxplot?




    # df_f1 = df_melt[df_melt['variable'] == 'f1'].groupby('model_size_factor', as_index=False).mean()

    # sns.scatterplot(x='model_size_factor', y='value', hue = 'value', data=df_f1, color='red', zorder=5)

    # df_f1 = df_melt[df_melt['variable'] == 'f1']
    # means = df_f1.groupby('model_size_factor', as_index=False).mean()
    # means['variable'] = 'f1'  # add variable column to match boxplot
    # sns.scatterplot(x='model_size_factor', y='value', hue='variable', data=means, color='red', zorder=5)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    # sns.boxplot(x='model_size_factor', y='value', hue='variable', data=df_melt)
    plt.savefig('model_size_factor_vs_best_loss_f1s.png', bbox_inches='tight')
################################################################################
def plot_two_columns_box_plot(columns):
    plt.boxplot(columns)
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Boxplot of Column1 and Column2')
    plt.savefig("2box_plots.png")


################################################################################
def toggle_col():
    pass

def toggle_row():
    pass

if __name__ == "__main__":

    args = get_multi_run_config()

    if args.train:
        run_training(args)
    elif args.viterbi_path:
        viterbi(args)
    elif args.eval_viterbi:
        df, grouped, parameter_that_are_not_in_df, eval_cols, loaded_cols, added_cols, cols_to_group_by = eval_viterbi(args)
    elif args.continue_training:
        continue_training(args)