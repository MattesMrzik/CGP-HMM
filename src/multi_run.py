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

import sys
sys.path.append("../scripts")
from read_true_alignment_with_out_coords_seq import read_true_alignment_with_out_coords_seq

out_path = "../../cgp_data"

def get_multi_run_config():

    parser = argparse.ArgumentParser(description='Run multiple trainings with different hyperparameters on cluster with slurm.')
    parser.add_argument('--train', action = 'store_true', help='Train with args specified in the methods in this module and write output to calculated path based on current time.')
    parser.add_argument('--mem', type = int, default = 20000, help='Max memory for slurm training.')
    parser.add_argument('--adaptive_ram', action = "store_true", help='Adaptive max memory for slurm training.')
    parser.add_argument('--max_jobs', type = int, default = 8, help='Max number of jobs for slurm training.')
    parser.add_argument('--partition', default = "snowball", help='Partition for slurm training')
    parser.add_argument('--continue_training', help='Path to multi_run directory for which to continue training, ie not the training for a particular grid point but the training of all grid points.')

    parser.add_argument('--viterbi_path', help='Path to multi_run directory for which to run Viterbi.')
    parser.add_argument('--use_init_weights_for_viterbi', action = 'store_true', help = 'Use the initial weights instead of the learned ones.')
    parser.add_argument('--eval_viterbi', help='Path to multi_run directory for which to evaluate Viterbi.')

    args = parser.parse_args()

    if args.use_init_weights_for_viterbi:
        assert args.viterbi_path, "if you pass --use_init_weights_for_viterbi. you must also pass --viterbi_path"

    assert args.train or args.viterbi_path or args.eval_viterbi or args.continue_training, "you must pass either --train or --viterbi_path or --eval_viterbi"
    return args


################################################################################

def get_cfg_with_args():
    cfg_with_args = {}

    # fasta_dir_path = "/home/s-mamrzi/cgp_data/train_data_set/new_train/good_exons_1"
    # fasta_dir_path = "/home/s-mamrzi/cgp_data/eval_data_set/good_exons_1_new"

    fasta_dir_path = "/home/s-mamrzi/cgp_data/eval_data_set/good_exons_2"

    exons = [dir for dir in os.listdir(fasta_dir_path) if not os.path.isfile(os.path.join(fasta_dir_path, dir)) ]


    # cfg_with_args["fasta"] = [f"{fasta_dir_path}/{exon}/introns/combined.fasta" for exon in exons]

    cfg_with_args["fasta"] = [f"{fasta_dir_path}/{exon}/combined.fasta" for exon in exons]

    get_exon_len = lambda exon_string: (int(exon_string.split("_")[-1]) - int(exon_string.split("_")[-2]))
    get_exon_codons = lambda exon_string: get_exon_len(exon_string) // 3

    exon_nCodons = [get_exon_codons(exon) for exon in exons]
    cfg_with_args["nCodons"] = exon_nCodons

    # cfg_with_args["model_size_factor"] = [0.75, 0.875, 1, 1.125, 1.25]
    cfg_with_args["model_size_factor"] = [1]

    cfg_with_args["priorA"] = [10]
    cfg_with_args["priorB"] = [10]
    # cfg_with_args["ll_growth_factor"] = [0.2 ,0]

    cfg_with_args["akzeptor_pattern_len"] = [5]
    cfg_with_args["donor_pattern_len"] = [5]

    cfg_with_args["global_log_epsilon"] = [1e-20]
    cfg_with_args["epochs"] = [30]

    cfg_with_args["learning_rate"] = [0.01]
    cfg_with_args["batch_size"] = [16]
    cfg_with_args["clip_gradient_by_value"] = [4]
    cfg_with_args["exon_skip_init_weight"] = [-1,2]
    cfg_with_args["optimizer"] = ["Adam"]


    cfg_with_args["left_intron_init_weight"] = [4.35] # this is usually 4 but left miss was way more often than right miss in eval data set

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
    # bind_args_together += [{"priorA", "ll_growth_factor"}]
    bind_args_together += [{"akzeptor_pattern_len", "donor_pattern_len"}]

    return bind_args_together

################################################################################

def get_cfg_without_args():
    cfg_without_args = '''
    use_thesis_weights
    exit_after_loglik_is_nan
    '''
    # bucket_by_seq_len
    # exon_skip_const

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


    print("Length grid_points =", len(grid_points))

    print("Do you want to continue enter? [y/n]")
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
                        assert i == 0, f"Found more than one line in file {grid_point_log_file_path}"
                        data  = json.loads(line.strip())
                        if data == point_in_grid:
                            before_fit_para_path = os.path.join(full_path, "before_fit_para")
                            if os.path.exists(before_fit_para_path):
                                print(f"Found point that was already calculated, ie before fit para exists in {full_path}")
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

        submission_file_name = f"{run_dir}/slurm_train_submission.sh"

        if not parsed_args.adaptive_ram:
            mem = parsed_args.mem
        else:
            max_len = 0
            for record in SeqIO.parse(fasta_path,"fasta"):
                max_len = max(max_len, len(record.seq))
            mem = ((max_len//2000)+1) * 5000

        with open(submission_file_name, "w") as file:
            file.write("#!/bin/bash\n")
            file.write(f"#SBATCH -J {os.path.basename(run_dir)[17:21]}\n")
            file.write(f"#SBATCH -N 1\n")
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

        with open(submission_file_name, "w") as file:
            file.write("#!/bin/bash\n")
            file.write(f"#SBATCH -J v_{os.path.basename(sub_path)[-6:]}\n")
            file.write(f"#SBATCH -N 1\n")
            file.write(f"#SBATCH -n 1\n")
            file.write("#SBATCH --mem 6000\n")
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
        print(f"Couldnt read_true_alignment_with_out_coords_seq({true_alignemnt_path})")
        return -1
    if not no_asserts:
        assert len(true_alignemnt) == 3, f"{true_alignemnt_path} Doesnt contain the Viterbi guess"
    aligned_seqs = {} # reference_seq, true_seq, viterbi_guess
    for seq in true_alignemnt:
        aligned_seqs[seq.id] = seq

    if not no_asserts:
        assert len(aligned_seqs) == 3, "Some seqs had same id"

    return aligned_seqs

################################################################################

def add_true_and_guessed_exons_coords_to_run_stats(run_stats, aligned_seqs):

    try:
        run_stats["start"] = aligned_seqs["true_seq"].seq.index("E") # inclusive
        run_stats["end"] = aligned_seqs["true_seq"].seq.index("r") # exclusive
    except:
        run_stats["start"] = -2
        run_stats["end"] = -2

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

def actual_epochs_to_run_stats(sub_path, max_epochs = None):
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
    return max_epoch

################################################################################

def time_and_ram_to_run_stats(train_run_dir):
    path_to_bench = f"{train_run_dir}/bench.log"
    RAM_in_mb_peak = 0
    min_time = 0
    max_time = 0
    fitting_time = float("inf")
    epoch_times = []

    last_epoch_start = 0
    with open(path_to_bench, "r") as log_file:
        for line in log_file:
            data = json.loads(line.strip())
            # print("data", data)
            RAM_in_mb_peak = max(RAM_in_mb_peak, data["RAM in kb"]/1024)
            min_time = min(min_time, data["time"])
            max_time= max(max_time, data["time"])
            if re.search("Training:model.fit\(\) end", data["description"]):
                fitting_time = data["time since passed start time"]

            if data["description"] == "epoch_end_callback":
                epoch_end = data["time"]
                if last_epoch_start != 0:
                    epoch_times.append(epoch_end - last_epoch_start)
                    last_epoch_start = 0

            if data["description"] == "epoch_begin_callback":
                last_epoch_start = data["time"]

    total_time = max_time - min_time

    epoch_times = epoch_times[1:]
    try:
        mean_epoch_time = sum(epoch_times)/len(epoch_times)
    except:
        return {"mbRAM": np.nan, "total_time": np.nan, "fitting_time": np.nan, "mean_epoch_time": np.nan}

    if fitting_time == float("inf"):
        return {"mbRAM": np.nan, "total_time": np.nan, "fitting_time": np.nan, "mean_epoch_time": np.nan}
    return {"mbRAM": RAM_in_mb_peak, "total_time": total_time, "fitting_time": fitting_time, "mean_epoch_time": mean_epoch_time}

################################################################################
def history_stats(train_run_dir):
    best_loss = float("inf")
    aprior = float("inf")
    bprior = float("inf")
    path_to_history = f"{train_run_dir}/history.log"
    if not os.path.exists(path_to_history):
        # path to loss.log
        path_to_history = f"{train_run_dir}/loss.log"
        if not os.path.exists(path_to_history):
            return {"best_loss": -1, "A_prior": -1, "B_prior": -1}
        with open(path_to_history, "r") as log_file:
            for i, line in enumerate(log_file):
                loss = float(line.strip().split(" ")[0])
                if loss < best_loss:
                    best_loss = loss
        return {"best_loss": best_loss, "A_prior": -1, "B_prior": -1}

    with open(path_to_history, "r") as log_file:
        data = json.load(log_file)
        for i,loss in enumerate(data["loss"]):
            if loss < best_loss:
                best_loss = loss
                aprior = data["A_prior"][i]
                bprior = data["B_prior"][i]

    return {"best_loss": best_loss, "A_prior": aprior, "B_prior": bprior}
################################################################################
def add_inserts(run_stats, aligned_seqs):
    # calculate the number of inserts

    # find the number if i chars in the viterbi guess
    inserts = 0
    for i in range(len(aligned_seqs["viterbi_guess"].seq)):
        if aligned_seqs["viterbi_guess"].seq[i] == "i":
            inserts += 1
    run_stats["inserts"] = inserts
################################################################################
def augustus_stats(train_run_dir):
    # evaluate the augustus output

    # seach for the option that was passed to --fasta in the file called_command.log that is in the train_run_dir
    fasta_path = get_dir_from_called_command_log_in_run_dir_to_base_dir_of_fasta(train_run_dir)

    path_to_augustus = f"{fasta_path}/augustus.out"

    if not os.path.exists(path_to_augustus):
        return {"augustus_start": -1, "augustus_end": -1}
    df = pd.read_csv(path_to_augustus, sep = "\t")
    df = df[df["feature"] == "CDS"]
    # if there is no such row add -1 to run_stats with names "augustus_start" and "augustus_end"
    if len(df) == 0:
        return {"augustus_start": -1, "augustus_end": -1}

    path_to_hints = f"{fasta_path}/augustus_hints.gff"
    if os.path.exists(path_to_hints):
        pd_hints = pd.read_csv(path_to_hints, sep = "\t", header = None)

        pd_hints = pd_hints[pd_hints[2] == "CDSpart"]
        exon_middle = pd_hints[3] + 1

    number_of_augustus_predictions = len(df)

    if len(df) > 1:
        # select the prediction that that overlaps with the middle of the exon
        df = df[(df["end"]>exon_middle.item()) & (df["start"]< exon_middle.item())]

    return {"augustus_start": df["start"].item() -1, "augustus_end": df["end"].item(), "number_of_augustus_predictions": number_of_augustus_predictions}

################################################################################

def get_dir_from_called_command_log_in_run_dir_to_base_dir_of_fasta(train_run_dir):
    path_to_called_command = f"{train_run_dir}/called_command.log"
    with open(path_to_called_command, "r") as log_file:
        for line in log_file:
            if x := re.search("--fasta\s(\S+)", line):
                fasta_path = x.group(1)
                break

    # from the fasta path extract the path that ends in the dir called something like exon_chr1_54853433_54853612
    # using a while loop and os dirname until the basename matches the regex
    while not re.match("exon_chr.+?_\d+_\d+", os.path.basename(fasta_path)):
        fasta_path = os.path.dirname(fasta_path)
    return fasta_path

################################################################################
def get_sum_of_seq_lens(run_dir):
    dir_of_fasta = get_dir_from_called_command_log_in_run_dir_to_base_dir_of_fasta(run_dir)
    path_to_combined_fasta = os.path.join(dir_of_fasta, "combined.fasta")
    # read fasta file
    seqs = []
    for record in SeqIO.parse(path_to_combined_fasta, "fasta"):
        seqs.append(record.seq)
    sum_of_lens = sum([len(seq) for seq in seqs])
    max_of_lens = max([len(seq) for seq in seqs])
    return {"sum_of_lens": sum_of_lens, "max_of_lens": max_of_lens}

################################################################################
def calc_run_stats(path) -> pd.DataFrame:

    stats_list = []
    run_sub_dirs = get_run_sub_dirs(path)
    for i, train_run_dir in enumerate(run_sub_dirs):
        if i %100 == 0:
            print(f"getting stats {i}/{len(run_sub_dirs)}")

        training_args = json.load(open(f"{train_run_dir}/passed_args.json"))
        actual_epochs =  actual_epochs_to_run_stats(train_run_dir, max_epochs = training_args['epochs'])
        ram_and_time_dict = time_and_ram_to_run_stats(train_run_dir)
        history_stats_dict = history_stats(train_run_dir)
        sum_of_seq_lens_dict = get_sum_of_seq_lens(train_run_dir)
        aligned_seqs_dict = {}
        for after_or_before in ["after", "before", "augustus"]:
            run_stats = {}

            if after_or_before == "augustus":
                if "after" not in aligned_seqs_dict:
                    continue
                aligned_seqs = aligned_seqs_dict["after"]
            else:
                if (aligned_seqs := get_viterbi_aligned_seqs(train_run_dir,after_or_before)) == -1:
                    continue
                aligned_seqs_dict[after_or_before] = aligned_seqs

            add_true_and_guessed_exons_coords_to_run_stats(run_stats, aligned_seqs)
            add_inserts(run_stats, aligned_seqs)

            run_stats["after_or_before"] = after_or_before
            run_stats["seq_len_from_multi_run"] = len(aligned_seqs["true_seq"].seq)


            if after_or_before == "augustus":
                augustus_dict = augustus_stats(train_run_dir)
                run_stats["v_start"] = augustus_dict["augustus_start"]
                run_stats["v_end"] = augustus_dict["augustus_end"]
                run_stats["number_of_augustus_predictions"] = augustus_dict["number_of_augustus_predictions"]

            run_stats["actual_epochs"] = actual_epochs
            run_stats["mbRAM"] = ram_and_time_dict["mbRAM"]
            run_stats["total_time"] = ram_and_time_dict["total_time"]
            run_stats["fitting_time"] = ram_and_time_dict["fitting_time"]
            run_stats["mean_epoch_time"] = ram_and_time_dict["mean_epoch_time"]
            run_stats["best_loss"] = history_stats_dict["best_loss"]
            run_stats["A_prior"] = history_stats_dict["A_prior"]
            run_stats["B_prior"] = history_stats_dict["B_prior"]
            run_stats["sum_of_seq_lens"] = sum_of_seq_lens_dict["sum_of_lens"]
            run_stats["max_of_seq_lens"] = sum_of_seq_lens_dict["max_of_lens"]

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

    print(df.columns)

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

################################################################################

def add_additional_eval_cols(df, args):

    df["size_of_q"] = 1+5+2+3*df["nCodons"] + 3*(df["nCodons"]-1) +5 +2+ 2

    df["p_nt_on_exon"]           = df["end"] -    df["start"]
    df["predicted_p_nt_on_exon"] = df["v_end"] - df["v_start"]


    df["deletes"] = df["nCodons"] - np.ceil(df["predicted_p_nt_on_exon"] / 3) + df["inserts"]


    df["n_nt_on_exon"]           = df["seq_len_from_multi_run"] - df["p_nt_on_exon"]
    df["predicted_n_nt_on_exon"] = df["seq_len_from_multi_run"] - df["predicted_p_nt_on_exon"]

    df["len_ratio"] = df["predicted_p_nt_on_exon"] / df["p_nt_on_exon"]
    df["true_left"] = df["start"] == df["v_start"]
    df["true_right"] = df["end"] == df["v_end"]

    def overlap(row):
        start_overlap = max(row["start"], row["v_start"])
        end_overlap = min(row["end"], row["v_end"])
        return max(0, end_overlap - start_overlap)

    df['tp_nt_on_exon'] = df.apply(lambda row: overlap(row), axis=1)
    df['fp_nt_on_exon'] = df["predicted_p_nt_on_exon"] - df["tp_nt_on_exon"]

    df['fn_nt_on_exon'] = df["p_nt_on_exon"] - df["tp_nt_on_exon"]
    df['tn_nt_on_exon'] = df["seq_len_from_multi_run"] - df["fn_nt_on_exon"] -df["tp_nt_on_exon"]  - df["fp_nt_on_exon"]

    df['sn_nt_on_exon'] = df["tp_nt_on_exon"] / df["p_nt_on_exon"]
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

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["correct"])).reset_index(name = "sum_correct")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["skipped_exon"])).reset_index(name = "sum_skipped_exon")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    # same for sum miss, wrap, incomplete, overlaps
    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["miss"])).reset_index(name = "sum_miss")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["wrap"])).reset_index(name = "sum_wrap")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["incomplete"])).reset_index(name = "sum_incomplete")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["overlaps"])).reset_index(name = "sum_overlaps")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    number_of_exons = df["sum_correct"] + df["sum_skipped_exon"] + df["sum_miss"] + df["sum_wrap"] + df["sum_incomplete"] + df["sum_overlaps"]

    df["f1"] = 2*df["sum_correct"] / (2*(number_of_exons)- df["sum_skipped_exon"])


    change_type_to_int = ["tp_nt", "tn_nt", "v_start", "v_end", "start", "end", "actual_epochs","tp_nt_on_exon", "fp_nt_on_exon", "fn_nt_on_exon", "tn_nt_on_exon"]
    for col in change_type_to_int:
        df[col] = df[col].astype(int)

    return df

################################################################################

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

################################################################################

def eval_viterbi(args):
    path = args.eval_viterbi
    df = load_or_calc_eval_df(path)
    loaded_cols = df.columns


    df = add_additional_eval_cols(df, args)
    df = sort_columns(df)
    added_cols = list(set(df.columns) - set(loaded_cols))

    _, parameters_with_less_than_one_arg = get_cols_to_group_by(args)

    cols_to_group_by, _ = get_cols_to_group_by(args)
    eval_cols = ["sn_nt", "sp_nt", "f1_nt", "MCC", "ACP", "correct", "true_left", "true_right", "incomplete", "wrap",  "overlaps", "miss", "skipped_exon", "5prime", "3prime", "WEScore", "f1"]
    grouped = df.groupby(cols_to_group_by).mean()[eval_cols].reset_index().sort_values("f1_nt")

    # add a new column right intron len
    df["right_intron_len"] = df["seq_len_from_multi_run"] - df["end"]
    # add a new column left intron len
    df["left_intron_len"] = df["start"]


    g1 = df.groupby(cols_to_group_by).mean()[eval_cols].reset_index()
    print('g1[(g1["priorA"] == 0) & (g1["epochs"] == 20) & (g1["exon_skip_init_weight"] == -4)]')

    return df, grouped, parameters_with_less_than_one_arg, eval_cols, loaded_cols, added_cols, cols_to_group_by

################################################################################

def anova_for_one_hyper(df, group, hyper_grid_para, predicted_column = "f1_nt_on_exon"):

    print(f"\n\nANOVA for {group}\n\n")
    from scipy import stats
    missing_one_hyper_para_col = list(set(hyper_grid_para) - set([group]))
    grouped_data = df.groupby(missing_one_hyper_para_col)

    for group_name, group_df in grouped_data:
        print("Group:", list(zip(missing_one_hyper_para_col, group_name)))
        group_values = []
        for level_name, level_df in group_df.groupby(group):
            group_values.append(level_df[predicted_column].values)

        gro = df.groupby(missing_one_hyper_para_col + [group]).mean()["f1_nt"].reset_index()
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
    plt.savefig('heatmap.png', bbox_inches='tight')
    plt.close()

    print(f"made heatmap, it took {np.round(time.perf_counter() - start,3)}")

################################################################################

def len_groups(df, nbins = 3, mask = [1,1,1], eval_cols = None, seq_len_bins = 3):

    if eval_cols is None:
        eval_cols = ["f1_nt_on_exon", "f1_nt", "f1", "correct", "miss", "true_left", "true_right", "total_time", "mbRAM", "mean_epoch_time"]

    # df = df[(df["after_or_before"]=="after") & (df["dataset_identifier"]=="all") & (df["exon_skip_init_weight"]==-1)]

    df = df[(df["after_or_before"]=="after")]


    df["exon_len"] = df["end"] - df["start"]
    bins_exon_len = list(range(0, df["exon_len"].max() + 1, int(df["exon_len"].max()/nbins)))
    df['exon_len_group'] = pd.cut(df['exon_len'], bins=bins_exon_len, labels=bins_exon_len[:-1])


    bins_size_of_q = list(range(0, df["size_of_q"].max() + 1, int(df["size_of_q"].max()/nbins)))
    df['size_of_q_group'] = pd.cut(df['size_of_q'], bins=bins_size_of_q, labels=bins_size_of_q[:-1] )

    bins_human_len = list(range(0, df["seq_len_from_multi_run"].max() + 1, int(df["seq_len_from_multi_run"].max()/seq_len_bins)))
    df['human_len_group'] = pd.cut(df['seq_len_from_multi_run'], bins=bins_human_len, labels=bins_human_len[:-1] )

    bins_sum_len = list(range(0, df["sum_of_seq_lens"].max() + 1, int(df["sum_of_seq_lens"].max()/seq_len_bins)))
    df['sum_len_group'] = pd.cut(df['sum_of_seq_lens'], bins=bins_sum_len, labels=bins_sum_len[:-1] )

    bins_max_len = list(range(0, df["max_of_seq_lens"].max() + 1, int(df["max_of_seq_lens"].max()/seq_len_bins)))
    df['max_len_group'] = pd.cut(df['max_of_seq_lens'], bins=bins_max_len, labels=bins_max_len[:-1] )

    print("exon_len_group", df["exon_len_group"])
    print("human_len_group", df["human_len_group"])
    print("sum_len_group", df["sum_len_group"])
    print("max_len_group", df["max_len_group"])
    print("size_of_q_group", df["size_of_q_group"])


    for value, pivot in zip(["mbRAM", "mean_epoch_time"],["max_len_group", "sum_len_group"]):
        groupby = [pivot, "size_of_q_group"]
        len_groups = df.groupby(groupby).mean()[eval_cols].reset_index()
        len_groups_std = df.groupby(groupby).std()[eval_cols].reset_index()
        len_groups_counts = df.groupby(groupby).count()[eval_cols].reset_index()
        m = len_groups.pivot(pivot, "size_of_q_group", value)
        s = len_groups_std.pivot(pivot, "size_of_q_group", value)
        c = len_groups_counts.pivot(pivot, "size_of_q_group", value)

        print(f"pivot table for {value}\\\\")
        for i in range(seq_len_bins):
            print(" & ". join([f"{m.iloc[i,j]:.0f} & $\pm${s.iloc[i,j]:.0f} & \\multicolumn{{1}}{{c|}}{{{c.iloc[i,j]}}}" for j in range(nbins)]), end = " \\\\ \n")
        print()

    groupby = ["human_len_group", "exon_len_group"]
    len_groups = df.groupby(groupby).mean()[eval_cols].reset_index()
    len_groups_counts = df.groupby(groupby).count()[eval_cols].reset_index()
    f1_nt = len_groups.pivot("human_len_group", "exon_len_group", "f1_nt_on_exon")
    f1_exon = len_groups.pivot("human_len_group", "exon_len_group", "correct")
    c = len_groups_counts.pivot("human_len_group", "exon_len_group", "f1_nt_on_exon") # or f1_nt_on_exon, its the same since the two other args to the method are the same


    print(f"pivot table for f1\\\\")
    for i in range(seq_len_bins):
        if i > 2:
            print(" & ". join([f"{f1_nt.iloc[i,j]*100:.1f} & {f1_exon.iloc[i,j]*100:.1f} & \\multicolumn{{1}}{{c|}}{{{c.iloc[i,j]}}}" for j in range(nbins)]), end = " \\\\ \n")
        else:
            print(" & ". join([f"{f1_nt.iloc[i,j]:.3f} & {f1_exon.iloc[i,j]:.3f} & \\multicolumn{{1}}{{c|}}{{{c.iloc[i,j]}}}" for j in range(nbins)]), end = " \\\\ \n")
    print()


    groupby = ["human_len_group", "sum_len_group", "exon_len_group"]
    groupby = [group for i, group in enumerate(groupby) if mask[i]]

    len_groups = df.groupby(groupby).mean()[eval_cols].reset_index()
    len_groups_std = df.groupby(groupby).std()[eval_cols].reset_index()
    len_groups_counts = df.groupby(groupby).count()[eval_cols].reset_index()


    return len_groups, len_groups_std, len_groups_counts

################################################################################

def plot_model_size_factor_vs_best_loss_f1s(df, figsize = (20,7), angle1 = 90, angle2 = 60, eval_cols = None, anchor = (1,0.5)):

    def scale_0_to_1(column):
        return (column - column.min()) / (column.max() - column.min())

    local_df = df.copy()


    # select only the after or before == after
    local_df = local_df[local_df["after_or_before"] == "after"]

    # add a column called py that is best_loss - a prior - b prior
    local_df["py"] = local_df["best_loss"] - local_df["A_prior"] - local_df["B_prior"]

    columns_to_scale = ["best_loss", "f1_nt_on_exon", "A_prior", "B_prior", "py"]
    columns_to_scale = ["f1_nt_on_exon"]

    make_bigger = 0.008
    for column in columns_to_scale:
        local_df[f"{column}_scaled"] = scale_0_to_1(local_df[column])
        local_df[f"{column}_scaled"] = local_df[f"{column}_scaled"] + np.random.choice([-make_bigger, make_bigger], size=len(local_df))

    local_df["inserts"] = local_df["inserts"]
    local_df["deletes"] = local_df["deletes"]

    measure_names = ["inserts", "deletes"]
    for column in measure_names:
        local_df[f"{column}_scaled"] = scale_0_to_1(local_df[column])


    columns_single_points_in_plot = ["f1_nt", "f1"]

    for column in columns_single_points_in_plot:
        local_df[f"{column}_scaled"] = local_df.groupby("model_size_factor")[column].transform("mean")
        local_df[f"{column}_scaled"] = local_df[f"{column}_scaled"] + np.random.choice([-make_bigger, make_bigger], size=len(local_df))

    value_vars = [f"{c}_scaled" for c in columns_to_scale + columns_single_points_in_plot + measure_names]

    for column in value_vars:
        print(column, local_df[column].min(), local_df[column].max())

    df_melt = pd.melt(local_df, id_vars='model_size_factor', value_vars= value_vars)

    plt.figure(figsize=figsize)
    ax = sns.boxplot(x='model_size_factor', y='value', hue='variable', data=df_melt)

    ax.legend(loc='center left', bbox_to_anchor=anchor)
    legend = ax.get_legend()
    new_labels = [r'exon family specific $F_1^{nt}$', r'$F_1^{nt}$', '$F_1^{exon}$',  'number of insertions',  'number of deletions'] # specify new labels here
    for t, l in zip(legend.texts, new_labels): t.set_text(l)

    ax.set_xlabel("Model Size Factor")
    ax.set_ylabel("values scaled to be in [0,1]")
    ax.set_title("The Influence of the Model Size Factor on the Exon Prediction Performance")
    plt.savefig('model_size_factor_vs_best_loss_f1s.png', bbox_inches='tight')
################################################################################
def plot_two_columns_box_plot(columns):
    plt.boxplot(columns)
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Boxplot of Column1 and Column2')
    plt.savefig("2box_plots.png")

################################################################################
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