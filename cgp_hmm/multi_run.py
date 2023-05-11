#!/usr/bin/env python3
from datetime import datetime
import numpy as np
import os
import json
import pandas as pd
import re
from itertools import product
import subprocess
import argparse

from helpers.add_gene_structure_to_alignment import read_true_alignment_with_out_coords_seq

out_path = "../../cgp_data"


def get_multi_run_config():

    parser = argparse.ArgumentParser(description='Config module description')
    parser.add_argument('--train', action = 'store_true', help='train with args specified in the methods in this module and write output to calculated path based on current time')
    parser.add_argument('--continue_training', help='path to multi_run dir for which to continue training')
    parser.add_argument('--viterbi_path', help='path to multi_run dir for which to run viterbi')
    parser.add_argument('--use_init_weights_for_viterbi', action = 'store_true', help = 'use the initial weights instead of the learned ones')
    parser.add_argument('--slurm_viterbi', action = 'store_true', help = 'submit Viterbi.py calls to slurm')
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

def get_cfg_without_args_that_are_switched():

    # cfg_without_args_that_are_switched = '''
    # exon_skip_const
    # '''
    #
    # cfg_without_args_that_are_switched = re.split("\s+", cfg_without_args_that_are_switched)[1:-1]

    pass

def get_cfg_with_args():
    cfg_with_args = {}

    # or read files in a passed dir
    fasta_dir_path = "../../cgp_data/good_exons_1"
    # exons = ["exon_chr1_8364055_8364255", \
    #         "exon_chr1_33625050_33625254"]

    exons = [dir for dir in os.listdir(fasta_dir_path) if not os.path.isfile(os.path.join(fasta_dir_path, dir)) ]

    # TODO i need to determine the nCodons that should be used for each fasta,
    # are there other parameters that defend on the sequences?

    cfg_with_args["fasta"] = [f"{fasta_dir_path}/{exon}/combined.fasta" for exon in exons]
    get_exon_len = lambda exon_string: (int(exon_string.split("_")[-1]) - int(exon_string.split("_")[-2]))
    get_exon_codons = lambda exon_string: get_exon_len(exon_string) // 3


    #  i can also pass the same arg to a parameter twice and bind it with another parameter to achieve
    # something like manual defined grid points
    cfg_with_args["nCodons"] = [get_exon_codons(exon) for exon in exons]
    # cfg_with_args["exon_skip_init_weight"] = [-2,-3,-4]
    # cfg_with_args["learning_rate"] = [0.1, 0.01]
    cfg_with_args["priorA"] = [1,0,3]
    cfg_with_args["priorB"] = [1,0,3]
    cfg_with_args["global_log_epsilon"] = [1e-20]
    cfg_with_args["epoch"] = [20]
    cfg_with_args["batch_size"] = [16]
    cfg_with_args["step"] = [16]
    cfg_with_args["clip_gradient_by_value"] = [5]
    cfg_with_args["prior_path"] = [" ../../cgp_data/priors/human/"]
    cfg_with_args["exon_skip_init_weight"] = [-2, -4, -10]


    return cfg_with_args

def get_bind_args_together(cfg_with_args):
    '''
    cant bind args without parameter
    '''
    bind_args_together = [set([key]) for key in cfg_with_args.keys()]
    bind_args_together += [{"fasta", "nCodons"}]
    # bind_args_together += [{"exon_skip_init_weight", "nCodons"}]
    bind_args_together += [{"priorA", "priorB"}]

    return bind_args_together

def get_cfg_without_args():
    cfg_without_args = '''
    internal_exon_model
    my_initial_guess_for_parameters
    logsumexp
    bucket_by_seq_len
    '''
    cfg_without_args = re.split("\s+", cfg_without_args)[1:-1]
    return cfg_without_args


def get_and_make_dir():
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M")

    multi_run_dir = f"{out_path}/multi_run_{date_string}"
    if not os.path.exists(multi_run_dir):
        os.makedirs(multi_run_dir)
    return multi_run_dir

def write_cfgs_to_file(multi_run_dir):
    cfg_with_args_path = f"{multi_run_dir}/cfg_with_args.json"
    with open(cfg_with_args_path, "w") as out_file:
        json.dump(get_cfg_with_args(), out_file)

    cfg_without_args_path = f"{multi_run_dir}/cfg_without_args.json"
    with open(cfg_without_args_path, "w") as out_file:
        json.dump(get_cfg_without_args(), out_file)

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

def no_overlapp(inp : list[set]) -> bool:
        for i in range(len(inp)):
            for j in range(i+1, len(inp)):
                if len(inp[i] & inp[j]) > 0:
                    return False
        return True

def merge(inp : list[set[str]]) -> list[set[str]]:
    while not no_overlapp(inp):
        merge_one_step(inp)

def zip_args(inp : list[set]) -> list[tuple[set, zip]]:
    zipped_args = []
    for merged_args in inp:
        zipped_args.append((merged_args,(zip(*[get_cfg_with_args()[key] for key in merged_args]))))
    return zipped_args

def get_grid_points(zipped_args : list[tuple[set, zip]]) -> list[list[list]]:
    '''return list of grid points.
    a gridpoint is a list of parameters.
    a single parameter is wrapped in a list.
    binded parameters are in the same list'''
    return list(product(*[arg[-1] for arg in zipped_args]))


def run_training_without_viterbi(args):

    binded_arg_names = get_bind_args_together(get_cfg_with_args())
    multi_run_dir = get_and_make_dir()
    write_cfgs_to_file(multi_run_dir)

    # [{'global_log_epsilon'}, {'epoch'}, {'step'}, {'clip_gradient_by_value'}, {'prior_path'}, {'exon_skip_init_weight'}, {'nCodons', 'fasta'}, {'priorB', 'priorA'}]
    merge(binded_arg_names)
    zipped_args = zip_args(binded_arg_names)

    arg_names = [single_arg for arg in zipped_args for single_arg in arg[0]]

    with open(f"{multi_run_dir}/arg_names.json", "w") as arg_names_file:
        json.dump(arg_names, arg_names_file)

    grid_points = get_grid_points(zipped_args)

    print("len get_grid_points", len(grid_points))

    print("do you want to continue enter [y/n]")
    while (x :=input()) not in "yn":
        print("input was no y or n")
    if x == "n":
        print("exiting")
        exit()

    with open(f"{multi_run_dir}/grid_points.json", "w") as grid_point_file:
        json.dump(grid_points, grid_point_file)

    with open(f"{multi_run_dir}/todo_grid_points.json", "w") as grid_point_file:
        json.dump(grid_points, grid_point_file)

    args.continue_training = multi_run_dir
    continue_training(args)

def continue_training(parsed_args):

    with open(f"{parsed_args.continue_training}/todo_grid_points.json", "r") as grid_point_file:
        grid_points = json.load(grid_point_file)

    with open(f"{parsed_args.continue_training}/arg_names.json", "r") as arg_names_file:
        # ['global_log_epsilon', 'epoch', 'step', 'clip_gradient_by_value', 'prior_path', 'exon_skip_init_weight', 'fasta', 'nCodons', 'priorB', 'priorA']
        arg_names = json.load(arg_names_file)

    for i, point_in_grid in enumerate(grid_points):
        print(f"calculating point in hyperparameter grid {i}/{len(grid_points)}")
        # [1e-20, 20, 8, 5, ' ../../cgp_data/priors/human/', -10, '../../cgp_data/good_exons_1/exon_chr1_1050426_1050591/combined.fasta', 55, 100, 100]
        args = [single for p in point_in_grid for single in p]
        pass_args = " ".join([f"--{arg_name} {arg}" for arg_name, arg in zip(arg_names, args)])
        pass_args += " " + " ".join([f"--{arg}" for arg in get_cfg_without_args()])

        from Config import get_dir_path_from_fasta_nCodons_and_out_path

        nCodons = args[arg_names.index("nCodons")]
        fasta_path = args[arg_names.index("fasta")]
        run_dir = get_dir_path_from_fasta_nCodons_and_out_path(parsed_args.continue_training, nCodons, fasta_path)

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        err_path = f"{run_dir}/err.log"
        out_path = f"{run_dir}/out.log"

        command = f"./main_programm.py {pass_args} --passed_current_run_dir {run_dir}"
        print("running", command)

        path_to_command_file = f"{run_dir}/called_command.log"
        with open(path_to_command_file, "w") as out_file:
            out_file.write(command)
            out_file.write("\n")

        # running command and directing out steams
        with open(out_path, "w") as out_handle:
            with open(err_path, "w") as err_handle:
                exit_code = subprocess.call(re.split("\s+", command), stderr = err_handle, stdout = out_handle)
                if exit_code != 0:
                    print("exit_code:", exit_code)
                    exit(1)

        with open(f"{parsed_args.continue_training}/todo_grid_points.json", "w") as grid_point_file:
            json.dump(grid_points[i+1:], grid_point_file)

        # TODO maybe if i ctrl C this the learning isnt findished, and the gridpoint is removed from todo list
        # bc i want to do viterbi afterwards, i should just look if the after fit parameters are there, and if so
        # then remove the gridpoint


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

def viterbi(path):
    for sub_path in get_run_sub_dirs(path):

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

        submission_file_name = f"{sub_path}/slurm_submission.sh"

        slurm_out_path = f"{sub_path}/slurm_out/"
        if not os.path.exists(slurm_out_path):
            os.makedirs(slurm_out_path)

        if args.slurm_viterbi:
            with open(submission_file_name, "w") as file:
                file.write("#!/bin/bash\n")
                file.write(f"#SBATCH -J viterbi_{os.path.basename(sub_path)}\n")
                file.write(f"#SBATCH -N 1\n")
                file.write(f"#SBATCH -n 1\n")
                file.write("#SBATCH --mem 2000\n")
                file.write("#SBATCH --partition=snowball\n")
                file.write(f"#SBATCH -o {sub_path}/slurm_out/out.%j\n")
                file.write(f"#SBATCH -e {sub_path}/slurm_out/err.%j\n")
                file.write("#SBATCH -t 02:00:00\n")
                file.write(command)
                file.write("\n")

            os.system(f"sbatch {submission_file_name}")
        else:
            print("running", command)
            subprocess.call(re.split("\s+", command))

def get_true_alignemnt_path(train_run_dir, after_or_before):
    return f"{train_run_dir}/true_alignment_{after_or_before}.clw"

def get_viterbi_aligned_seqs(train_run_dir, after_or_before):
    true_alignemnt_path = get_true_alignemnt_path(train_run_dir, after_or_before)
    try:
        true_alignemnt = read_true_alignment_with_out_coords_seq(true_alignemnt_path)
    except:
        print(f"couldnt read_true_alignment_with_out_coords_seq({true_alignemnt_path})")
        return -1

    assert len(true_alignemnt) == 3, f"{true_alignemnt_path} doesnt contain the viterbi guess"
    aligned_seqs = {} # reference_seq, true_seq, viterbi_guess
    for seq in true_alignemnt:
        aligned_seqs[seq.id] = seq
    assert len(aligned_seqs) == 3, "some seqs had same id"

    return aligned_seqs

def get_run_stats(path) -> pd.DataFrame:

    stats = {}
    for train_run_dir, after_or_before in product(get_run_sub_dirs(path), ["after", "before"]):
        run_stats = {}
        if (aligned_seqs := get_viterbi_aligned_seqs(train_run_dir, after_or_before)) == -1:
            continue

        add_true_and_guessed_exons_coords_to_run_stats(run_stats, aligned_seqs, get_true_alignemnt_path(train_run_dir, after_or_before))
        run_stats["after_or_before"] = after_or_before

        training_args = json.load(open(f"{train_run_dir}/passed_args.json"))
        add_actual_epochs_to_run_stats(train_run_dir, run_stats, max_epochs = training_args['epochs'])
        stats[(train_run_dir, after_or_before)] = run_stats, training_args

    df = pd.DataFrame(columns=list(set(training_args.keys()) or set(run_stats.keys())))

    for run_stats, training_args in stats.values():
        new_row = pd.DataFrame({**training_args, **run_stats}, index=[0])
        df = pd.concat([df, new_row], axis = 0, ignore_index = True)


    # remove cols, ie name of parameter for training runs, whos args are const across runs
    cols_to_keep = df.columns[df.nunique() > 1]
    # i want to keep run stats even if those results are const across runs
    cols_to_keep = list(set(list(cols_to_keep) + list(run_stats.keys())))
    df = df[cols_to_keep]

    df["fasta"] = df["fasta_path"].apply(os.path.dirname).apply(os.path.basename)
    df["passed_current_run_dir"] = df["passed_current_run_dir"].apply(os.path.basename)

    return df

def add_actual_epochs_to_run_stats(sub_path, run_stats, max_epochs = None):
    # Epoch 6/20
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

def add_true_and_guessed_exons_coords_to_run_stats(run_stats, aligned_seqs, true_alignemnt_path):
    run_stats["true_start"] = aligned_seqs["true_seq"].seq.index("E") # inclusive
    run_stats["true_end"] = aligned_seqs["true_seq"].seq.index("r") # exclusive

    for i in range(len(aligned_seqs["viterbi_guess"].seq)):
        if aligned_seqs["viterbi_guess"].seq[i:i+2] == "AG":
            run_stats["guessed_start"] = i+2
        if aligned_seqs["viterbi_guess"].seq[i:i+2] == "GT":
            run_stats["guessed_end"] = i
    if "guessed_start" not in run_stats:
        run_stats["guessed_start"] = -1
    if "guessed_end" not in run_stats:
        run_stats["guessed_end"] = -1

    if run_stats["guessed_start"] == -1 or run_stats["guessed_end"] == -1:
        assert run_stats["guessed_end"] + run_stats["guessed_start"] == -2, f"if viterbi didnt find start it is assumend that it also didnt find end, bc i dont want to handle this case, true_alignemnt_path = {true_alignemnt_path}"


def get_cols_to_group_by():
    parameters_with_more_than_one_arg = [name for name, args in get_cfg_with_args().items() if len(args) > 1]
    parameters_with_more_than_one_arg.remove("fasta")
    parameters_with_more_than_one_arg.remove("nCodons")

    parameters_with_more_than_one_arg += ["after_or_before"]

    return parameters_with_more_than_one_arg

def sort_columns(df):
    sorted_columns = ["passed_current_run_dir", "actual_epochs", "true_start", "true_end", "guessed_start", "guessed_end", "exon_len", "overlap", "overlap_single_ratio", "overlap_ratio_per_grid_point", "after_or_before", "priorA", "priorB", "exon_skip_init_weight"]
    remaining_columns = list(set(df.columns) - set(sorted_columns))
    df = df[sorted_columns + remaining_columns]
    return df

def add_additional_eval_cols(df):
    df["exon_len"] = df["true_end"] - df["true_start"]

    def overlap(row):
        start_overlap = max(row['true_start'], row['guessed_start'])
        end_overlap = min(row['true_end'], row['guessed_end'])
        return max(0, end_overlap - start_overlap)
    df['overlap'] = df.apply(lambda row: overlap(row), axis=1)
    df['overlap_single_ratio'] = df["overlap"] / df["exon_len"]

    cols_to_group_by = get_cols_to_group_by()
    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["exon_len"])).reset_index(name = "sum_exon_lens")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    new_col = df.groupby(cols_to_group_by).apply(lambda x: sum(x["overlap"])).reset_index(name = "sum_overlap_lens")
    df = pd.merge(df, new_col, on = cols_to_group_by, how = "left")

    df["overlap_ratio_per_grid_point"] = df["sum_overlap_lens"] / df["sum_exon_lens"]
    return df

def eval_viterbi(path):
    df = load_or_calc_eval_df(path)

    # print(df.groupby(["priorA", "priorB", "exon_skip_init_weight", "fasta"]).apply(np.std))
    # print(df.groupby(["priorA", "priorB", "exon_skip_init_weight", "fasta"]).size())

    df = add_additional_eval_cols(df)
    df = sort_columns(df)

    cols_to_group_by = get_cols_to_group_by()
    grouped = df.groupby(cols_to_group_by).apply(lambda x: sum(x["overlap"]/sum(x["exon_len"]))).reset_index(name = "grouped_overlap_ratio_per_grid_point").sort_values("grouped_overlap_ratio_per_grid_point")

    return df, grouped

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
            df = get_run_stats(path)
            df.to_csv(path_to_out_csv, header = True)
    else:
        df = get_run_stats(path)
        df.to_csv(path_to_out_csv, header = True)
    return df

# pd.set_option('display.max_columns', None)
# pd.options.display.width = 0
# pd.set_option("display.max_rows", None)
# df.sort_values(by = "exon_len", ascending = 1)


def toggle_col():
    pass

def toggle_row():
    pass

if __name__ == "__main__":

    args = get_multi_run_config()

    if args.train:
        run_training_without_viterbi(args)
    if args.viterbi_path:
        viterbi(args.viterbi_path)
    if args.eval_viterbi:
        df, grouped = eval_viterbi(args.eval_viterbi)
    if args.continue_training:
        continue_training(args)