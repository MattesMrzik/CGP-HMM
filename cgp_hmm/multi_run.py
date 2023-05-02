#!/usr/bin/env python3
from datetime import datetime
import os
import json
import pandas as pd
import re
from itertools import product
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Config module description')
parser.add_argument('--train', action = 'store_true', help='number of codons')
parser.add_argument('--viterbi_path', help='path to multi_run dir for which to run viterbi')
parser.add_argument('--use_init_weights_for_viterbi', action = 'store_true', help = 'use the initial weights instead of the learned ones')
parser.add_argument('--slurm_viterbi', action = 'store_true', help = 'submit Viterbi.py calls to slurm')
parser.add_argument('--eval_viterbi', help='path to multi_run dir for which to evaluation of viterbi')
parser.add_argument('--threads_for_viterbi', type = int, default = 1, help='how many threads should be used for viterbi')
args = parser.parse_args()

if args.use_init_weights_for_viterbi:
    assert args.viterbi_path, "if you pass --use_init_weights_for_viterbi. you must also pass --viterbi_path"

# do i want to create bench to run on slurm submission which might be able to run both tf.learning and c++ viterbi
# or create 2 modes in bench one for training on apphub and one for running viterbi and evaluating


assert args.train or args.viterbi_path or args.eval_viterbi, "you must pass either --train or --viterbi_path or --eval_viterbi"

cfg_with_args = {}

out_path = "../../cgp_data"

# or read files in a passed dir
fasta_dir_path = "../../cgp_data/good_exons_1"
exons = ["exon_chr1_8364055_8364255", \
        "exon_chr1_33625050_33625254"]

exons = [dir for dir in os.listdir(fasta_dir_path) if not os.path.isfile(os.path.join(fasta_dir_path, dir)) ]

# TODO i need to determine the nCodons that should be used for each fasta,
# are there other parameters that defend on the sequences?

cfg_with_args["fasta"] = [f"{fasta_dir_path}/{exon}/combined.fasta" for exon in exons]
get_exon_len = lambda exon_string: (int(exon_string.split("_")[-1]) - int(exon_string.split("_")[-2]))
get_exon_codons = lambda exon_string: get_exon_len(exon_string) // 3

cfg_with_args["nCodons"] = [get_exon_codons(exon) for exon in exons]
# cfg_with_args["exon_skip_init_weight"] = [-2,-3,-4]
# cfg_with_args["learning_rate"] = [0.1, 0.01]
cfg_with_args["priorA"] = [2]
cfg_with_args["priorB"] = [2]
cfg_with_args["global_log_epsilon"] = [1e-20]
cfg_with_args["epoch"] = [1]
cfg_with_args["step"] = [8]
cfg_with_args["clip_gradient_by_value"] = [5]
cfg_with_args["prior_path"] = [" ../../cgp_data/priors/human/"]
cfg_with_args["scale_prior"] = [1e-3]

bind_args_together = [set([key]) for key in cfg_with_args.keys()]
bind_args_together += [{"fasta", "nCodons"}]
# bind_args_together += [{"exon_skip_init_weight", "nCodons"}]
bind_args_together += [{"priorA", "priorB"}]


cfg_without_args = '''
internal
my_initial_guess_for_parameters
logsumexp
'''
cfg_without_args = re.split("\s+", cfg_without_args)[1:-1]


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
        json.dump(cfg_with_args, out_file)

    cfg_without_args_path = f"{multi_run_dir}/cfg_without_args.json"
    with open(cfg_without_args_path, "w") as out_file:
        json.dump(cfg_without_args, out_file)

def run_training_without_viterbi(bind_args_together):
    def merge(inp):
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

    def no_overlapp(inp):
        for i in range(len(inp)):
            for j in range(i+1, len(inp)):
                if len(inp[i] & inp[j]) > 0:
                    return False
        return True

    while not no_overlapp(bind_args_together):
        merge(bind_args_together)

    zipped_args = []
    for merged_args in bind_args_together:
        zipped_args.append((merged_args,(zip(*[cfg_with_args[key] for key in merged_args]))))

    for point_in_grid in product(*[arg[-1] for arg in zipped_args]):
        arg_names = [single_arg for arg in zipped_args for single_arg in arg[0]]
        args = [single for p in point_in_grid for single in p]
        pass_args = " ".join([f"--{arg_name} {arg}" for arg_name, arg in zip(arg_names, args)])
        pass_args +=  " " + " ".join([f"--{arg}" for arg in cfg_without_args])

        command = f"./main_programm.py {pass_args} --out_path {multi_run_dir}"
        print("running", command)

        subprocess.call(re.split("\s+", command))
        # status = os.system(command)
        # exit_status = os.WEXITSTATUS(status)
        # if exit_status != 0:
        #     print("exit_status:", exit_status)
        #     exit(1)


if args.train:
    multi_run_dir = get_and_make_dir()
    write_cfgs_to_file(multi_run_dir)
    run_training_without_viterbi(bind_args_together)

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

if args.viterbi_path:
    for sub_path in get_run_sub_dirs(args.viterbi_path):
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
                   --viterbi_threads {args.threads_for_viterbi} \
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
                file.write(f"#SBATCH -n {args.threads_for_viterbi}\n")
                file.write("#SBATCH --mem 2000\n")
                file.write("#SBATCH --partition=snowball\n")
                file.write(f"#SBATCH -o {sub_path}/slurm_out/out.%j\n")
                file.write(f"#SBATCH -e {sub_path}/slurm_out/err.%j\n")
                file.write("#SBATCH -t 02:00:00\n")
                file.write(command)

            os.system(f"sbatch {submission_file_name}")
        else:
            print("running", command)
            subprocess.call(re.split("\s+", command))

def load_run_stats() -> pd.DataFrame:
    from helpers.add_gene_structure_to_alignment import read_true_alignment_with_out_coords_seq
    stats = {}
    for sub_path in get_run_sub_dirs(args.eval_viterbi):
        run_stats = {}
        # the genomic seq, true exon pos, viterbi guess is optional
        true_alignemnt_path = f"{sub_path}/true_alignment.clw"
        true_alignemnt = read_true_alignment_with_out_coords_seq(true_alignemnt_path)

        assert len(true_alignemnt) == 3, f"true_alignment of {sub_path} doesnt contain the viterbi guess"
        aligned_seqs = {} # reference_seq, true_seq, viterbi_guess
        for seq in true_alignemnt:
            aligned_seqs[seq.id] = seq
        assert len(aligned_seqs) == 3, "some seqs had same id"

        run_stats["true_start"] = aligned_seqs["true_seq"].seq.index("E") # inclusive
        run_stats["true_end"] = aligned_seqs["true_seq"].seq.index("r") # exclusive

        for i in range(len(aligned_seqs["viterbi_guess"].seq)):
            if aligned_seqs["viterbi_guess"].seq[i:i+2] == "AG":
                run_stats["guessed_start"] = i+2
            if aligned_seqs["viterbi_guess"].seq[i:i+2] == "GT":
                run_stats["guessed_end"] = i

        json_data = json.load(open(f"{sub_path}/passed_args.json"))
        json_tuple = tuple(json_data.items())
        stats[json_tuple] = run_stats, json_data


    eval_cols = ["guessed_start", "guessed_end", "true_start", "true_end"]
    df = pd.DataFrame(columns=list(json_data.keys()) + eval_cols)

    for key, value in stats.items():
        value[1].update(value[0])
        new_row = pd.DataFrame(value[1], index=[0])
        df = pd.concat([df, new_row], axis = 0, ignore_index = True)

    cols_to_keep = df.columns[df.nunique() > 1]
    cols_to_keep = list(set(list(cols_to_keep) + eval_cols))
    df = df[cols_to_keep]

    return df

if args.eval_viterbi:
    '''
    start stop of true vs guess
    len seq,
    len exon
    len exon guess
    overlapp len vs non overlapp len
    '''

    df = load_run_stats()
    print(df)


# exon level genauikgkeit also wie viel vom exon wird getroffen
# also zuerst die lägen anddieren und dann durch gesamt länge teilen aslo große exons sind wichtiger zu treffen

