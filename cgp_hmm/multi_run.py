#!/usr/bin/env python3
from datetime import datetime
import os
import json
import re
from itertools import product
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Config module description')
parser.add_argument('--train', action = 'store_true', help='number of codons')
parser.add_argument('--viterbi_path', help='path to multi_run dir for which to run viterbi')
parser.add_argument('--eval_viterbi', help='path to multi_run dir for which to evaluation of viterbi')
parser.add_argument('--threads_for_viterbi', type = int, default = 1, help='how many threads should be used for viterbi')
args = parser.parse_args()

# do i want to create bench to run on slurm submission which might be able to run both tf.learning and c++ viterbi
# or create 2 modes in bench one for training on apphub and one for running viterbi and evaluating


assert args.train or args.viterbi_path or args.eval_viterbi, "you must pass either --train or --viterbi_path or --eval_viterbi"

cfg_with_args = {}

out_path = "../../cgp_data"

# or read files in a passed dir
fasta_dir_path = "/home/mattes/Documents/cgp_data/good_exons"
exons = ["exon_chr1_1045963_1046088", \
        "exon_chr1_155188007_155188063", \
        "exon_chr1_74369206_74369264"]

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
cfg_with_args["step"] = [2]
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
        # self.parser.add_argument('--viterbi_parent_input_dir', help = 'path to dir containing the config_attr.json and paratemeters dir used for viterbi')
        # self.parser.add_argument('--in_viterbi_path', help = 'if viteribi is already calculated, path to viterbi file which is then written to the alignment')
        # self.parser.add_argument('--viterbi_threads', type = int, default = 1, help = 'how many threads for viterbi.cc')
        # self.parser.add_argument('--path_to_dir_where_most_recent_dir_is_selected', help = 'path_to_dir_where_most_recent_dir_is_selected')

        # fasta path doesnt need to get calculated since it is saved in the cfg in viterbi_parent_input_dir

        force_overwrite = False
        if force_overwrite:
            command = f"./Viterbi.py --only_first_seq --viterbi_parent_input_dir {sub_path} --viterbi_threads {args.threads_for_viterbi} --force_overwrite"
        if not force_overwrite:
            command = f"./Viterbi.py --only_first_seq --viterbi_parent_input_dir {sub_path} --viterbi_threads {args.threads_for_viterbi}"

        print("running", command)
        subprocess.call(re.split("\s+", command))

if args.eval_viterbi:
    '''
    start stop of true vs guess
    len seq,
    len exon
    len exon guess
    overlapp len vs non overlapp len
    '''


    from add_gene_structure_to_alignment import read_true_alignment_with_out_coords_seq
    stats = {}
    for sub_path in get_run_sub_dirs(args.eval_viterbi):
        exon_stats = {}
        json_data = json.load(open(f"{sub_path}/passed_args.json"))
        # the genomic seq, true exon pos, viterbi guess is optional
        true_alignemnt_path = f"{sub_path}/true_alignment.clw"
        print(true_alignemnt_path)
        true_alignemnt = read_true_alignment_with_out_coords_seq(true_alignemnt_path)


        assert len(true_alignemnt) == 3, f"true_alignment of {sub_path} doesnt contain the viterbi guess"
        aligned_seqs = {} # reference_seq, true_seq, viterbi_guess
        for seq in true_alignemnt:
            aligned_seqs[seq.id] = seq
        assert len(aligned_seqs) == 3, "some seqs had same id"

        exon_stats["true_start"] = aligned_seqs["true_seq"].seq.index("E") # inclusive
        exon_stats["true_end"] = aligned_seqs["true_seq"].seq.index("r") # exclusive

        for i in range(len(aligned_seqs["viterbi_guess"].seq)):
            if aligned_seqs["viterbi_guess"].seq[i:i+2] == "AG":
                exon_stats["guessed_start"] = i+2
            if aligned_seqs["viterbi_guess"].seq[i:i+2] == "GT":
                exon_stats["guessed_end"] = i

        stats[json_data["fasta_path"]] = exon_stats


    for exon, exon_stats in stats.items():
        print("exon", exon)
        for key, value in exon_stats.items():
            print(key, value)
        print()

#     exon level genauikgkeit also wie viel vom exon wird getroffen
# also zuerst die lägen anddieren und dann durch gesamt länge teilen aslo große exons sind wichtiger zu treffen

