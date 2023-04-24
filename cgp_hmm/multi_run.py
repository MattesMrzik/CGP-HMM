#!/usr/bin/env python3
from datetime import datetime
import os
import json
import re
from itertools import product

cfg_with_args = {}

out_path = "../../cgp_data"

# or read files in a passed dir
fasta_dir_path = "/home/mattes/Documents/cgp_data/out_3Exons_primates.lst_20_15_20_50_15_20_15_20_241-mammalian-2020v2.hal"
exons = ["exon_chr1_67095234_67095421", \
         "exon_chr1_67096251_67096321", \
         "exon_chr1_67103237_67103382"]

# TODO i need to determine the nCodons that should be used for each fasta,
# are there other parameters that defend on the sequences?

cfg_with_args["fasta"] = [f"{fasta_dir_path}/{exon}/combinded.fasta" for exon in exons]
cfg_with_args["nCodons"] = [5,6,7]
cfg_with_args["exon_skip_init_weight"] = [-2,-3,-4]
cfg_with_args["priorA"] = [1,3]
cfg_with_args["priorB"] = [1,3]
cfg_with_args["learning_rate"] = [0.1, 0.01]

bind_args_together = [set([key]) for key in cfg_with_args.keys()]
bind_args_together += [{"fasta", "nCodons"}]
bind_args_together += [{"exon_skip_init_weight", "nCodons"}]
bind_args_together += [{"priorA", "priorB"}]


cfg_without_args = '''
internal
'''
cfg_without_args = re.split("\s+", cfg_without_args)


def get_and_make_dir():
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M")

    multi_run_dir = f"{out_path}/multi_run_{date_string}"
    if not os.path.exists(multi_run_dir):
        os.makedirs(multi_run_dir)
    return multi_run_dir

def write_cfgs_to_file(multi_run_dir):
    cfg_with_args_path = f"{multi_run_dir}/cfg_with_args.cfg"
    with open(cfg_with_args_path, "w") as out_file:
        json.dump(cfg_with_args, out_file)

    cfg_without_args_path = f"{multi_run_dir}/cfg_without_args.cfg"
    with open(cfg_without_args_path, "w") as out_file:
        json.dump(cfg_without_args, out_file)


multi_run_dir = get_and_make_dir()
write_cfgs_to_file(multi_run_dir)

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
        print(pass_args)

run_training_without_viterbi(bind_args_together)


