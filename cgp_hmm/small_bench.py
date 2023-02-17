#!/usr/bin/env python3

import argparse
import os
from Config import Config
import re

config = Config("small_bench")

codons = []
if config.range_codon:
    if len(config.range_codon) == 2:
        codons += list(range(int(config.range_codon[0]), int(config.range_codon[1])+1))
    elif len(config.range_codon) == 3:
        codons += list(range(int(config.range_codon[0]), int(config.range_codon[1])+1), int(config.range_codon[2]))
    else:
        print("pass 2 integers a,b > 0 with a < b, or 3 integers a,b,c > 0 with a < b and c stepsize")
        exit(1)
if config.nCodonsList:
    for c in config.nCodonsList:
        codons.append(int(c))
codons = sorted(list(set(codons)))
if len(codons) == 0:
    print("pls specify nCodons")
    exit(1)

from datetime import datetime

types = ["dd", "ss", "sd"]

# TODO: # write bench and iterating over seqlen for seq_len in [(c * 3 + 8) * (1 + factor/5) for factor in range(1,25)]:

for c in codons:
    config.nCodons = c
    for type in types:
        config.AB = type
        config.A_is_dense = type[0] == "d"
        config.A_is_sparse = not config.A_is_dense
        config.B_is_dense = type[1] == "d"
        config.B_is_sparse = not config.B_is_dense
        for _ in range(config.repeat):
            if os.path.exists("stop"):
                os.system("rm stop")
                exit()
            with open ("small_bench_run_log.txt", "a") as file:
                command = f"./main_programm.py {config.get_args_as_str('main_programm')}"
                print("running:", command)

                # write call and exit status to file
                status = os.system(command)
                status = os.WEXITSTATUS(status)
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                file.write(dt_string)
                file.write("\n")
                file.write(command)
                file.write("\n")
                file.write("exit status " + str(status))
                file.write("\n")

                if config.exit_on_nan and status != 0:
                    exit()

import Utility

Utility.plot_time_and_ram(codons, types)
