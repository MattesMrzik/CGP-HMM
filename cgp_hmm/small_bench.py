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

types = []
if config.typesList:
    for t in config.typesList:
        types.append(int(t))
types = sorted(list(set(types)))
if len(types) == 0:
    print("pls specify types")
    exit(1)


if config.alpha_i_gradient_list:
    alpha_i_gradient_list = sorted(config.alpha_i_gradient_list)
    exit(1) # bc  alpha_i_gradient_list = [0]  needs to be changed
if not config.manual_traning_loop:
    print("--manual_traning_loop was not passed, so --alpha_i_gradient_list is not used")
    alpha_i_gradient_list = [0] 

from datetime import datetime

for c in codons:
    config.nCodons = c
    for t in types:
        for i in alpha_i_gradient_list:
            config.call_type = t
            print("config.call_type =",  config.call_type)
            for _ in range(config.repeat):
                if os.path.exists("stop"):
                    os.system("rm stop")
                    exit()
                with open ("small_bench_run_log.txt", "a") as file:
                    command = f"./main_programm.py {config.get_args_as_str('main_programm')}"
                    if len(alpha_i_gradient_list) > 0:
                        command += f" --alpha {i}"
                    print("running:", command)
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



    # for nCodons in range(11,26):
    #     for type in [2,4]:
    #         # run(f"/usr/bin/time --verbose --output bench/{nCodons}.bench.txt ./main_programm.py -c {nCodons} " )
    #         os.system(f"./main_programm.py -c {nCodons} -t {type}")
