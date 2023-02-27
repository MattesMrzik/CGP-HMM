#!/usr/bin/env python3

import argparse
import os
from Config import Config
import re

config = Config("small_bench")


check_AB = False

if check_AB:
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

check_batch_size = False
if check_seq_len:
    import Utility
    import matplotlib.pyplot as plt
    y_times = {}
    y_ram = {}
    seq_lens = [(config.nCodons * 4 + 8) * factor for factor in range(1,10)]
    seq_lens += [5000,10000]
    for seq_len in seq_lens:
        config.seq_len = seq_len
        command = f"./main_programm.py {config.get_args_as_str('main_programm')}"
        print("running:", command)
        status = os.system(command)
        if status != 0:
            print("status != 0")
            exit()

        file_path = f"bench/{config.nCodons}codons/{config.AB}_call_type.log"
        y_time, max_ram_in_gb = Utility.get_time_and_ram_from_bench_file(file_path)
        y_times[seq_len] = y_time
        y_ram[seq_len] = max_ram_in_gb
    fig = plt.figure(figsize=(12, 12))
    time_axis = fig.add_subplot(111)


    y_times = [y_times[seq_len] for seq_len in seq_lens]
    y_ram   = [y_ram[seq_len]   for seq_len in seq_lens]

    color = 'tab:red'
    time_axis.set_xlabel('seq_len')
    time_axis.set_ylabel('time in sec', color=color)
    time_axis.plot(seq_lens, y_times, "rx")

    print("times", y_times)
    print("ram", y_ram)

    ram_axis = time_axis.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ram_axis.set_ylabel('ram_peak in gb', color=color)  # we already handled the x-label with ax1
    ram_axis.plot(seq_lens, y_ram, "bo")

    title = f"time and ram of different seq_len, type = {config.AB}, nCodons = {config.nCodons}"
    time_axis.title.set_text(title)
    fig.tight_layout()


    plt.savefig("bench_seq_len.png")
    show_diagramm = True
    if show_diagramm:
        plt.show()

check_batch_size = True
if check_batch_size:
    import Utility
    import matplotlib.pyplot as plt
    y_times = {}
    y_ram = {}
    batch_sizes = [i for i in range(1,65,4)]
    for batch_size in batch_sizes:
        config.batch_size = batch_size
        command = f"./main_programm.py {config.get_args_as_str('main_programm')}"
        print("running:", command)
        status = os.system(command)
        if status != 0:
            print("status != 0")
            exit()

        file_path = f"bench/{config.nCodons}codons/{config.AB}_call_type.log"
        y_time, max_ram_in_gb = Utility.get_time_and_ram_from_bench_file(file_path)
        y_times[batch_size] = y_time
        y_ram[batch_size] = max_ram_in_gb
    fig = plt.figure(figsize=(12, 12))
    time_axis = fig.add_subplot(111)


    y_times = [y_times[batch_size] for batch_size in batch_sizes]
    y_ram   = [y_ram[batch_size]   for batch_size in batch_sizes]

    color = 'tab:red'
    time_axis.set_xlabel('batch_size')
    time_axis.set_ylabel('time in sec', color=color)
    time_axis.plot(batch_sizes, y_times, "rx")

    print("times", y_times)
    print("ram", y_ram)

    ram_axis = time_axis.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ram_axis.set_ylabel('ram_peak in gb', color=color)  # we already handled the x-label with ax1
    ram_axis.plot(batch_sizes, y_ram, "bo")

    title = f"time and ram of different batch_size, type = {config.AB}, nCodons = {config.nCodons}"
    time_axis.title.set_text(title)
    fig.tight_layout()


    plt.savefig("bench_batch_size.png")
    show_diagramm = True
    if show_diagramm:
        plt.show()
