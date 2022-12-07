#!/usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser(
    description=' # 0:A;B sparse, 1:A dense, 2:B dense, 3:A;B dense, 4:fullmodel')
parser.add_argument('-r', '--range_codon', nargs='+', help='usage: < -r 1 10 > to run 1 2 3 4 5 6 7 8 9 10 codons')
parser.add_argument('-t', '--types', nargs="+", help='types of cell.call() that should be used')
parser.add_argument('-c', '--nCodons', nargs="+", help ='usage: < -c 10 20 50 > to run 10 20 50 codons')
parser.add_argument('--repeat', type = int, default = 1, help ='repeat everthing [r] times')
parser.add_argument('--exit_on_nan', action='store_true', help ="exit_on_nan")
parser.add_argument('--dont_generate_new_seqs', action='store_true', help ="dont_generate_new_seqs")
parser.add_argument('--use_simple_seq_gen', action='store_true', help ="use_simple_seq_gen and not MSAgen")



args = parser.parse_args()

codons = []
if args.range_codon:
    if len(args.range_codon) == 2:
        codons += list(range(int(args.range_codon[0]), int(args.range_codon[1])+1))
    elif len(args.range_codon) == 3:
        codons += list(range(int(args.range_codon[0]), int(args.range_codon[1])+1), int(args.range_codon[2]))
    else:
        print("pass 2 integers a,b > 0 with a < b, or 3 integers a,b,c > 0 with a < b and c stepsize")
        exit(1)
if args.nCodons:
    for c in args.nCodons:
        codons.append(int(c))
codons = sorted(list(set(codons)))
if len(codons) == 0:
    print("pls specify nCodons")
    exit(1)

types = []
if args.types:
    for t in args.types:
        types.append(int(t))
types = sorted(list(set(types)))
if len(types) == 0:
    print("pls specify types")
    exit(1)

from datetime import datetime

for c in codons:
    for t in types:
        for _ in range(args.repeat):
            if os.path.exists("stop"):
                os.system("rm stop")
                exit()
            with open ("small_bench_run_log.txt", "a") as file:
                command = f"./main_programm.py -c {c} -t {t} --opti SGD --batch_begin_exit_when_nan_and_write_weights__layer_call_write_input --epochs 1 --steps 4 {'--dont_generate_new_seqs' if args.dont_generate_new_seqs else ''} {'--use_simple_seq_gen' if args.use_simple_seq_gen else ''}"
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

                if args.exit_on_nan and status != 0:
                    exit()


    # for nCodons in range(11,26):
    #     for type in [2,4]:
    #         # run(f"/usr/bin/time --verbose --output bench/{nCodons}.bench.txt ./main_programm.py -c {nCodons} " )
    #         os.system(f"./main_programm.py -c {nCodons} -t {type}")
