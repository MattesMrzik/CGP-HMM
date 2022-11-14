#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(
    description='description')
parser.add_argument('-c', '--nCodons',
                    help='number of codons')
parser.add_argument('-t', '--type',
                    help='type of cell.call():  0:A;B sparse, 1:A dense, 2:B dense, 3:A;B dense, 4:fullmodel')
parser.add_argument('-p', '--path',
                    help='path to src')
parser.add_argument('-b', action='store_true', help ="exit after first batch, you may use this when verbose is True in cell.call()")
parser.add_argument('-n', action='store_true', help ="exit_after_loglik_is_nan, you may use this when verbose is True in cell.call()")
parser.add_argument('-v', '--verbose', nargs = "?", const = "2", help ="verbose E,R, alpha, A, B to file, pass 1 for shapes, 2 for shapes and values")
parser.add_argument('-s', '--verbose_to_stdout', action='store_true', help ="verbose to stdout instead of to file")

args = parser.parse_args()

import matplotlib.pyplot as plt
import tensorflow as tf
from Training import fit_model
from Training import make_dataset
import Utility
import numpy as np
from Bio import SeqIO
from Utility import run
import WriteData

from Utility import remove_old_bench_files
from Utility import remove_old_verbose_files

from CgpHmmCell import CgpHmmCell

config = {}

config["nCodons"] = int(args.nCodons) if args.nCodons else 1
config["order"] = 2
config["order_transformed_input"] = True
config["call_type"] = int(args.type) if args.type else 3 # 0:A;B sparse, 1:A dense, 2:B dense, 3:A;B dense, 4:fullmodel

config["alphabet_size"] = 4
config["src_path"] = "." if not args.path else args.path
config["fasta_path"] = f"{config['src_path']}/output/{config['nCodons']}codons/out.seqs.{config['nCodons']}codons.fa"
config["bench_path"] = f"{config['src_path']}/bench/{config['nCodons']}codons/{config['call_type']}_{config['order_transformed_input']}orderTransformedInput.log"
config["exit_after_first_batch"] = args.b
config["exit_after_loglik_is_nan"] = args.n
config["verbose"] = int(args.verbose) if args.verbose else 0
config["print_to_file"] = not args.verbose_to_stdout

print("config =", config)


nCodons = config["nCodons"]

run(f"mkdir -p {config['src_path']}/output/{nCodons}codons/")
run(f"mkdir -p {config['src_path']}/verbose")
run(f"mkdir -p {'/'.join(config['bench_path'].split('/')[:-1])}")
run(f"rm {config['src_path']}/{config['bench_path']}")
run(f"rm {config['src_path']}/verbose/{nCodons}codons.txt")

run(f"python3 {config['src_path']}/useMSAgen.py -c {nCodons}")

model, history = fit_model(config)
print("done fit_model()")
# model.save("my_saved_model")

with open(f"{config['src_path']}/output/{nCodons}codons/loss.log", "w") as file:
    for loss in history.history['loss']:
        file.write(str(loss))
        file.write("\n")

plt.plot(history.history['loss'])
plt.savefig(f"{config['src_path']}/progress.png")

cell = CgpHmmCell(config)
cell.transition_kernel = model.get_weights()[0]
cell.emission_kernel = model.get_weights()[1]
cell.init_kernel = model.get_weights()[2]

def printA():
    global cell, model
    print(".\t", end ="")
    for state in range(cell.state_size[0]):
        print(Utility.state_id_to_description(state, cell.nCodons), end = "\t")
    print()
    for state in range(cell.state_size[0]):
        print(Utility.state_id_to_description(state, cell.nCodons), end = "\t")
        for goal_state in cell.A[state]:
            print((tf.math.round(goal_state*100)/100).numpy(), end = "\t")
        print()
# printA()

def printB():
    global cell, model
    for state in range(len(cell.B)):
        tf.print(Utility.state_id_to_description(state, cell.nCodons))
        tf.print(tf.math.round(cell.B[state]*100).numpy()/100, summarize = -1)
        tf.print("---------------------------------------------")
# printB()

def printI():
    global cell, model
    for state in range(len(cell.I)):
        print(Utility.state_id_to_description(state, cell.nCodons), end = "\t")
        print(tf.math.round(cell.I[state,0]*100).numpy()/100)
# printI()

#  bc with call type 4 A_dense fails
if not config["call_type"] == 4:
    WriteData.write_to_file(cell.A_dense, f"{config['src_path']}/output/{nCodons}codons/A.{nCodons}codons.txt")
    WriteData.write_to_file(tf.transpose(cell.B_dense), f"{config['src_path']}/output/{nCodons}codons/B.{nCodons}codons.txt")
    WriteData.write_order_transformed_B_to_csv(cell.B_dense, f"{config['src_path']}/output/{nCodons}codons/B.{nCodons}codons.csv", config["order"], nCodons)

    WriteData.write_to_file(cell.I_dense, f"{config['src_path']}/output/{nCodons}codons/I.{nCodons}codons.txt")

    # running Viterbi
    run(f"{config['src_path']}/Viterbi " + config["fasta_path"] + " " + str(nCodons))

    stats = {"start_not_found" : 0,\
             "start_too_early" : 0,\
             "start_correct" : 0,\
             "start_too_late" : 0,\
             "stop_not_found" : 0,\
             "stop_too_early" : 0,\
             "stop_correct" : 0,\
             "stop_too_late" : 0}

    with open(f"{config['src_path']}/output/{nCodons}codons/viterbi.{nCodons}codons.csv", "r") as viterbi_file:
        with open(f"{config['src_path']}/output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt", "r") as start_stop_file:
            for v_line in viterbi_file:
                try:
                    ss_line = start_stop_file.readline()
                except:
                    print("ran out of line in :" + f"out.start_stop_pos.{nCodons}codons.txt")
                    quit(1)
                if ss_line[:3] == "seq" or len(ss_line) <= 1:
                    continue
                true_start = int(ss_line.split(";")[0])
                true_stop = int(ss_line.split(";")[1].strip())
                try:
                    viterbi_start = v_line.split("\t").index("stA")
                except:
                    viterbi_start = -1
                try:
                    viterbi_stop = v_line.split("\t").index("st1")
                except:
                    viterbi_stop = -1
                # print(f"true_start = {true_start} vs viterbi_start = {viterbi_start}")
                # print(f"true_stop = {true_stop} vs viterbi_stop = {viterbi_stop}")

                if viterbi_start == -1:
                    stats["start_not_found"] += 1
                    if viterbi_stop != -1:
                        print("found stop but not start")
                        quit(1)
                elif viterbi_start < true_start:
                    stats["start_too_early"] += 1
                elif viterbi_start == true_start:
                    stats["start_correct"] += 1
                else:
                    stats["start_too_late"] += 1

                if viterbi_stop == -1:
                    stats["stop_not_found"] += 1
                elif viterbi_stop < true_stop:
                    stats["stop_too_early"] += 1
                elif viterbi_stop == true_stop:
                    stats["stop_correct"] += 1
                else:
                    stats["stop_too_late"] += 1

    nSeqs = sum([v for v in stats.values()])/2 # div by 2 bc every seq appears twice in stats (in start and stop)

    with open(f"{config['src_path']}/output/{nCodons}codons/statistics.txt", "w") as file:
        for key, value in stats.items():
            file.write(key + "\t" + str(value/nSeqs) + "\n")

if config["nCodons"] < 10:
    run(f"python3 {config['src_path']}/Visualize.py -c {nCodons} -o {config['order']} {'-t' if config['order_transformed_input'] else ''}")
