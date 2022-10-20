#!/usr/bin/env python3

# import cgp_hmm
import matplotlib.pyplot as plt
import tensorflow as tf
from Training import fit_model
from Training import make_dataset
import Utility
import numpy as np
from Bio import SeqIO
from Utility import run
import WriteData

from CgpHmmCell import CgpHmmCell

def prRed(skk): print(f"Cell\033[91m {skk} \033[00m")

nCodons = 1
order_transformed_input = True
order = 2

run(f"python3 useMSAgen.py -c {nCodons}")

path = f"output/{nCodons}codons/out.seqs.{nCodons}codons.fa"

prRed("path = " + path)
model, history = fit_model(path, nCodons, order_transformed_input, order)
# model.save("my_saved_model")

plt.plot(history.history['loss'])
plt.savefig("progress.png")

cell = CgpHmmCell(nCodons, order_transformed_input)
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

WriteData.write_to_file(cell.A, f"output/{nCodons}codons/A.{nCodons}codons.txt")
WriteData.write_to_file(cell.B, f"output/{nCodons}codons/B.{nCodons}codons.txt")
WriteData.write_order_transformed_B_to_csv(cell.B, f"output/{nCodons}codons/B.{nCodons}codons.csv", order, nCodons)

WriteData.write_to_file(cell.I, f"output/{nCodons}codons/I.{nCodons}codons.txt")

# running Viterbi
run("./Viterbi " + path + " " + str(nCodons))

stats = {"start_not_found" : 0,\
         "start_too_early" : 0,\
         "start_correct" : 0,\
         "start_too_late" : 0,\
         "stop_not_found" : 0,\
         "stop_too_early" : 0,\
         "stop_correct" : 0,\
         "stop_too_late" : 0}

with open(f"output/{nCodons}codons/viterbi.{nCodons}codons.csv", "r") as viterbi_file:
    with open(f"output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt", "r") as start_stop_file:
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

with open(f"output/{nCodons}codons/statistics.txt", "w") as file:
    for key, value in stats.items():
        file.write(key + "\t" + str(value/nSeqs) + "\n")

run(f"python3 Visualize.py -c {nCodons} -o {order} {'-t' if order_transformed_input else ''}")
