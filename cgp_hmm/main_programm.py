#!/usr/bin/env python3

# import cgp_hmm
import matplotlib.pyplot as plt
import tensorflow as tf
from Training import fit_model
from Training import make_dataset
import Utility
import numpy as np
from Bio import SeqIO
from pyutils import run # my local python package

from CgpHmmCell import CgpHmmCell

def prRed(skk): print(f"Cell\033[91m {skk} \033[00m")

# states, emissions = Utility.generate_state_emission_seqs(a,b,n,l)
# with open("../rest/sparse_toy_emissions.out","w") as out_handle:
#     for id, emission in enumerate(emissions):
#         out_handle.write(">id" + str(id) + "\n")
#         out_handle.write("".join([["A","C","G","T"][i] for i in emission]) + "\n")

path = "../data/artificial/single_exon_gene_flanked_by_nonCoding/seq_gen.out.with_utr.fasta"
path = "/home/mattes/Seafile/Meine_Bibliothek/Uni/Master/CGP-HMM-python-project/data/artificial/MSAgen/test.out.seqs.fa"

nCodons = 3
run(f"python3 ../data/artificial/MSAgen/useMSAgen.py -c {nCodons}")

path = f"/home/mattes/Documents/CGP-HMM-python-project/cgp_hmm/out.seqs.{nCodons}codons.fa"

prRed("path = " + path)
model, history = fit_model(path, nCodons)
# model.save("my_saved_model")

plt.plot(history.history['loss'])
plt.savefig("progress.png")

cell = CgpHmmCell(nCodons)
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

Utility.write_to_file(cell.A, f"A.{nCodons}codons.txt")
Utility.write_to_file(cell.B, f"B.{nCodons}codons.txt")
Utility.write_to_file(cell.I, f"I.{nCodons}codons.txt")

# running Viterbi
run("./main " + path + " " + str(nCodons))

stats = {"start_not_found" : 0,\
         "start_too_early" : 0,\
         "start_correct" : 0,\
         "start_too_late" : 0,\
         "stop_not_found" : 0,\
         "stop_too_early" : 0,\
         "stop_correct" : 0,\
         "stop_too_late" : 0}

with open(f"viterbi.{nCodons}codons.csv", "r") as viterbi_file:
    with open(f"out.start_stop_pos.{nCodons}codons.txt", "r") as start_stop_file:
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
for key, value in stats.items():
    print(key, value/nSeqs)
