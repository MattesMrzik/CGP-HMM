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

#
# states, emissions = Utility.generate_state_emission_seqs(a,b,n,l)
# with open("../rest/sparse_toy_emissions.out","w") as out_handle:
#     for id, emission in enumerate(emissions):
#         out_handle.write(">id" + str(id) + "\n")
#         out_handle.write("".join([["A","C","G","T"][i] for i in emission]) + "\n")

path = "../data/artificial/single_exon_gene_flanked_by_nonCoding/seq_gen.out.with_utr.fasta"
path = "/home/mattes/Seafile/Meine_Bibliothek/Uni/Master/CGP-HMM-python-project/data/artificial/MSAgen/test.out.seqs.fa"
prRed("path = " + path)
model, history = fit_model(path)
# model.save("my_saved_model")

plt.plot(history.history['loss'])
plt.savefig("progress.png")

cell = CgpHmmCell()

# printing A
cell.transition_kernel = model.get_weights()[0]
print(".\t", end ="")
for state in range(cell.state_size[0]):
    print(Utility.state_id_to_description(state, cell.nCodons), end = "\t")
print()
for state in range(cell.state_size[0]):
    print(Utility.state_id_to_description(state, cell.nCodons), end = "\t")
    for goal_state in cell.A[state]:
        print((tf.math.round(goal_state*100)/100).numpy(), end = "\t")
    print()

# printing B
cell.emission_kernel = model.get_weights()[1]
for state in range(len(cell.B)):
    tf.print(Utility.state_id_to_description(state, cell.nCodons))
    tf.print(tf.math.round(cell.B[state]*100).numpy()/100, summarize = -1)
    tf.print("---------------------------------------------")

# printing I
cell.init_kernel = model.get_weights()[2]
for state in range(len(cell.I)):
    print(Utility.state_id_to_description(state, cell.nCodons), end = "\t")
    print(tf.math.round(cell.I[state,0]*100).numpy()/100)

Utility.write_to_file(cell.A, "A.txt")
Utility.write_to_file(cell.B, "B.txt")
Utility.write_to_file(cell.I, "I.txt")

# running Viterbi
run("./main " + path)
