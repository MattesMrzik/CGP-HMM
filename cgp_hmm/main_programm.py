#!/usr/bin/env python3

# import cgp_hmm
import matplotlib.pyplot as plt
import tensorflow as tf
from Training import fit_model
import Utility
import numpy as np
from Bio import SeqIO

from CgpHmmCell import CgpHmmCell

#
# states, emissions = Utility.generate_state_emission_seqs(a,b,n,l)
# with open("../rest/sparse_toy_emissions.out","w") as out_handle:
#     for id, emission in enumerate(emissions):
#         out_handle.write(">id" + str(id) + "\n")
#         out_handle.write("".join([["A","C","G","T"][i] for i in emission]) + "\n")

model, history = fit_model("../data/artificial/single_exon_gene_flanked_by_nonCoding/seq_gen.out.with_utr.fasta")
# model.save("my_saved_model")

plt.plot(history.history['loss'])
plt.show()

cell = CgpHmmCell()
nCodons_used_in_model = 2

cell.transition_kernel = model.get_weights()[0]
print(tf.math.round(cell.A*100)/100)

cell.emission_kernel = model.get_weights()[1]
print(tf.math.round(cell.B*100)/100)
