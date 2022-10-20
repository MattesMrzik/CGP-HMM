#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from Bio import SeqIO
from Utility import run
from Utility import higher_order_emission_to_id

def read_data_one_hot(path, alphabet = ["A","C","G","T"]):
    seqs = []
    AA_to_id = dict([(aa, id) for id, aa in enumerate(alphabet)])
    with open(path,"r") as handle:
        for record in SeqIO.parse(handle,"fasta"):
            seq = record.seq
            seq = list(map(lambda x: AA_to_id[x], seq))
            seqs.append(seq)

    return tf.one_hot(seqs, len(alphabet))

def read_data(path, alphabet = ["A","C","G","T"]):
    seqs = []
    AA_to_id = dict([(aa, id) for id, aa in enumerate(alphabet)])
    with open(path,"r") as handle:
        for record in SeqIO.parse(handle,"fasta"):
            seq = record.seq
            seq = list(map(lambda x: AA_to_id[x], seq))
            seqs.append(seq)

    return seqs

# let order = 1, seq = AG
# then seqs = (IA = 4*(alphabet_size (=4) + 1)^1 + 0*5^0) = 20
#             (AG = 0*(alphabet_size (=4) + 1)^1 + 2*5^0) = 2
def read_data_with_order(path, order, alphabet = ["A","C","G","T"]):
    seqs = []
    AA_to_id = dict([(aa, id) for id, aa in enumerate(alphabet)])
    with open(path,"r") as handle:
        for record in SeqIO.parse(handle,"fasta"):
            seq = record.seq
            seq_of_tuple_ids = []
            last_bases = [4] * order # 4 is padded left flank
            for base in seq:
                t = (last_bases + [AA_to_id[base]])
                print("t =", t)
                seq_of_tuple_ids.append(higher_order_emission_to_id(t, len(alphabet), order))
                last_bases = last_bases[1:] + [AA_to_id[base]]
            seqs.append(seq_of_tuple_ids)
            break
    return seqs
