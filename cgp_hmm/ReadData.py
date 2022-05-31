#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from Bio import SeqIO

def read_data_one_hot(path, alphabet = ["A","C","G","T"]):
    seqs = []
    AA_to_id = dict([(aa, id) for id, aa in enumerate(alphabet)])
    with open(path,"r") as handle:
        for record in SeqIO.parse(handle,"fasta"):
            seq = record.seq
            seq = list(map(lambda x: AA_to_id[x], seq))
            seqs.append(seq)

    return tf.one_hot(seqs, len(alphabet))
