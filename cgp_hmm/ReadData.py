#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from Bio import SeqIO
from Utility import run
from Utility import higher_order_emission_to_id
from Utility import get_dicts_for_emission_tuple_and_id_conversion
import re

def read_data_one_hot(path, alphabet = ["A","C","G","T"]):
    seqs = []
    base_to_id = dict([(base, id) for id, base in enumerate(alphabet)])
    with open(path,"r") as handle:
        for record in SeqIO.parse(handle,"fasta"):
            seq = record.seq
            seq = list(map(lambda x: base_to_id[x], seq))
            seqs.append(seq)

    return tf.one_hot(seqs, len(alphabet))

def read_data(path, alphabet = ["A","C","G","T"]):
    seqs = []
    base_to_id = dict([(base, id) for id, base in enumerate(alphabet)])
    with open(path,"r") as handle:
        for record in SeqIO.parse(handle,"fasta"):
            seq = record.seq
            seq = list(map(lambda x: base_to_id[x], seq))
            seqs.append(seq)

    return seqs

# let order = 1, seq = AG
# then seqs = (IA = 4*(alphabet_size (=4) + 1)^1 + 0*5^0) = 20
#             (AG = 0*(alphabet_size (=4) + 1)^1 + 2*5^0) = 2
def read_data_with_order(path, order, alphabet = ["A","C","G","T"], add_one_terminal_symbol = False, verbose = False):
    seqs = []
    def log(s):
        if verbose:
            print(s)
    emi_to_id = get_dicts_for_emission_tuple_and_id_conversion(alphabet_size = len(alphabet), order = order)[0]
    base_to_id = dict([(base, id) for id, base in enumerate(alphabet)])
    with open(path,"r") as handle:
        log(f"opened: {path}")
        for record in SeqIO.parse(handle,"fasta"):
            seq = record.seq
            log(f"seq = {seq}")
            seq_of_tuple_ids = [] # 21, 124
            last_bases = [4] * order # 4 is padded left flank
            for base in seq:
                t = (last_bases + [base_to_id[base]])
                seq_of_tuple_ids.append(emi_to_id[tuple(t)])
                last_bases = last_bases[1:] + [base_to_id[base]]
            if add_one_terminal_symbol:
                seq_of_tuple_ids.append(emi_to_id[tuple("X")])
            seqs.append(seq_of_tuple_ids)
    log(f"read {len(seqs)} sequnces")
    return seqs

def get_batch_input_from_tf_printed_file(path):
    input = []
    with open(path, "r") as file:
        seq = []
        for line in file:
            line = line.strip()
            if len(line) == 0:
                input.append(seq)
                seq = []
            else:
                line = re.sub("[\[\]]","", line)
                line = line.split(" ")
                line = [float(x) for x in line]
                seq.append(line)
        if len(seq) != 0:
            input.append(seq)
    return tf.constant(input, dtype = tf.float32)
