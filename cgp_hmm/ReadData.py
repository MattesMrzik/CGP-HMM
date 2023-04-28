#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from Bio import SeqIO
from Utility import run
import re
from Utility import append_time_ram_stamp_to_file
from itertools import product
from random import randint
import time

# def read_data_one_hot(path, alphabet = ["A","C","G","T"]):
#     seqs = []
#     base_to_id = dict([(base, id) for id, base in enumerate(alphabet)])
#     with open(path,"r") as handle:
#         for record in SeqIO.parse(handle,"fasta"):
#             seq = record.seq
#             seq = list(map(lambda x: base_to_id[x], seq))
#             seqs.append(seq)
#
#     return tf.one_hot(seqs, len(alphabet))
#
# def read_data(path, alphabet = ["A","C","G","T"]):
#     seqs = []
#     base_to_id = dict([(base, id) for id, base in enumerate(alphabet)])
#     with open(path,"r") as handle:
#         for record in SeqIO.parse(handle,"fasta"):
#             seq = record.seq
#             seq = list(map(lambda x: base_to_id[x], seq))
#             seqs.append(seq)
#
#     return seqs

def convert_data_one_hot_with_Ns_spread_str_to_numbers(seqs) -> list[list[float]]:
    return list(map(lambda l: [[float(x) for x in i.split("_")] for i in l], seqs))

def read_data_one_hot_with_Ns_spread_str(config, add_one_terminal_symbol = False) -> list[list[str]]:
    '''
    bc i can only pad batches with scalar i have multi hot encoded strings for every base A,C,G,T,N,a,c,g,t
    and pad with str that encodes terminal emission
    then i can convert these a string corresponding to a base to its multi hot encoded version
    '''
    start = time.perf_counter()
    run_id = randint(0,100)
    append_time_ram_stamp_to_file(f"read_data_one_hot_with_Ns_spread_str() start {run_id}", config.bench_path, start)
    seqs = []
    base_to_id_dict = dict([(base, id) for id, base in enumerate("ACGTI")])
    def base_to_id(b):
        try:
            return base_to_id_dict[b]
        except:
            return 5
    with open(config.fasta_path,"r") as handle:
        for record in SeqIO.parse(handle,"fasta"):
            seq = record.seq
            seq_of_one_hot = []
            last_bases = [4] * config.order # 4 is padded left flank
            for i, base in enumerate(seq):
                t = (last_bases + [base_to_id(base)])
                matching_ids = [] # t might contain N, so more that one id match to this tuple
                allowd_bases = [[0,1,2,3] if b == 5 else [b] for b in t]
                # TODO hier sind dann ja auch stop codons erlaubt
                # B hat dann aber an der position einfach ein null, sodass die
                # wkeiten der restlichen 3 emisioionenn genommen werden
                # wenn aber mehr als 1 N in t ist, dass ist das vielleicht komisch gewichtet
                allowed_ho_emissions = list(product(*allowd_bases))
                entry_of_one_hot = np.zeros(config.model.number_of_emissions)
                for allowed_ho_emission in allowed_ho_emissions:
                    entry_of_one_hot[config.model.emission_tuple_to_id(allowed_ho_emission)] = 1
                entry_of_one_hot /= sum(entry_of_one_hot)
                entry_of_one_hot = "_".join([str(x) for x in entry_of_one_hot])
                seq_of_one_hot.append(entry_of_one_hot)
                last_bases = last_bases[1:] + [base_to_id(base)] if config.order > 0 else []
                # print(list(entry_of_one_hot))
                # for i in range(len(entry_of_one_hot)):
                #     if entry_of_one_hot[i] != 0:
                #         print(config.model.emission_id_to_str(i), end = ", ")
            if add_one_terminal_symbol:
                entry_of_one_hot = np.zeros(config.model.number_of_emissions, dtype = np.float32)
                entry_of_one_hot[config.model.emission_tuple_to_id("X")] = 1
                entry_of_one_hot = "_".join([str(x) for x in entry_of_one_hot])
                seq_of_one_hot.append(entry_of_one_hot)
            seqs.append(seq_of_one_hot)
    append_time_ram_stamp_to_file(f"read_data_one_hot_with_Ns_spread_str() end   {run_id}", config.bench_path, start)
    return seqs
# def read_data_one_hot_with_Ns_spread(config, add_one_terminal_symbol = False):
#     seqs = []
#     base_to_id = dict([(base, id) for id, base in enumerate("ACGTIN")])
#     with open(config.fasta_path,"r") as handle:
#         for record in SeqIO.parse(handle,"fasta"):
#             seq = record.seq
#             seq_of_one_hot = []
#             last_bases = [4] * config.order # 4 is padded left flank
#             for i, base in enumerate(seq):
#                 t = (last_bases + [base_to_id[base]])
#                 matching_ids = [] # t might contain N, so more that one id match to this tuple
#                 allowd_bases = [[0,1,2,3] if b == 5 else [b] for b in t]
#                 # TODO hier sind dann ja auch stop codons erlaubt
#                 # B hat dann aber an der position einfach ein null, sodass die
#                 # wkeiten der restlichen 3 emisioionenn genommen werden
#                 # wenn aber mehr als 1 N in t ist, dass ist das vielleicht komisch gewichtet
#                 allowed_ho_emissions = list(product(*allowd_bases))
#                 entry_of_one_hot = np.zeros(config.model.number_of_emissions)
#                 for allowed_ho_emission in allowed_ho_emissions:
#                     entry_of_one_hot[config.model.emission_tuple_to_id(allowed_ho_emission)] = 1
#                 entry_of_one_hot /= sum(entry_of_one_hot)
#                 seq_of_one_hot.append(entry_of_one_hot)
#                 last_bases = last_bases[1:] + [base_to_id[base]] if config.order > 0 else []
#                 # print(list(entry_of_one_hot))
#                 # for i in range(len(entry_of_one_hot)):
#                 #     if entry_of_one_hot[i] != 0:
#                 #         print(config.model.emission_id_to_str(i), end = ", ")
#             if add_one_terminal_symbol:
#                 entry_of_one_hot = np.zeros(config.model.number_of_emissions)
#                 entry_of_one_hot[config.model.emission_tuple_to_id("X")] = 1
#                 seq_of_one_hot.append(entry_of_one_hot)
#             seqs.append(seq_of_one_hot)
#     return seqs


def read_data_with_order(config, alphabet = ["A","C","G","T"], add_one_terminal_symbol = False, verbose = False):
    seqs = []
    def log(s):
        if verbose:
            print(s)

    base_to_id = dict([(base, id) for id, base in enumerate(alphabet)])
    with open(config.fasta_path,"r") as handle:
        log(f"opened: {config.fasta_path}")
        for record in SeqIO.parse(handle,"fasta"):
            seq = record.seq
            log(f"seq = {seq}")
            seq_of_tuple_ids = [] # 21, 124
            last_bases = [4] * config.order # 4 is padded left flank
            for base in seq:
                t = (last_bases + [base_to_id[base]])
                seq_of_tuple_ids.append(config.model.emission_tuple_to_id(t))
                last_bases = last_bases[1:] + [base_to_id[base]] if config.order > 0 else []
            if add_one_terminal_symbol:
                seq_of_tuple_ids.append(config.model.emission_tuple_to_id("X"))
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
