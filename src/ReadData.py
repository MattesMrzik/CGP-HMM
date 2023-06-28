#!/usr/bin/env python3

import os
import tensorflow as tf
import numpy as np
from Bio import SeqIO
from Utility import run
import re
from Utility import append_time_ram_stamp_to_file
from itertools import product
import time

def convert_data_one_hot_with_Ns_spread_str_to_numbers(seqs) -> list[list[float]]:
    return list(map(lambda l: [[float(x) for x in i.split("_")] for i in l], seqs))

def read_data_one_hot_with_Ns_spread_str(config, add_one_terminal_symbol = False) -> list[list[str]]:
    '''
    bc i can only pad batches with scalar i have multi hot encoded strings for every base A,C,G,T,N,a,c,g,t
    and pad with str that encodes terminal emission
    then i can convert these a string corresponding to a base to its multi hot encoded version
    '''
    start = time.perf_counter()
    append_time_ram_stamp_to_file(f"read_data_one_hot_with_Ns_spread_str() start ", config.bench_path, start)
    seqs = []
    base_to_id_dict = dict([(base, id) for id, base in enumerate("ACGTI")])
    def base_to_id(b):
        b = b.upper()
        try:
            return base_to_id_dict[b]
        except:
            return 5

    with open(config.fasta_path,"r") as handle:
        if config.only_primates or config.only_diverse:
            with open(config.primates_path, "r") as primates_handle:
                primates = set()
                # this file is just a list of ids of primates
                for line in primates_handle:
                    primates.add(line.strip())

            # a list containing the ids of the species that should be used as input
            species_to_use = []
            species_in_combinded_fasta = set()
            number_of_primats_in_current_fasta = 0

            for record in SeqIO.parse(handle,"fasta"):
                species_name = re.search("(\w+?_\w+?)\.", record.id).group(1)
                species_in_combinded_fasta.add(species_name)
                if species_name in primates:
                    number_of_primats_in_current_fasta += 1
                    if config.only_primates:
                        species_to_use.append(species_name)
            # print("number_of_primats_in_current_fasta", number_of_primats_in_current_fasta)

        if config.only_diverse and config.only_diverse != -1:
            # list all files in a dir whos path is stored in config.only_diverse
            all_files = []
            for file in os.listdir(config.only_diverse):
                if file.endswith(".txt"):
                    all_files.append(os.path.join(config.only_diverse, file))
            all_files = sorted(all_files)
            for file in all_files:
                # print("checking file:", file)
                species_in_file = set()
                with open(file, "r") as file_handle:
                    for line in file_handle:
                        species_in_file.add(line.strip())
                # find the size of the overlap of the set species_in_combinded_fasta and species_in_file
                overlap = species_in_combinded_fasta.intersection(species_in_file)
                # print("species_in_combinded_fasta", species_in_combinded_fasta)
                # print("species_in_file", species_in_file)
                if len(overlap) == number_of_primats_in_current_fasta:
                    # print("found file with same size as primates")

                    # TODO i have to add human sqeuence to the input
                    overlap.add("Homo_sapiens")
                    species_to_use = list(overlap)
                    break

    with open(config.fasta_path,"r") as handle:
        for record in SeqIO.parse(handle,"fasta"):
            if config.only_primates or config.only_diverse:
                # print("using species:", species_to_use)
                species_name = re.search("(\w+?_\w+?)\.", record.id).group(1)
                # print("species_to_use", species_to_use)
                # print("species_name", species_name)
                if species_name not in species_to_use:
                    continue

            if config.only_human_to_train:
                species_name = re.search("(\w+?_\w+?)\.", record.id).group(1)
                # print("species_to_use", species_to_use)
                # print("species_name", species_name)
                if not re.search("Homo_sap",species_name):
                    continue


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
    append_time_ram_stamp_to_file(f"read_data_one_hot_with_Ns_spread_str() end ", config.bench_path, start)
    assert len(seqs) != 0, "no seqs read"
    print("actual number of species used as input:", len(seqs))

    if config.only_human_to_train:
        seqs.append(seqs[0])
        print("added human sequence to input")
        print("len(seqs)", len(seqs))

    return seqs

def remove_long_seqs(seqs):
    len_of_seqs = [len(seq) for seq in seqs]
    # median
    med = np.median(len_of_seqs)
    # create a list starting in 2 ending in 10 with step size 0.1
    l = np.arange(2,10,0.1)
    for f in l:
        number_of_too_long_seqs = 0
        for lenght in len_of_seqs:
            if lenght > f * med:
                number_of_too_long_seqs += 1
        if number_of_too_long_seqs < 0.09 * len(seqs):
            break

    # remove all seqs that are longer than f * med
    seqs = [seq for seq in seqs if len(seq) <= f * med]

    # assert that there are still more than 90% of the seqs left
    assert len(seqs) > 0.89 * len(len_of_seqs), "to many seqs removed"
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

