#!/usr/bin/env python3
import sys
import argparse
from Bio import SeqIO
import random
import numpy as np
import os
import re

from Utility import run


parser = argparse.ArgumentParser(
    description='description')
parser.add_argument('-c', '--nCodons', required=True, help='number of codons')
parser.add_argument('-l', '--seq_len', required=True, type = int, help='seq_len')
parser.add_argument('-n', '--num_seqs', type = int, help='length os seq is length of coding times this factor')
parser.add_argument('-cd', '--coding_dist', type = float, default = 0.2, help='coding_dist')
parser.add_argument('-ncd', '--noncoding_dist', type = float, default = 0.4, help='noncoding_dist')
parser.add_argument('--dont_strip_flanks', action='store_true', help ="dont_strip_flanks")
parser.add_argument('-p', '--path', help = 'path to src')
parser.add_argument('--insertions', action = 'store_true', help = 'simulate inserions at random positions in seqs selected by random')
parser.add_argument('--deletions', action = 'store_true', help = 'simulate deletions at random positions in seqs selected by random')

args = parser.parse_args()

if not args.path:
    args.path = "."

sys.path.insert(0, f"{args.path}/../MSAgen")

import MSAgen

nCodons = int(args.nCodons)
num_seqs = 100 if not args.num_seqs else args.num_seqs

genlen = 3 * nCodons
seqlen = args.seq_len

if args.insertions:
    number_of_insertions = nCodons // 10
    genlen += 3 * number_of_insertions

sequences, posDict = MSAgen.generate_sequences(num_sequences = int(num_seqs), # the number of sequences to generate
                                               seqlen = int(seqlen), # length of each sequence (in bp)
                                               genelen = int(genlen), # length of the gene in each sequence (in bp, can be 0)
                                               coding_dist = args.coding_dist, # branch length of the underlying tree for simulated gene evolution
                                               noncoding_dist = args.noncoding_dist) # branch length for flanking regions

if not os.path.exists(f"{args.path}/output/{nCodons}codons/"):
    os.makedirs(f"{args.path}/output/{nCodons}codons/")

# MSAgen output
with open(f"{args.path}/output/{nCodons}codons/MSAgen.out","w") as file:
    for seq in sequences:
        file.write(">" + seq.id + "\n")
        file.write(str(seq.seq) + "\n")

# true state seq, with yet unnumerated codons and inserts
for seq in sequences:
    seq.true_state_seq = "5" * posDict["5flank_len"]
    seq.true_state_seq += "ATG"
    seq.true_state_seq += "C" * genlen
    seq.true_state_seq += "STP"
    seq.true_state_seq += "3" * posDict["3flank_len"]
    seq.startATGPos = posDict["start_codon"]
    seq.stopPos     = posDict["stop_codon"]

if args.insertions:
    print("useMSAgen: number_of_insertions =", number_of_insertions)
    positions_of_insertions = np.random.choice(np.arange(nCodons + number_of_insertions), size = number_of_insertions)
    positions_of_insertions = sorted(positions_of_insertions, reverse = True)
    percent_of_seq_that_have_insertion = .15
    for position_of_insertion in positions_of_insertions:
        seq_has_insertion = np.random.choice([True, False], size = num_seqs, p = [percent_of_seq_that_have_insertion, 1 - percent_of_seq_that_have_insertion])
        for i, seq in enumerate(sequences):
            position_of_insertion_in_seq = posDict["5flank_len"] + len("ATG") + 3 * position_of_insertion
            if not seq_has_insertion[i]:
                seq.seq = seq.seq[:position_of_insertion_in_seq] + "iii" + seq.seq[position_of_insertion_in_seq + 3:]
                seq.startATGPos -= 3
            seq.true_state_seq = seq.true_state_seq[:position_of_insertion_in_seq] + "iii" + seq.true_state_seq[position_of_insertion_in_seq + 3:]

if args.deletions:
    number_of_deletions = number_of_insertions
    print("useMSAgen: number_of_deletions =", number_of_deletions)
    possible_positions_of_deletions = list(set(np.arange(nCodons + number_of_insertions)).difference(positions_of_insertions))
    positions_of_deletions = np.random.choice(possible_positions_of_deletions, size = number_of_deletions)
    positions_of_deletions = sorted(positions_of_deletions, reverse = True)
    percent_of_seqs_that_have_deletions = .15
    for position_of_deletion in positions_of_deletions:
        seq_has_deletions = np.random.choice([True, False], size = num_seqs, p = [percent_of_seqs_that_have_deletions, 1 - percent_of_seqs_that_have_deletions])
        for i, seq in enumerate(sequences):
            position_of_deletion_in_seq = posDict["5flank_len"] + len("ATG") + 3 * position_of_deletion
            if seq_has_deletions[i]:
                seq.startATGPos -= 3
                seq.seq = seq.seq[:position_of_deletion_in_seq] + "ddd" + seq.seq[position_of_deletion_in_seq + 3:]

# stripping seqs to have unequal lengths
strip_flanks = not args.dont_strip_flanks
if strip_flanks:
    for seq in sequences:
        # strips seq somewhere in first half of 5flank
        # and somewhere in second half of 3flank

        assert posDict["5flank_len"] >= 1, "posDict[5flank_len] >= 2"
        assert posDict["3flank_len"] >= 1, "posDict[3flank_len] >= 2"

        strip_5flank_len = np.random.randint(0,posDict["5flank_len"])
        strip_3flank_len = np.random.randint(0,posDict["3flank_len"])

        # print("seq.seq =", seq.seq)
        # print(strip_5flank_len, strip_3flank_len)
        if strip_3flank_len != 0:
            seq.seq = seq.seq[strip_5flank_len : -strip_3flank_len]
            seq.true_state_seq = seq.true_state_seq[strip_5flank_len : -strip_3flank_len]
        else:
            seq.seq = seq.seq[strip_5flank_len :]
            seq.true_state_seq = seq.true_state_seq[strip_5flank_len :]
            # print("seq.seq =", seq.seq)
            # print()

        seq.startATGPos = posDict["start_codon"] - strip_5flank_len
        seq.stopPos     = posDict["stop_codon"] - strip_5flank_len

# write true MSA to file:
with open(f"{args.path}/output/{nCodons}codons/trueMSA.txt", "w") as file:
    # landmarks = ""
    # for i in range(seqlen):
    #     if i%10 == 0:
    #         landmarks = landmarks[:-len(str(i)) + 1] +  str(i)
    #     else:
    #         landmarks += " "
    # landmarks += "\n"
    # file.write(landmarks)

    # first line
    first_line = ""
    first_line +="5" * posDict["5flank_len"]
    first_line +="ATG"
    codon_id = 0
    i = 0
    while i < len(sequences[0].true_state_seq):
        if sequences[0].true_state_seq[i] == "C":
            first_line +="C" + (str(codon_id) if codon_id >= 10 else "0" + str(codon_id))
            codon_id += 1
            i += 3
        elif sequences[0].true_state_seq[i] == "i":
            first_line +="i" + (str(codon_id) if codon_id >= 10 else "0" + str(codon_id))
            i += 3
        else:
            i += 1
    first_line +="STP"
    first_line +="3" * posDict["3flank_len"]
    first_line +="\n"
    file.write(first_line)

    # end first line

    for seq in sequences:
        file.write("-" * (posDict["5flank_len"] - seq.startATGPos))
        file.write(str(seq.seq))
        file.write("-" * (len(first_line) -1 - len(str(seq.seq)) - posDict["5flank_len"] + seq.startATGPos))
        file.write("\n")
run(f"head {args.path}/output/{nCodons}codons/trueMSA.txt")

# fasta file
with open(f"{args.path}/output/{nCodons}codons/out.seqs.{nCodons}codons.fa","w") as file:
    # SeqIO.write(sequences, file, "fasta")
    for i, seq in enumerate(sequences):
        #                                 index of seq in batch
        file.write(">" + seq.id + "000" + str(i%32) + "\n")
        file.write(re.sub("iii|ddd", "", str(seq.seq)) + "\n")

run(f"head {args.path}/output/{nCodons}codons/out.seqs.{nCodons}codons.fa")

# profile of coding seq
# with open(f"{args.path}/output/{nCodons}codons/profile.{nCodons}codons.txt", "w") as file:
#     for i in range(len(coding_seqs[0])):
#         profile = {"A":0, "C":0, "G":0, "T":0}
#         for seq in coding_seqs:
#             profile[seq[i]] += 1/len(coding_seqs)
#         for c in ["A","C","G","T"]:
#             file.write(str(int(profile[c]*100)/100))
#             file.write("\t")
#         file.write("\n")
# run(f"head {args.path}/output/{nCodons}codons/profile.{nCodons}codons.txt")

# start and stop positions
with open(f"{args.path}/output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt","w") as file:
    for seq in sequences:
        file.write(">" + seq.id)
        file.write(";")
        file.write(str(seq.startATGPos) + ";" + str(seq.stopPos) + ";" + str(len(re.sub("iii|ddd", "", str(seq.seq)))))
        file.write("\n")

# run("/home/mattes/Documents/CGP-HMM-python-project/data/artificial/muscle3.8.31_i86linux64" +
#      f" -in output/{nCodons}codons/out.seqs.{nCodons}codons.fa -out output/{nCodons}codons/out.seqs.{nCodons}codons.align.fa -clw")

# run(f"head output/{nCodons}codons/out.seqs.{nCodons}codons.align.fa")


run(f"head {args.path}/output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt")
