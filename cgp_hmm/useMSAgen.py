#!/usr/bin/env python3
import sys

sys.path.insert(0, "../MSAgen")
import MSAgen

from Bio import SeqIO
import subprocess
import random
import numpy as np

from Utility import run

import argparse

parser = argparse.ArgumentParser(
    description='description')
parser.add_argument('-c', '--nCodons', required=True,
                    help='number of codons')
parser.add_argument('-l', '--length_factor',
                    help='length os seq is length of coding times this factor')
parser.add_argument('-n', '--num_seqs', type = int,
                    help='length os seq is length of coding times this factor')



args = parser.parse_args()
print("args.length_factor =", args.length_factor)
nCodons = int(args.nCodons)
num_seqs = 100 if not args.num_seqs else args.num_seqs

genlen = 3 * nCodons # ATG and STOP are not on gene
seqlen = genlen * (2 if not args.length_factor else float(args.length_factor))
seqlen += 6 # start and stop codon
seqlen += 2 # ig states
print("seqlen =", seqlen)


sequences, posDict = MSAgen.generate_sequences(num_sequences = int(num_seqs), # the number of sequences to generate
                                               seqlen = int(seqlen), # length of each sequence (in bp)
                                               genelen = int(genlen), # length of the gene in each sequence (in bp, can be 0)
                                               coding_dist = 0.2, # branch length of the underlying tree for simulated gene evolution
                                               noncoding_dist = 0.4) # branch length for flanking regions

coding_seqs = []
for seq in sequences:
    coding_seqs += [seq.seq[posDict["5flank_len"] : posDict["3flank_start"]]]
    # seq.name = ">" + seq.name
    # seq.description = ">" + seq.description
    # seq.id = ">" + seq.id

    print(seq)


# stripping seqs to have unequal lengths
strip_flanks = True
if strip_flanks:
    for seq in sequences:
        # strips seq somewhere in first half of 5flank
        # and somewhere in second half of 3flank

        assert posDict["5flank_len"] >= 2, "posDict[5flank_len] >= 2"
        assert posDict["3flank_len"] >= 2, "posDict[3flank_len] >= 2"

        strip_5flank_len = np.random.randint(1,posDict["5flank_len"])
        strip_3flank_len = np.random.randint(1,posDict["3flank_len"])

        print("seq.seq =", seq.seq)
        print(strip_5flank_len, strip_3flank_len)
        seq.seq = seq.seq[strip_5flank_len : -strip_3flank_len]
        print("seq.seq =", seq.seq)
        print()

        seq.startATGPos = posDict["start_codon"] - strip_5flank_len
        seq.stopPos     = posDict["stop_codon"] - strip_5flank_len

        # new_5_flank_len = random.randint(1, int(posDict["5flank_len"]))
        # new_3_flank_len = random.randint(1, int(posDict["3flank_len"]))
        #
        # assert new_5_flank_len >= 1, "new_5_flank_len is not >= 1"
        # assert new_3_flank_len >= 1, "new_3_flank_len is not >= 1"
        # assert (posDict["5flank_len"] - new_5_flank_len) >= 0, f"smaller 0: {(posDict['5flank_len'] - new_5_flank_len)}"
        # assert posDict["3flank_start"] + new_3_flank_len <= len(seq.seq), f"larger than seqlen {posDict['3flank_start'] + new_3_flank_len} <= {len(seq.seq)}"
        #
        # print_seq = True
        # if print_seq:
        #     print(seq.seq)
        #
        # seq.seq = seq.seq[(posDict["5flank_len"] - new_5_flank_len) : posDict["3flank_start"] + new_3_flank_len]
        # if print_seq:
        #     print("-" * (posDict["5flank_len"] - new_5_flank_len), end = "")
        #     print(seq.seq, end = "")
        #     print("-" * (posDict["3flank_len"] - new_3_flank_len))
        #
        #     print("-" * (posDict["5flank_len"] - new_5_flank_len), end = "")
        #     print("5" * new_5_flank_len, end = "")
        #     print("ATG", end = "")
        #     print("*" * genlen, end = "")
        #     print("STO", end = "")
        #     print("3" * new_3_flank_len, end = "")
        #     print("-" * (posDict["3flank_len"] - new_3_flank_len))
        #
        #
        # seq.startATGPos = new_5_flank_len
        # seq.stopPos = genlen + new_5_flank_len + 3
        # if print_seq:
        #     print(seq.seq)
        #     print("".join(["+" if i in [seq.startATGPos, seq.stopPos] else " " for i in range(len(seq.seq))]))
        #     print()
        # print("".join([str(i) for i in range(10)]*10))
        # print(seq.seq)
        # print("start codon =", seq.startATGPos)
        # print("stop codon =", seq.stopPos)

run(f"mkdir -p output/{nCodons}codons/")

# containing ATG and Stop
with open(f"output/{nCodons}codons/coding_seqs.{nCodons}codons.txt","w") as file:
    for sequence in coding_seqs:
        file.write(str(sequence))
        file.write("\n")

# writing a file that contains a profile of the coding_seqs
with open(f"output/{nCodons}codons/profile.{nCodons}codons.txt", "w") as file:
    for i in range(len(coding_seqs[0])):
        profile = {"A":0, "C":0, "G":0, "T":0}
        for seq in coding_seqs:
            profile[seq[i]] += 1/len(coding_seqs)
        for c in ["A","C","G","T"]:
            file.write(str(int(profile[c]*100)/100))
            file.write("\t")
        file.write("\n")

with open(f"output/{nCodons}codons/out.seqs.{nCodons}codons.fa","w") as file:

    # SeqIO.write(sequences, file, "fasta")

    for seq in sequences:
        file.write(">" + seq.id + "\n")
        file.write(str(seq.seq) + "\n")
import os
os.system(f"cat output/{nCodons}codons/out.seqs.{nCodons}codons.fa")

with open(f"output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt","w") as file:
    for seq in sequences:
        file.write(">" + seq.id)
        file.write("\n")
        file.write(str(seq.startATGPos) + ";" + str(seq.stopPos) + ";" + str(len(seq.seq)))
        file.write("\n")
        file.write("\n")

# run("/home/mattes/Documents/CGP-HMM-python-project/data/artificial/muscle3.8.31_i86linux64" +
#      f" -in output/{nCodons}codons/out.seqs.{nCodons}codons.fa -out output/{nCodons}codons/out.seqs.{nCodons}codons.align.fa -clw")

# run(f"head output/{nCodons}codons/out.seqs.{nCodons}codons.align.fa")

run(f"head output/{nCodons}codons/profile.{nCodons}codons.txt")

run(f"head output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt")
