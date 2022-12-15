#!/usr/bin/env python3
import sys


from Bio import SeqIO
import subprocess
import random
import numpy as np

from Utility import run

import argparse

parser = argparse.ArgumentParser(
    description='description')
parser.add_argument('-c', '--nCodons', required=True, help='number of codons')
parser.add_argument('-l', '--length_factor', default = 2.0, type = float, help='length os seq is length of coding times this factor')
parser.add_argument('-n', '--num_seqs', type = int, help='length os seq is length of coding times this factor')
parser.add_argument('-cd', '--coding_dist', type = float, default = 0.2, help='coding_dist')
parser.add_argument('-ncd', '--noncoding_dist', type = float, default = 0.4, help='noncoding_dist')
parser.add_argument('--dont_strip_flanks', action='store_true', help ="dont_strip_flanks")
parser.add_argument('-p', '--path', help = 'path to src')



args = parser.parse_args()

if not args.path:
    args.path = "."

sys.path.insert(0, f"{args.path}/../MSAgen")

import MSAgen

nCodons = int(args.nCodons)
num_seqs = 100 if not args.num_seqs else args.num_seqs

genlen = 3 * nCodons
seqlen = genlen * args.length_factor
seqlen += 6 # start and stop codon
seqlen += 2 # ig states

sequences, posDict = MSAgen.generate_sequences(num_sequences = int(num_seqs), # the number of sequences to generate
                                               seqlen = int(seqlen), # length of each sequence (in bp)
                                               genelen = int(genlen), # length of the gene in each sequence (in bp, can be 0)
                                               coding_dist = args.coding_dist, # branch length of the underlying tree for simulated gene evolution
                                               noncoding_dist = args.noncoding_dist) # branch length for flanking regions

with open(f"{args.path}/output/{nCodons}codons/MSAgen_untrancated_seqs.{nCodons}codons.fa","w") as file:
    for seq in sequences:
        file.write(">" + seq.id + "\n")
        file.write(str(seq.seq) + "\n")
coding_seqs = []
for seq in sequences:
    coding_seqs += [seq.seq[posDict["5flank_len"] : posDict["3flank_start"]]]
    # seq.name = ">" + seq.name
    # seq.description = ">" + seq.description
    # seq.id = ">" + seq.id

    # print(seq)


# stripping seqs to have unequal lengths

strip_flanks = not args.dont_strip_flanks
if strip_flanks:
    for seq in sequences:
        # strips seq somewhere in first half of 5flank
        # and somewhere in second half of 3flank

        assert posDict["5flank_len"] >= 2, "posDict[5flank_len] >= 2"
        assert posDict["3flank_len"] >= 2, "posDict[3flank_len] >= 2"

        strip_5flank_len = np.random.randint(1,posDict["5flank_len"])
        strip_3flank_len = np.random.randint(1,posDict["3flank_len"])

        # print("seq.seq =", seq.seq)
        # print(strip_5flank_len, strip_3flank_len)
        seq.seq = seq.seq[strip_5flank_len : -strip_3flank_len]
        # print("seq.seq =", seq.seq)
        # print()

        seq.startATGPos = posDict["start_codon"] - strip_5flank_len
        seq.stopPos     = posDict["stop_codon"] - strip_5flank_len


run(f"mkdir -p {args.path}/output/{nCodons}codons/")

# containing ATG and Stop
with open(f"{args.path}/output/{nCodons}codons/coding_seqs.{nCodons}codons.txt","w") as file:
    for sequence in coding_seqs:
        file.write(str(sequence))
        file.write("\n")

# writing a file that contains a profile of the coding_seqs
with open(f"{args.path}/output/{nCodons}codons/profile.{nCodons}codons.txt", "w") as file:
    for i in range(len(coding_seqs[0])):
        profile = {"A":0, "C":0, "G":0, "T":0}
        for seq in coding_seqs:
            profile[seq[i]] += 1/len(coding_seqs)
        for c in ["A","C","G","T"]:
            file.write(str(int(profile[c]*100)/100))
            file.write("\t")
        file.write("\n")

with open(f"{args.path}/output/{nCodons}codons/out.seqs.{nCodons}codons.fa","w") as file:

    # SeqIO.write(sequences, file, "fasta")

    for seq in sequences:
        for i in range(32):
            file.write(">" + seq.id + "000" + str(i) + "\n")
            file.write(str(seq.seq) + "\n")

run(f"head {args.path}/output/{nCodons}codons/out.seqs.{nCodons}codons.fa")

with open(f"{args.path}/output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt","w") as file:
    for seq in sequences:
        file.write(">" + seq.id)
        file.write("\n")
        if args.dont_strip_flanks:
            file.write(str(posDict['start_codon']) + ";" + str(posDict['stop_codon']) + ";" + str(len(seq.seq)))
        else:
            file.write(str(seq.startATGPos) + ";" + str(seq.stopPos) + ";" + str(len(seq.seq)))
        file.write("\n")
        file.write("\n")

# run("/home/mattes/Documents/CGP-HMM-python-project/data/artificial/muscle3.8.31_i86linux64" +
#      f" -in output/{nCodons}codons/out.seqs.{nCodons}codons.fa -out output/{nCodons}codons/out.seqs.{nCodons}codons.align.fa -clw")

# run(f"head output/{nCodons}codons/out.seqs.{nCodons}codons.align.fa")

run(f"head {args.path}/output/{nCodons}codons/profile.{nCodons}codons.txt")

run(f"head {args.path}/output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt")
