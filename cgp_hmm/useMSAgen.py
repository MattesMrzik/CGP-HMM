#!/usr/bin/env python3
import sys

sys.path.insert(0, "../MSAgen")
import MSAgen

from Bio import SeqIO
import subprocess
import random

from Utility import run

import argparse

parser = argparse.ArgumentParser(
    description='description')
parser.add_argument('-c', '--nCodons', required=True,
                    help='number of codons')

args = parser.parse_args()

nCodons = int(args.nCodons)

seqlen = int(nCodons * 3 * 2 + 6)

sequences, posDict = MSAgen.generate_sequences(num_sequences = 100, # the number of sequences to generate
                                               seqlen = seqlen, # length of each sequence (in bp)
                                               genelen = int(nCodons * 3), # length of the gene in each sequence (in bp, can be 0)
                                               coding_dist = 0.2, # branch length of the underlying tree for simulated gene evolution
                                               noncoding_dist = 0.4) # branch length for flanking regions

coding_seqs = []
for seq in sequences:
    coding_seqs += [seq.seq[posDict["5flank_len"] : posDict["3flank_start"]]]


# stripping seqs to have unequal lengths
strip_flanks = True
if strip_flanks:
    for seq in sequences:
        # strips seq somewhere in first half of 5flank
        # and somewhere in second half of 3flank
        strip_left_length = random.randint(0, int(posDict["5flank_len"]/2))
        strip_right_length = random.randint(int(posDict['3flank_start'] \
                             + (seqlen - posDict['3flank_start'])/2) \
                             , int(seqlen))
        seq.seq = seq.seq[strip_left_length : strip_right_length]
        seq.startATGPos = posDict["start_codon"] - strip_left_length
        seq.stopPos = posDict["stop_codon"] - strip_left_length
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
    SeqIO.write(sequences, file, "fasta")

with open(f"output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt","w") as file:
    for seq in sequences:
        file.write(seq.id)
        file.write("\n")
        file.write(str(seq.startATGPos) + ";" + str(seq.stopPos))
        file.write("\n")
        file.write("\n")

# run("/home/mattes/Documents/CGP-HMM-python-project/data/artificial/muscle3.8.31_i86linux64" +
#      f" -in output/{nCodons}codons/out.seqs.{nCodons}codons.fa -out output/{nCodons}codons/out.seqs.{nCodons}codons.align.fa -clw")

run(f"head output/{nCodons}codons/out.seqs.{nCodons}codons.align.fa")

run(f"head output/{nCodons}codons/profile.{nCodons}codons.txt")

run(f"head output/{nCodons}codons/out.start_stop_pos.{nCodons}codons.txt")
