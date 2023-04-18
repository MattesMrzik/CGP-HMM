#!/usr/bin/env python3
import argparse
from Bio import SeqIO
import pandas as pd


def find_ag_gt_pairs(sequence, min_distance, max_distance):
    ag_gt_pairs = []
    for i in range(len(sequence)-1):
        if sequence[i:i+2] == "AG":
            for j in range(i+2, min(i+max_distance+1, len(sequence)-1)):
                if sequence[j:j+2] == "GT":
                    separation = j-i-2
                    if separation >= min_distance and separation < max_distance:
                        ag_gt_pairs.append((i, j, separation))
    return ag_gt_pairs

parser = argparse.ArgumentParser()
parser.add_argument("fasta_file", help="path to input FASTA file")
parser.add_argument("--min_distance", type=int, default=10, help="minimum distance between AG and GT (default: 10)")
parser.add_argument("--max_distance", type=int, default=100, help="maximum distance between AG and GT (default: 100)")
args = parser.parse_args()

ag_gt_pairs = []
for sequence in SeqIO.parse(args.fasta_file, "fasta"):
    seq = sequence.seq
    pairs = find_ag_gt_pairs(str(sequence.seq), args.min_distance, args.max_distance)
    for pair in pairs:
        ag_gt_pairs.append({"AG_id": pair[0], "GT_id": pair[1], "separation": pair[2]})

df = pd.DataFrame(ag_gt_pairs)
pd.set_option("display.max_rows", None)
print(df)