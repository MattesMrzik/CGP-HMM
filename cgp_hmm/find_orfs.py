#!/usr/bin/env python3
import re
import pandas as pd
from Bio import SeqIO
import argparse

def find_ag_gt_pairs(sequence, min_distance, max_distance):
    ag_gt_pairs = []
    pattern = re.compile(r"AG.{%d,%d}GT" % (min_distance, max_distance))
    matches = pattern.finditer(sequence)
    for match in matches:
        ag_gt_pairs.append((match.start(), match.end()-2, match.end()-match.start()-4))
    return ag_gt_pairs

parser = argparse.ArgumentParser()
parser.add_argument("fasta_file", help="path to input FASTA file")
parser.add_argument("--min_distance", type=int, default=10, help="minimum distance between AG and GT (default: 10)")
parser.add_argument("--max_distance", type=int, default=100, help="maximum distance between AG and GT (default: 100)")
args = parser.parse_args()

ag_gt_pairs = []
for sequence in SeqIO.parse(args.fasta_file, "fasta"):
    pairs = find_ag_gt_pairs(str(sequence.seq), args.min_distance, args.max_distance)
    for pair in pairs:
        ag_gt_pairs.append({"seq_id" : sequence.id, "AG_id": pair[0], "GT_id": pair[1], "separation": pair[2]})

df = pd.DataFrame(ag_gt_pairs)
print(df)