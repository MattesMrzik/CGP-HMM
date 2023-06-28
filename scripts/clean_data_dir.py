#!/usr/bin/env python3
import os
import argparse
import re

parser = argparse.ArgumentParser(description='Config module description')
parser.add_argument('-p', required=1, help='number of codons')
parser.add_argument('-d', type = int, default = 0, help='delete dirs with less -d subfiles')
args = parser.parse_args()

entries = sorted(list(os.listdir(args.p)))
for entry in entries:
    entry = os.path.join(args.p, entry)
    if os.path.isdir(entry):
        if not re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}", entry):
            continue
        n_sub_files = len(list(os.listdir(entry)))
        print(entry, n_sub_files)
        if n_sub_files < args.d:
            os.system(f"mv {entry} {args.p}/trash")