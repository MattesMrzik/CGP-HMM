#!/usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser(
    description='description')
parser.add_argument('-r', '--range_codon', nargs='+',
                    help='usage: < -r 1 10 > to run 1 2 3 4 5 6 7 8 9 10 codons')
parser.add_argument('-t', '--types', nargs="+",
                    help='types of cell.call() that should be used')
parser.add_argument('-c', '--nCodons', nargs="+",
                    help ='usage: < -c 10 20 50 > to run 10 20 50 codons')

args = parser.parse_args()

codons = []
if args.range_codon:
    if len(args.range_codon) == 2:
        codons += list(range(int(args.range_codon[0]), int(args.range_codon[1])+1))
    else:
        print("pass 2 integers a,b > 0 with a < b")
        exit(1)
if args.nCodons:
    for c in args.nCodons:
        codons.append(int(c))
codons = sorted(list(set(codons)))
if len(codons) == 0:
    print("pls specify nCodons")
    exit(1)

types = []
if args.types:
    for t in args.types:
        types.append(int(t))
types = sorted(list(set(types)))
if len(types) == 0:
    print("pls specify types")
    exit(1)

for c in codons:
    for t in types:
        os.system(f"./main_programm.py -c {c} -t {t}")

# for nCodons in range(11,26):
#     for type in [2,4]:
#         # run(f"/usr/bin/time --verbose --output bench/{nCodons}.bench.txt ./main_programm.py -c {nCodons} " )
#         os.system(f"./main_programm.py -c {nCodons} -t {type}")
