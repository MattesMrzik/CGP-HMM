#!/usr/bin/env python3

cfg = {}

# or read files in a passed dir
fasta_dir_path = "../../cgp_data/good_exons"
exons = ["chr1_123_123", \
         "chr1_234_234"]
cfg["fasta"] = [f"{fasta_dir_path}/{exon}/combinded.fasta" for exon in exons]