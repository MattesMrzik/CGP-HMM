#!/usr/bin/env python3

import re
import shutil
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse
import pandas as pd
import os

from get_internal_exon import combine_fasta_files, extract_info_and_check_bed_file, get_input_files_with_human_at_0


import sys
sys.path.append("..")
from multi_run import get_viterbi_aligned_seqs


def get_left_introns(dir_path):
    # read the stats table into a dataframe
    df = pd.read_csv(os.path.join(dir_path, "stats_table.csv"), sep = ";")

    total_rows = len(df)
    res = 50
    num_tenths = total_rows//res

    # loop over the rows in the df
    for i, (index, row) in enumerate(df.iterrows()):

        if num_tenths != 0 and (i + 1) % num_tenths == 0:
            print(f"get_left_introns [{'#' *((i + 1) // num_tenths)}{' ' * (res - (i + 1) // num_tenths)}]", end = "\r")

        exon_dir = os.path.join(dir_path, row["exon"])
        bed_dir_path = os.path.join(exon_dir, "species_bed")

        # create a new dir that holds the left inron seqs
        intron_dir_path = os.path.join(exon_dir, "introns")
        if not os.path.exists(intron_dir_path):
            os.makedirs(intron_dir_path)

        path_to_human_seq = os.path.join(exon_dir, "species_seqs/non_stripped/Homo_sapiens.fa")
        human_len = len(next(SeqIO.parse(path_to_human_seq, "fasta")).seq)
        path_to_human_bed = os.path.join(exon_dir, "human.bed")
        human_bed = pd.read_csv(path_to_human_bed, sep="\t", header=None)
        strand = human_bed.iloc[0, 5]


        for bed_file in os.listdir(bed_dir_path):
            species_name = bed_file.split(".")[0]
            # this needs to contain the "human_strand" key and the "len_of_seq_substring_in_human" key
            extra_exon_data = {}
            # read table into df
            extra_exon_data["human_strand"] = strand
            # get the length of the substring in the human sequence
            # get the len of the seq in this fasta using SeqIO
            extra_exon_data["len_of_seq_substring_in_human"] = human_len

            extra_seq_data = {}

            if not extract_info_and_check_bed_file(bed_dir_path, species_name, extra_seq_data, extra_exon_data):
                continue

            if not extra_seq_data["middle_exon"]:
                continue

            ca_intron_stop = extra_seq_data["middle_exon_lift_start"] - extra_seq_data["seq_start_in_genome"] - row["exon_len"]//2

            if ca_intron_stop < 0:
                continue

            # read the species fasta
            # path to stripped fasta seq of species
            species_fast_seq = os.path.join(exon_dir, "species_seqs/non_stripped", species_name + ".fa")
            try:
                record = next(SeqIO.parse(species_fast_seq, "fasta"))
            except:
                continue
            id = record.id
            seq = str(record.seq)
            ori_len = len(seq)
            intron_seq = seq[10 : ca_intron_stop-10]

            intron_path = os.path.join(intron_dir_path, species_name + ".fa")
            with open(intron_path, "w") as out_intron_handle:
                record = SeqRecord(Seq(intron_seq), id=id)
                SeqIO.write(record , out_intron_handle, "fasta")

            # i need to combine these
            # humna needs to be at first position


        # combine the intron seqs
        combined_fasta_path = os.path.join(exon_dir, "introns/combined.fasta")
        files = get_input_files_with_human_at_0(intron_dir_path)
        combine_fasta_files(combined_fasta_path, files)

def remove_only_human_exon(dir_path):

    for file_path in os.listdir(dir_path):
        fullpath = os.path.join(dir_path, file_path)
        print("fullpath", fullpath)
        if os.path.isdir(fullpath):
            path_to_missing_human_exon_dir = os.path.join(fullpath, "missing_human_exon")
            if not os.path.exists(path_to_missing_human_exon_dir):
                os.makedirs(path_to_missing_human_exon_dir)

            path_to_combined_fasta = os.path.join(fullpath, "combined.fasta")

            output_file = os.path.join(path_to_missing_human_exon_dir, "combined_fasta.fa")

            aligned_seqs = get_viterbi_aligned_seqs(fullpath, " ", true_alignemnt_path = os.path.join(fullpath, "true_alignment.clw"), no_asserts = True)
            print("aligned_seqs", aligned_seqs)
            if aligned_seqs ==  -1:
                continue

            human_seq_without_exon = []
            human_seq = [name for name in aligned_seqs.keys() if re.search("Homo_sapiens", name)][0]
            for base, true_state in zip(aligned_seqs[human_seq], aligned_seqs["true_seq"]):
                if true_state in ["l", "r"]:
                    human_seq_without_exon.append(base)

            human_seq_without_exon = "".join(human_seq_without_exon)

            records = []
            for record in SeqIO.parse(path_to_combined_fasta, "fasta"):
                if record.id == human_seq:
                    print("human_seq_without_exon", human_seq_without_exon)
                    new_record = SeqRecord(Seq(human_seq_without_exon), id=record.id)
                    records.append(new_record)
                else:
                    records.append(record)

            print()
            with open(output_file, "w") as handle:
                SeqIO.write(records, handle, "fasta")

if __name__ == "__main__":

    # add a agrument path to parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the stats table")
    args = parser.parse_args()

    get_left_introns(args.path)
    remove_only_human_exon(args.path)