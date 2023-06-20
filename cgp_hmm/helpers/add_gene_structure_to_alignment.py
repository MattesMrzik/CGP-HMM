#!/usr/bin/python3
import argparse
import re
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os


def get_reference_seq(input_algn, reference) -> tuple[Seq, AlignIO.MultipleSeqAlignment]:
    '''
    input_algn: path to clw alignment of combined.fasta
    reference = "Homo_sapiens"
    returns the seq name, here: Homo_sapiens.chr1
    '''
    alignment = AlignIO.read(input_algn, "clustal")
    for seq_record in alignment:
        if seq_record.id.startswith(reference):
            reference_seq = seq_record.seq


    return reference_seq, alignment

def read_true_alignment_with_out_coords_seq(true_alignment_path) -> AlignIO.MultipleSeqAlignment:
    '''
    true_alignment_path: path to file like:
    CLUSTAL X (1.81) multiple sequence alignment

    coords_fasta                        in this fasta 0, in genome 1045886
    numerate_line                       |         |         |         |         |
    Homo_sapiens.chr1                   GCCAGCGCTACCCTGGGGCttcatggggtggggtggggtcacCCGAGCCA
    true_seq                            llllllllllllllllllllllllllllllllllllllllllllllllll
    viterbi_guess                       lllllllllllllllllllllllllllllllll---AGc 0c 1c 2c 3

    returns the alignment without the seqs "coords_fasta" and "numerate_line"
    '''
    cured_true_alignment = ""
    with open(true_alignment_path, "r") as true_alignment_file:
        for line in true_alignment_file:
            if not line.startswith("coords_fasta") and not line.startswith("numerate_line"):
                cured_true_alignment += line


    temp_file_path = f"{true_alignment_path}_temp"
    with open(temp_file_path, "w") as temp:
        temp.write(cured_true_alignment)

    true_alignment = AlignIO.read(temp_file_path, "clustal")
    os.system(f"rm {temp_file_path}")

    return true_alignment


if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Insert gaps in a new sequence to match gaps in an existing alignment")

    parser.add_argument("--input_algn", type=str, help="path to input alignment file in CLUSTAL format")
    parser.add_argument("--output_algn", type=str, help="path to output alignment file in CLUSTAL format")
    parser.add_argument("--true_alignment", type=str, help="path to true alignment file in FASTA format")
    parser.add_argument("--reference", default = "Homo_sapiens", help="seq_record.id used as reference for gaps")

    args = parser.parse_args()

    true_alignment = read_true_alignment_with_out_coords_seq(args.true_alignment)

    reference_seq, alignment = get_reference_seq(args.input_algn, args.reference)


    for seq_record in true_alignment:
        if seq_record.id == "true_seq":
            true_state_seq = seq_record.seq

    # Insert gaps in the new sequence to match gaps in the alignment
    new_aligned_seq = ""
    j = 0
    for i in range(len(reference_seq)):
        if reference_seq[i] == "-":
            new_aligned_seq += "-"
        else:
            new_aligned_seq += true_state_seq[j]
            j += 1

    record = SeqRecord(Seq(new_aligned_seq),
                    id="true_seq")

    seqs = [record] + [seq for seq in alignment]
    alignment = MultipleSeqAlignment(seqs)

    # Write out the aligned sequences to a new CLUSTAL formatted file
    AlignIO.write(alignment, args.output_algn, "clustal")

