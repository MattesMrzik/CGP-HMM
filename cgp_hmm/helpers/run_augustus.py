#!/usr/bin/env python3

import os
import re
import subprocess
import pandas
from Bio import SeqIO
import json

from Viterbi import get_true_gene_strucutre_from_fasta_description_as_fasta

def create_gff_from_fasta_description(exon_dir_path : str) -> None:
    path_to_human_fasta = f"{exon_dir_path}/species_seqs/stripped/Homo_sapiens.fa"
    for i, record in enumerate(SeqIO.parse(path_to_human_fasta, "fasta")):
        assert i == 0, f"there was more than one seq in {path_to_human_fasta}"
        assert (x:= re.search("(\{.*\})", record.description)), f"no matching description found in '{record.description}' of file '{path_to_human_fasta}'"
        description = json.loads(x.group(1))
        # {'seq_start_in_genome_+_strand': 90231027, 'seq_stop_in_genome_+_strand': 90231351, 'exon_start_in_human_genome_+_strand': 90231094, 'exon_stop_in_human_genome_+_strand': 90231213, 'seq_start_in_genome_cd_strand': 90231351, 'seq_stop_in_genome_cd_strand': 90231027, 'exon_start_in_human_genome_cd_strand': 90231213, 'exon_stop_in_human_genome_cd_strand': 90231094, 'middle_exon': 1}
        _,_, d = get_true_gene_strucutre_from_fasta_description_as_fasta(description)
        print(d["exon_start"],d["exon_end"])

def run_augustus_for_dir(path : str) -> None:
    for i, file_or_dir in enumerate(os.listdir(path)):
        if i > 64:
            break
        full_path = os.path.join(path, file_or_dir)
        if os.path.isdir(full_path):

            create_gff_from_fasta_description(full_path)
            command = f"{path_to_augustus_exe} --species=human --strand=forward {full_path}/species_seqs/stripped/Homo_sapiens.fa"

            try:
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
                # The captured output will be stored in the 'output' variable
                out_array = output.split("\n")
                non_comments = [line for line in out_array if (not line.startswith("#") and len(line) > 0)]
                if len(non_comments) == 0:
                    continue
                non_comments = [line.split("\t") for line in non_comments]
                df = pandas.DataFrame(non_comments)
                print(i)
                df.columns = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
                print(full_path)
                print(df)
            except subprocess.CalledProcessError as e:
                print("Error executing command:", e.output)

            # exit_code = subprocess.call(re.split("\s+", command))#, stderr = err_handle, stdout = out_handle)
            # if exit_code != 0:
            #     print("exit_code:", exit_code)

if __name__ == "__main__":
    path = "/home/s-mamrzi/cgp_data/good_exons_2"
    path_to_augustus_exe ="/home/s-mamrzi/Augustus/bin/augustus"
    run_augustus_for_dir(path)