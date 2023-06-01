#!/usr/bin/env python3

import os
import re
import subprocess
import pandas
from Bio import SeqIO
import json

def create_gff_from_fasta_description(exon_dir_path : str) -> None:
    path_to_human_fasta = f"{exon_dir_path}/species_seqs/stripped/Homo_sapiens.fa"
    for i, record in enumerate(SeqIO.parse(path_to_human_fasta, "fasta")):
        assert i == 0, f"there was more than one seq in {path_to_human_fasta}"
        assert (x:= re.search("(\{.*\})", record.description)), f"no matching description found in '{record.description}' of file '{path_to_human_fasta}'"
        description = json.loads(x.group(1))
        # {'seq_start_in_genome_+_strand': 90231027,
        # 'seq_stop_in_genome_+_strand': 90231351,
        # 'exon_start_in_human_genome_+_strand': 90231094,
        # 'exon_stop_in_human_genome_+_strand': 90231213,
        # 'seq_start_in_genome_cd_strand': 90231351,
        # 'seq_stop_in_genome_cd_strand': 90231027,
        # 'exon_start_in_human_genome_cd_strand': 90231213,
        # 'exon_stop_in_human_genome_cd_strand': 90231094,
        # 'middle_exon': 1}
        on_reverse_strand = description["exon_start_in_human_genome_cd_strand"] != description["exon_start_in_human_genome_+_strand"]
        if not on_reverse_strand:
            exon_start = description["exon_start_in_human_genome_+_strand"] - description["seq_start_in_genome_+_strand"]
            exon_end = description["exon_stop_in_human_genome_+_strand"] - description["seq_start_in_genome_+_strand"]
        else:
            exon_start = description["seq_start_in_genome_cd_strand"] - description["exon_start_in_human_genome_cd_strand"] # checked this with picture and it seemes correct
            exon_end = description["seq_start_in_genome_cd_strand"] - description["exon_stop_in_human_genome_cd_strand"]

        def write_gff(name, start, stop,):
            with open(name, "w") as out:
                line = [record.id, "description", "CDS", str(start), str(stop), "0", "+", ".", "src=M"]
                out.write("\t".join(line))
                out.write("\n")

        out_file_path = f"{exon_dir_path}/human.gff"
        write_gff(out_file_path, exon_start, exon_end)
        hints_file_path = f"{exon_dir_path}/augustus_hints.gff"
        exon_len = exon_end - exon_start

        size_of_hints_cds = 11 # since min exon len is 24

        write_gff(hints_file_path, exon_start + exon_len//2 - size_of_hints_cds, exon_start + exon_len//2 + size_of_hints_cds)
        return hints_file_path

def run_augustus_for_dir(path : str) -> None:
    dir_content = sorted(list(os.listdir(path)))
    number_found = 0
    number_not_found = 0
    for i, file_or_dir in enumerate(dir_content):
        full_path = os.path.join(path, file_or_dir)
        if os.path.isdir(full_path):

            hintsfile = create_gff_from_fasta_description(full_path)
            command = f"{path_to_augustus_exe} --species=human --hintsfile={hintsfile} --strand=forward {full_path}/species_seqs/stripped/Homo_sapiens.fa"
            print("command", command)
            # stanke was zeigen wo es nicht funcktioniert
            # --minCoding_len 1

            # ich kanna auch ewzingen ddass erste und letzt position ein intron ist.

            try:
                output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
                # The captured output will be stored in the 'output' variable
                out_array = output.split("\n")
                non_comments = [line for line in out_array if (not line.startswith("#") and len(line) > 0)]
                if len(non_comments) == 0:
                    number_not_found += 1
                    print(f"nothing found for {full_path}")
                    continue
                non_comments = [line.split("\t") for line in non_comments]
                df = pandas.DataFrame(non_comments)
                print(i)
                df.columns = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
                print(full_path)
                print(df)
                number_found += 1
            except subprocess.CalledProcessError as e:
                print("Error executing command:", e.output)

            # exit_code = subprocess.call(re.split("\s+", command))#, stderr = err_handle, stdout = out_handle)
            # if exit_code != 0:
            #     print("exit_code:", exit_code)

    print(f"number_found {number_found}, number_not_found {number_not_found}")

if __name__ == "__main__":
    path = "/home/s-mamrzi/cgp_data/good_exons_2"
    path_to_augustus_exe ="/home/s-mamrzi/Augustus/bin/augustus"
    run_augustus_for_dir(path)