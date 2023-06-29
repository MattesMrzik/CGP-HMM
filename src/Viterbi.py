#!/usr/bin/env python3
import os
import pandas as pd
import json
import re
import time
from Bio import SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys

sys.path.insert(0, "../scripts")
from convert_kernels_to_matrices import convert_kernel_files_to_matrices_files


################################################################################
def retrieve_homo_sapiens_from_combinded_fasta(fasta_path):
        try:
            fasta_data = SeqIO.parse(fasta_path, "fasta")
            for record in fasta_data:
                if re.search("Homo_sapiens", record.id):
                    human_fasta = record
                    return human_fasta
            try:
                human_fasta.id
            except:
                print("fasta_true_state_seq_and_optional_viterbi_guess_alignment: no human id found: return")
                return
        except:
            print("seqIO could not parse", fasta_path)
            return
################################################################################
def read_viterbi_json(model, viterbi_path) -> list[tuple[int,str]]:
        l = []
        if viterbi_path != None:
            assert model != None, "if you want to include the viterbi guess, you also have to provide a model to convert the state seq to a string"

            try:
                file = open(viterbi_path)
            except:
                print("could not open", viterbi_path)
                return
            try:
                json_data = json.load(file)
            except:
                print("json could not parse", file)
                return

            assert len(json_data) == 1, f"more than one seq in {viterbi_path}"

            if type(json_data[0]) is list: #[[0,1,2],[0,0,1],[1,2,3,4,5]]
                description_seq = []
                for seq_id, seq in enumerate(json_data):
                    for nth_state, state in enumerate(seq):
                        description = model.state_id_to_str(state)
                        description_seq.append((state,description))
                    l.append(description_seq)
            # else: # [0,0,0,01,2,3,4,4,4,4]
            #     for nth_state, state in enumerate(json_data):
            #         description = model.state_id_to_str(state)
            #         l.append((state,description))
        return l
################################################################################
def viterbi_tuple_list_to_fasta(viterbi_path, human_fasta, l : list[int,str]) -> str:
    '''return st. like  llll---AGc 0c 1c 2c 3c 4c 5c 6c 7c 8c 9c10c11c12GT---rrrr'''
    viterbi_as_fasta = ""
    if viterbi_path == None:
        viterbi_as_fasta = " " * len(human_fasta.seq)
    else:
        i = 0
        while i < len(l[0]):
            state_id, description = l[0][i]
            if description == "left_intron":
                viterbi_as_fasta += "l"
            elif description == "right_intron":
                viterbi_as_fasta += "r"
            elif description == "A":
                viterbi_as_fasta += "A"
            elif description == "AG":
                viterbi_as_fasta += "G"
            elif description == "G":
                viterbi_as_fasta += "G"
            elif description == "GT":
                viterbi_as_fasta += "T"
            elif description.startswith("i_"):
                insert_id = re.search(r"i_(\d+),", description).group(1)
                if len(insert_id) > 2:
                    viterbi_as_fasta += "I" # this indicates that insert id is larger than 100, bc only the last two digits can be displayed
                else:
                    viterbi_as_fasta += "i"
                if l[0][i+1][1] == "G":
                    i-=2 # -2 bc adding 2 and 1 anyways
                elif l[0][i+2][1] == "G":
                    viterbi_as_fasta += "."
                    i-=1
                else:
                    viterbi_as_fasta += ("." + insert_id)[-2:]
                i+=2
            elif description.startswith("c_"):
                insert_id = re.search(r"c_(\d+),", description).group(1)
                if len(insert_id) > 2:
                    viterbi_as_fasta += "C" # this indicates that insert id is larger than 100, bc only the last two digits can be displayed
                else:
                    viterbi_as_fasta += "c"
                if l[0][i+1][1] == "G":
                    i-=2
                elif l[0][i+2][1] == "G":
                    viterbi_as_fasta += "."
                    i-=1
                else:
                    viterbi_as_fasta += ("." + insert_id)[-2:]
                i+=2
            else:
                viterbi_as_fasta += "-"

            i += 1

        # removing terminal symbol
        viterbi_as_fasta = viterbi_as_fasta[:-1]
        print("viterbi_as_fasta", viterbi_as_fasta)

        assert l[0][-1][1] == "ter", "Model.py last not terminal"
    return viterbi_as_fasta
################################################################################
def get_true_gene_strucutre_from_fasta_description_as_fasta(coords) -> tuple[str, bool, dict]:
    on_reverse_strand = coords["exon_start_in_human_genome_cd_strand"] != coords["exon_start_in_human_genome_+_strand"]
    if not on_reverse_strand:
        left_intron_len = (coords["exon_start_in_human_genome_+_strand"] - coords["seq_start_in_genome_+_strand"])
        exon_len = (coords["exon_stop_in_human_genome_+_strand"] - coords["exon_start_in_human_genome_+_strand"])
        right_intron_len = (coords["seq_stop_in_genome_+_strand"] - coords["exon_stop_in_human_genome_+_strand"])
    else:
        left_intron_len = (coords["seq_start_in_genome_cd_strand"] - coords["exon_start_in_human_genome_cd_strand"])
        exon_len = (coords["exon_start_in_human_genome_cd_strand"] - coords["exon_stop_in_human_genome_cd_strand"])
        right_intron_len = (coords["exon_stop_in_human_genome_cd_strand"] - coords["seq_stop_in_genome_cd_strand"])
    true_seq = "l" * left_intron_len
    true_seq += "E" * exon_len
    true_seq += "r" * right_intron_len
    d = {"left_intron_len": left_intron_len, \
         "exon_len" : exon_len, \
         "right_intron_len" : right_intron_len}
    return true_seq, on_reverse_strand, d
################################################################################
def get_coords_record(len_of_line_in_clw, coords, seq_len, on_reverse_strand):
    coords_fasta = ""
    for line_id in range(seq_len//len_of_line_in_clw):
        in_fasta = line_id*len_of_line_in_clw
        if not on_reverse_strand:
            coords_line = f"in this fasta {in_fasta}, in genome {in_fasta + coords['seq_start_in_genome_+_strand']}"
        else:
            coords_line = f"in this fasta {in_fasta}, in genome {coords['seq_start_in_genome_cd_strand']- in_fasta}"
        coords_fasta += coords_line + " " * (len_of_line_in_clw - len(coords_line))

    last_line_len = seq_len - len(coords_fasta)
    coords_fasta += " " * last_line_len

    coords_fasta_record = SeqRecord(seq = Seq(coords_fasta), id = "coords_fasta")
    return coords_fasta_record
################################################################################
def get_numerate_line_record(seq_len, len_of_line_in_clw = 50):
    numerate_line = ""
    for i in range(seq_len):
        i_line = i % len_of_line_in_clw
        if i_line % 10 == 0:
            numerate_line += "|"
        else:
            numerate_line += " "

    numerate_line_record = SeqRecord(seq = Seq(numerate_line), id = "numerate_line")
    return numerate_line_record
################################################################################
def fasta_true_state_seq_and_optional_viterbi_guess_alignment(fasta_path, viterbi_out_path = None, model = None, out_dir_path = "."):
    '''
    fasta_path is config.fasta_path (combined.fasta in most cases)
    assumes viterbi only contains prediction for human
    then writes an alignemnt of the human sequence
    and additional lines with coordinates to indicate where the exon is in the genome
    '''
    len_of_line_in_clw = 50

    if (human_fasta := retrieve_homo_sapiens_from_combinded_fasta(fasta_path)) is None:
        return

    if viterbi_out_path is not None:
        l = read_viterbi_json(model, viterbi_out_path)
        viterbi_as_fasta = viterbi_tuple_list_to_fasta(viterbi_out_path, human_fasta, l)

        viterbi_as_fasta_path = re.sub("_cc_output", "", viterbi_out_path)
        viterbi_as_fasta_path = re.sub(".json", ".fasta", viterbi_as_fasta_path)

        viterbi_record = SeqRecord(seq = Seq(viterbi_as_fasta), id = "viterbi_guess")
        with open(viterbi_as_fasta_path, "w") as viterbi_fa_file:
            SeqIO.write(viterbi_record, viterbi_fa_file, "fasta")

        # writing a gff file the the prediction with
        # coordinates of the exon with respect to the input fasta
        viterbi_as_gff_path = re.sub(".fasta", ".gff", viterbi_as_fasta_path)
        with open(viterbi_as_gff_path, "w") as file:
            start = viterbi_as_fasta.find("AG")
            end = viterbi_as_fasta.find("GT")

            if start != -1 and end != -1:
                file.write(human_fasta.id)
                file.write("\t")
                file.write("viterbi")
                file.write("\t")
                file.write("exon")
                file.write("\t")
                file.write(str(start + 1))
                file.write("\t")
                file.write(str(end + 2))
                file.write("\t")
                file.write(".")
                file.write("\t")
                file.write("+")
                file.write("\t")
                file.write(".")
                file.write("\t")
                file.write("ID=exon1;")
                file.write("\n")

    try:
        coords = json.loads(re.search("({.*})", human_fasta.description).group(1))
        coords["seq_start_in_genome_+_strand"]
        coords["exon_start_in_human_genome_cd_strand"]
        coords["exon_start_in_human_genome_+_strand"]
        coords["exon_stop_in_human_genome_+_strand"]
        coords["exon_stop_in_human_genome_cd_strand"]
        coords["seq_stop_in_genome_cd_strand"]
        coords["seq_stop_in_genome_+_strand"]
    except:
        print("description of fasta doenst contain the data necessary to build .clw file")
        return

    true_seq, on_reverse_strand, _ = get_true_gene_strucutre_from_fasta_description_as_fasta(coords)
    true_seq_record = SeqRecord(seq = Seq(true_seq), id = "true_seq")

    numerate_line_record = get_numerate_line_record(len(true_seq), len_of_line_in_clw = len_of_line_in_clw)

    coords_fasta_record = get_coords_record(len_of_line_in_clw, coords, len(true_seq), on_reverse_strand)

    exon_contains_ambiguous_bases = ""
    for base, e_or_i in zip(human_fasta.seq, true_seq_record.seq):
        if e_or_i == "E" and base in "acgtnN":
            exon_contains_ambiguous_bases = "_exon_contains_ambiguous_bases"

    if viterbi_out_path == None:
        records = [coords_fasta_record, numerate_line_record, human_fasta, true_seq_record]
    else:
        records = [coords_fasta_record, numerate_line_record, human_fasta, true_seq_record, viterbi_record]

    # print lens of seq of records
    for record in records:
        print(record.id, len(record.seq))

    if re.search("intron", fasta_path):
        for record in records:
            record.seq = record.seq[:len(viterbi_record.seq)]

    alignment = MultipleSeqAlignment(records)

    after_or_before = ""
    if viterbi_out_path != None:
        print(viterbi_out_path)

        found_after = re.search("after", os.path.basename(viterbi_out_path))
        found_before = re.search("before", os.path.basename(viterbi_out_path))

        assert not (found_after and found_before), "found both 'after' and 'before' in viterbi_cc_ file name"
        assert found_after or found_before, "found no 'after' or 'before' in viterbi_cc_ file name"

        after_or_before = "_after" if found_after else "_before"
        print(after_or_before)

    alignment_out_path = f"{out_dir_path}/true_alignment{after_or_before}{exon_contains_ambiguous_bases}.clw"

    with open(alignment_out_path, "w") as output_handle:
        AlignIO.write(alignment, output_handle, "clustal")
    print("wrote alignment to", alignment_out_path)

################################################################################
################################################################################
################################################################################
def run_cc_viterbi(config, matr_dir):
    start = time.perf_counter()
    seq_path = f"--seqs_path {config.fasta_path}.json"
    only_first_seq = f"--only_first_seq" if config.only_first_seq else ""

    after_or_before = os.path.basename(matr_dir).split("_")[0]
    assert after_or_before in ["before", "after"], "Viterbi.py. Directories of matrices must begin with either 'before' or 'after'"
    out_path = f"--out_path {config.current_run_dir}/viterbi_cc_output_{after_or_before}.json"


    command = f"{config.viterbi_exe} -j {config.viterbi_threads} {seq_path} {only_first_seq} {out_path} --dir_path_for_para {matr_dir}"
    print("starting", command)
    os.system(command)
    print("done viterbi. it took ", time.perf_counter() - start)
################################################################################
def main(config):
    after_or_before = "after_fit_para" if config.after_or_before == "a" else "before_fit_para"
    matr_dir = f"{config.current_run_dir}/{after_or_before}"

    print("start converting kernels to matrices")
    convert_kernel_files_to_matrices_files(config, matr_dir)
    print("done with converting kernels")

    seqs_json_path = f"{config.fasta_path}.json"
    print("fasta.json is calculated")
    from ReadData import read_data_one_hot_with_Ns_spread_str
    from ReadData import convert_data_one_hot_with_Ns_spread_str_to_numbers
    seqs = read_data_one_hot_with_Ns_spread_str(config, add_one_terminal_symbol = True)
    seqs_out = convert_data_one_hot_with_Ns_spread_str_to_numbers(seqs)
    with open(seqs_json_path, "w") as out_file:
        json.dump(seqs_out, out_file)
    print("finished calculating fasta.json")

    after_or_before = os.path.basename(matr_dir).split("_")[0]
    assert after_or_before in ["before", "after"], "Viterbi.py. Folder of matrices must begin with either 'before_' or 'after_'"

    # if viteribi is already calculated, path to viterbi file which is then written to the alignment
    if config.in_viterbi_path:
        assert config.manual_passed_fasta, "when viterbi.py and --in_viterbi_path also pass --fasta"
        fasta_true_state_seq_and_optional_viterbi_guess_alignment(config.fasta_path, config.in_viterbi_path, config.model, out_dir_path = config.current_run_dir)

    out_viterbi_file_path = f"{config.current_run_dir}/viterbi_cc_output_{after_or_before}.json"
    if not config.in_viterbi_path:
        run_cc_viterbi(config, matr_dir)

        fasta_true_state_seq_and_optional_viterbi_guess_alignment(config.fasta_path, out_viterbi_file_path, config.model, out_dir_path = config.current_run_dir)

    os.system(f"rm {seqs_json_path}")

if __name__ == "__main__":

    from Config import Config

    print("started making config in viterbi.py")
    config = Config()
    config.init_for_viterbi()
    print("done with making config in vertbi.py")

    main(config)

