#!/usr/bin/env python3
from datetime import datetime
import pandas as pd
import numpy as np
import json
import argparse
import os
import time
import re
from Bio import SeqIO
import math
import matplotlib.pyplot as plt

from get_exons_df import load_or_calc_df

def get_output_dir():
    lengths_config_str = str(args.min_left_neighbour_exon_len)
    lengths_config_str += "_" + str(args.len_of_left_to_be_lifted)
    lengths_config_str += "_" + str(args.min_left_neighbour_intron_len)
    lengths_config_str += "_" + str(args.min_exon_len)
    lengths_config_str += "_" + str(args.len_of_exon_middle_to_be_lifted)
    lengths_config_str += "_" + str(args.min_right_neighbour_intron_len)
    lengths_config_str += "_" + str(args.len_of_right_to_be_lifted)
    lengths_config_str += "_" + str(args.min_right_neighbour_exon_len)

    # dirs
    # output_dir = f"{args.path}/out_{'' if not args.n else str(args.n) + 'Exons_'}{args.species.split('/')[-1]}_{lengths_config_str}_{args.hal.split('/')[-1]}"
    output_dir = f"{args.path}/exons_{args.species.split('/')[-1]}_{lengths_config_str}_{args.hal.split('/')[-1]}"
    if not os.path.exists(output_dir):
        os.system(f"mkdir -p {output_dir}")
    return output_dir
################################################################################
################################################################################
################################################################################
def create_lift_over_query_bed_file(row : pd.Series = None, \
                                    out_path : str = None) -> None:

    '''
    create the bed file of reference sequence that should be lifted over
    '''
    # seq     start           stop            name    score   strand
    # chr1    67093589        67093604        left    0       -
    with open(out_path, "w") as bed_file:
        def add_bed_line(start, end, name , seq = row["seq"], score = "0", strand = row["strand"]):
            bed_file.write(seq + "\t")
            bed_file.write(start + "\t")
            bed_file.write(end + "\t")
            bed_file.write(name + "\t")
            bed_file.write(score + "\t")
            bed_file.write(strand + "\n")

        base_name = f"exon_{row['start']}_{row['end']}_{row['exon_id']}"

        # left and right neighbouring exon
        for left_or_right in ["left","right"]:
            add_bed_line(start = str(row[f"{left_or_right}_lift_start"]), \
                         end = str(row[f"{left_or_right}_lift_end"]), \
                         name = f"{base_name}_{left_or_right}")

        # middle of exon
        left_middle = (row["end"] + row['end'] - args.len_of_exon_middle_to_be_lifted)//2
        right_middle = left_middle + args.len_of_exon_middle_to_be_lifted # this index is not part of the area to be lifted
        add_bed_line(start = str(left_middle), \
                     end = str(right_middle), \
                     name = f"{base_name}_middle")
################################################################################
def run_liftover(args = None, \
                 human_exon_to_be_lifted_path : str = None, \
                 species_name : str = None, \
                 out_dir : str = None) -> bool:

    bed_file_path = f"{out_dir}/{species_name}.bed"
    # if not args.use_old_bed:
    command = f"time halLiftover {args.hal} Homo_sapiens {human_exon_to_be_lifted_path} {species_name} {bed_file_path}"
    print("running:", command)
    os.system(command)
    return True
    # else:
    #     bed_files = [f for f in os.listdir(out_dir) if f.endswith(".bed")]
    #     for bed_file in bed_files:
    #         # if bed_file.startswith(single_species):
    #         #     return f"{out_dir}/{bed_file}"
    #         if bed_file == f"{species_name}.bed":
    #             return True
################################################################################
def extract_info_and_check_bed_file(bed_dir_path : str = None, \
                                    species_name : str = None, \
                                    extra_seq_data : dict = None, \
                                    extra_exon_data : dict = None) -> bool:
    '''
    bed_dir_path: path to bed dir
    species_name: name of species
    extra_seq_data: is modified: dict for info about the seq (of a species) for the exon which is associated with this bed file
    extra_exon_data: is not modified, contains info about the human exon

    returns bool whether the file should be used for hal2fasta

    '''
    bed_file_path = f"{bed_dir_path}/{species_name}.bed"

    if os.path.getsize(bed_file_path) == 0:
        os.system(f"mv {bed_file_path} {bed_dir_path}/{species_name}_errorcode_empty.bed")
        return False

    species_bed_df = pd.read_csv(bed_file_path, delimiter = "\t", header = None)
    species_bed_df.columns = ["seq", "start", "stop", "name", "score", "strand"]

    species_bed_df["name"] = species_bed_df["name"].apply(lambda s: s.split("_")[-1])# only keep, left right or middle


    def swap_left_and_right(s):
        return ["left", "middle", "right"][["right","middle","left"].index(s)]

    # swap left and right if the lifted over seq is on a different strand than the human querey seq.bed
    species_bed_df["swapped_name"] = species_bed_df[["name","strand"]].apply(lambda s: swap_left_and_right(s["name"]) if s["strand"] != extra_exon_data["human_strand"] else s["name"], axis = 1)
    species_bed_df["start"] = species_bed_df["start"].astype(int)
    species_bed_df["stop"] = species_bed_df["stop"].astype(int)
    species_bed_df = species_bed_df.sort_values("start")


    # some coords are split bc of indels, i remove some rows that are close to each other
    # TODO i could also check if they are the same type (left, middle or right)
    rows_to_keep = []
    last_stop = -1
    for i, (index, row) in enumerate(species_bed_df.iterrows()):
        if last_stop == -1:
            last_stop = row["stop"]
            rows_to_keep.append(i)
            continue
        if abs(last_stop - row["start"]) > 7: # 7 is choosen arbitrarily
            rows_to_keep.append(i)
        last_stop = row["stop"]
    species_bed_df = species_bed_df.iloc[rows_to_keep,:]


    # finding left-right or left-middle-right pairs
    # that can be used for hal2fasta
    names_list = species_bed_df["swapped_name"].tolist()
    found_left_right_id = -1
    found_left_middle_right_id = -1
    for i in range(len(names_list) - 1):
        if i < len(names_list) - 2:
            if names_list[i:i+3] == ["left","middle","right"]:
                if len(species_bed_df.iloc[i:i+3,:]["seq"].unique()) == 1 \
                    and len(species_bed_df.iloc[i:i+3,:]["strand"].unique()) == 1:
                    # TODO also that they arent overlapping, which might be the case anyways
                    found_left_middle_right_id = i
                    break
        if names_list[i:i+2] == ["left","right"]:
            if len(species_bed_df.iloc[i:i+2,:]["seq"].unique()) == 1 \
                    and len(species_bed_df.iloc[i:i+2,:]["strand"].unique()) == 1:
                found_left_right_id = i

    if found_left_right_id == -1 and found_left_middle_right_id == -1:
        os.system(f"mv {bed_dir_path}/{species_name}.bed {bed_dir_path}/{species_name}_errorcode_no_pair_found.bed")
        return False
    if found_left_middle_right_id == -1:
        if found_left_right_id != -1:
            os.system(f"mv {bed_dir_path}/{species_name}.bed {bed_dir_path}/{species_name}_no_middle.bed")
            species_bed_df = species_bed_df.iloc[found_left_right_id:found_left_right_id+2,:]
            # not returning False, just noting info that middle is missing
    else:
        species_bed_df = species_bed_df.iloc[found_left_middle_right_id:found_left_middle_right_id+3,:]

    print(found_left_right_id, found_left_middle_right_id, species_bed_df)
    assert len(species_bed_df["seq"].unique()) == 1, f"unequal_seqs assertion error, {species_bed_df['seq'].unique()} {species_bed_df}"
    assert len(species_bed_df["strand"].unique()) == 1, f"unequal_strands assertion error, {species_bed_df['strand'].unique()} {species_bed_df}"

    print(species_bed_df[species_bed_df["swapped_name"] == "right"]["stop"].values[0])
    extra_seq_data["seq_start_in_genome"] = species_bed_df[species_bed_df["swapped_name"] == "left"]["stop"].values[0]
    extra_seq_data["seq_stop_in_genome"] =  species_bed_df[species_bed_df["swapped_name"] == "right"]["start"].values[0]

    if extra_seq_data["seq_start_in_genome"] > extra_seq_data["seq_stop_in_genome"]:
        os.system(f"mv {bed_dir_path}/{species_name}.bed {bed_dir_path}/{species_name}_errorcode_start_greater_stop.bed")
        return False


    extra_seq_data["middle_exon"] = found_left_middle_right_id != -1


    extra_seq_data["on_reverse_strand"] = species_bed_df["strand"].unique()[0] == "-"
    extra_seq_data["seq_name"] = species_bed_df["seq"].unique()[0]
    extra_seq_data["len_of_seq_substring_in_single_species"] = extra_seq_data["seq_stop_in_genome"] - extra_seq_data["seq_start_in_genome"]

    threshold = 1
    l1 = extra_seq_data["len_of_seq_substring_in_single_species"]
    l2 = extra_exon_data["len_of_seq_substring_in_human"]
    if abs(math.log10(l1) - math.log10(l2)) >= threshold:
        os.system(f"mv {bed_dir_path}/{species_name}.bed {bed_dir_path}/{species_name}_errorcode_lengths_differ_substantially.bed")
        return False

    return True
################################################################################
def run_hal_2_fasta(args = None, \
                    species_name : str = None, \
                    start : int = None, \
                    len : int = None, \
                    seq : str = None, \
                    outpath  : str = None):
    '''
    parameters:
        seq: e.g. chr1
    '''
    command = f"time hal2fasta {args.hal} {species_name} --start {start} --length {len} --sequence {seq} --ucscSequenceNames --outFaPath {outpath}"
    print("running:", command)
    os.system(command)
    os.system(f"head {outpath}")
################################################################################
def write_extra_data_to_fasta_description_and_reverse_complement(fa_path : str = None, \
                                                                 extra_seq_data : dict = None, \
                                                                 row : dict = None):
    for i, record in enumerate(SeqIO.parse(fa_path, "fasta")):
        assert i == 0, f"found more than one seq in fasta file {fa_path}"

        # write coordinates in genome to seq description
        with open(fa_path, "w") as out_file:
            assert len(record.seq) == extra_seq_data["seq_stop_in_genome"] - extra_seq_data["seq_start_in_genome"], "non stripped: actual seq len and calculated coordinate len differ"

            # if exon is on - strand
            # extracetd fasta is from + strand
            # TAA -> gets converted to TTA
            # these coords are from bed file and hg38:
            # 1 start in genome +     2 exon start +      3 exon stop +      4 stop in genome +
            # these are adjusted to the reversed and complemented fasta

            description = {"seq_start_in_genome_+_strand" : extra_seq_data["seq_start_in_genome"], \
                           "seq_stop_in_genome_+_strand" : extra_seq_data["seq_stop_in_genome"], \
                           "exon_start_in_human_genome_+_strand": row['start'], \
                           "exon_stop_in_human_genome_+_strand" : row['end'], \
                           "seq_start_in_genome_cd_strand" : extra_seq_data["seq_start_in_genome"] if not extra_seq_data["on_reverse_strand"] else extra_seq_data["seq_stop_in_genome"], \
                           "seq_stop_in_genome_cd_strand" : extra_seq_data["seq_start_in_genome"] if extra_seq_data["on_reverse_strand"] else extra_seq_data["seq_stop_in_genome"], \
                           "exon_start_in_human_genome_cd_strand" : row["start"] if not extra_seq_data["on_reverse_strand"] else row["end"], \
                           "exon_stop_in_human_genome_cd_strand" : row["end"] if not extra_seq_data["on_reverse_strand"] else row["start"], \
                           "middle_exon" : extra_seq_data["middle_exon"]}

            for key in description.keys():
                description[key] = int(description[key])

            print("description", description)
            record.description = json.dumps(description)
            if extra_seq_data["on_reverse_strand"]:
                reverse_seq = record.seq.reverse_complement()
                record.seq = reverse_seq
            SeqIO.write(record, out_file, "fasta")
################################################################################
def strip_seqs(fasta_file = None, row = None, out_path = None, extra_seq_data = None):
    '''
    args:
        fasta_file: path to fasta file
        seq_dict: info about seq that was lifted and hal2fastaed
    '''
    if os.path.exists(fasta_file):
        for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
            with open(out_path, "w") as stripped_seq_file:
                left_strip_len = int(args.min_left_neighbour_exon_len /2)
                right_strip_len = int(args.min_right_neighbour_exon_len/2)
                record.seq = record.seq[left_strip_len : - right_strip_len]
                seq_start_in_genome = extra_seq_data["seq_start_in_genome"] + left_strip_len
                seq_stop_in_genome = seq_start_in_genome + extra_seq_data["len_of_seq_substring_in_single_species"] - left_strip_len - right_strip_len
                assert seq_stop_in_genome - seq_start_in_genome == len(record.seq), "stripped: actual seq len and calculated coordinate len differ"
                description = {"seq_start_in_genome_+_strand" : seq_start_in_genome, \
                               "seq_stop_in_genome_+_strand" : seq_stop_in_genome, \
                               "exon_start_in_human_genome_+_strand": row['start'], \
                               "exon_stop_in_human_genome_+_strand" : row['end'], \
                               "seq_start_in_genome_cd_strand" : seq_start_in_genome if not extra_seq_data["on_reverse_strand"] else seq_stop_in_genome, \
                               "seq_stop_in_genome_cd_strand" : seq_start_in_genome if extra_seq_data["on_reverse_strand"] else seq_stop_in_genome, \
                               "exon_start_in_human_genome_cd_strand" : row["start"] if not extra_seq_data["on_reverse_strand"] else row["end"], \
                               "exon_stop_in_human_genome_cd_strand" : row["end"] if not extra_seq_data["on_reverse_strand"] else row["start"], \
                               "middle_exon" : extra_seq_data["middle_exon"]}

                for key in description.keys():
                    description[key] = int(description[key])

                record.description = json.dumps(description)
                SeqIO.write(record, stripped_seq_file, "fasta")
################################################################################
def convert_short_acgt_to_ACGT(outpath, input_files, threshold):
    def capitalize_lowercase_subseqs(seq, threshold_local):
        pattern = f"(?<![a-z])([a-z]{{1,{threshold_local}}})(?![a-z])"
        def repl(match):
            return match.group(1).upper()
        result = re.sub(pattern, repl, seq)
        return str(result)
    with open(outpath, "w") as output_handle:
        for input_file in input_files:
            with open(outpath, "a") as output_handle, open(input_file, "r") as in_file_handle:
                for i, record in enumerate(SeqIO.parse(in_file_handle, "fasta")):
                    assert i == 0, f"convert_short_acgt_to_ACGT found more than one seq in fasta file {input_file}"
                    new_seq = capitalize_lowercase_subseqs(str(record.seq), threshold)
                    output_handle.write(f">{record.id} {record.description}\n")
                    output_handle.write(f"{new_seq}\n")
                    # new_record = record.__class__(seq = "ACGT", id="homo", name="name", description="description")
                    # SeqIO.write(new_record, output_handle, "fasta")
                    # this produced
                    # File "/usr/lib/python3/dist-packages/Bio/File.py", line 72, in as_handle
                    # with open(handleish, mode, **kwargs) as fp:
################################################################################
def get_input_files_with_human_at_0(from_path = None):
    '''
    to combine fasta files to one with human at first position
    '''
    input_files = [f"{from_path}/{f}" for f in os.listdir(from_path) if f.endswith(".fa")]
    input_files = sorted(input_files, key = lambda x: (0 if re.search("Homo_sapiens", x) else 1, x))
    assert re.search("Homo_sapiens", input_files[0]), "homo sapiens not in first pos of combined.fasta"
    return input_files
################################################################################
def combine_fasta_files(output_file = None, input_files = None):
    with open(output_file, "w") as out:
        for input_file in input_files:
            for seq_record in SeqIO.parse(input_file, "fasta"):
                SeqIO.write(seq_record, out, "fasta")
        print("combined fasta files", output_file)
    out.close()
################################################################################
def calc_stats_table(args):
    stats_df = pd.DataFrame(columns = ["path", "exon", "exon_len", "human_seq_len", \
                                 "exon_len_to_human_len_ratio", "median_len", \
                                 "exon_len_to_median_len_ratio","average_len", \
                                 "exon_len_to_average_len", "num_seqs", "middle_ratio", \
                                 "ambiguous"])
    dir = args.stats_table
    for exon in os.listdir(dir):
        exon_dir = os.path.join(dir, exon)
        if os.path.isdir(exon_dir):
            # exon might be st like: exon_chr1_67095234_67095421
            exon_coords = list(map(int, exon.split("_")[-2:]))
            exon_len = exon_coords[1] - exon_coords[0]
            lens = []
            contains_middle_exon = 0
            for record in SeqIO.parse(f"{exon_dir}/combined.fasta","fasta"):
                lens.append(len(record.seq))
                description_json = json.loads(re.search("(\{.*\})", record.description).group(1))
                print("description_json", description_json)
                print("description_json[middle]", description_json["middle_exon"])
                contains_middle_exon += 1 if description_json["middle_exon"] else 0
                if record.id.startswith("Homo_sapiens"):
                    human_len = len(record.seq)

            median_len =  np.median(lens)
            average_len = np.average(lens)

            if os.path.exists(f"{exon_dir}/true_alignment_exon_contains_ambiguous_bases.clw"):
                ambiguous = 1
            elif os.path.exists(f"{exon_dir}/true_alignment.clw"):
                ambiguous = -1
            else:
                ambiguous = 0

            new_row_dict = {"path" : exon_dir, \
                            "exon" : exon, \
                            "exon_len" : exon_len, \
                            "human_seq_len" : human_len, \
                            "exon_len_to_human_len_ratio" : exon_len/human_len, \
                            "median_len" :median_len,\
                            "exon_len_to_median_len_ratio" : exon_len/median_len, \
                            "average_len" : average_len, \
                            "exon_len_to_average_len" : exon_len/average_len, \
                            "num_seqs" : len(lens),
                            "middle_ratio" : contains_middle_exon/ len(lens), \
                            "ambiguous" : ambiguous}

            stats_df.loc[len(stats_df)] = new_row_dict
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    stats_df.to_csv(f'{dir}/stats_table.csv', index=True, header=True, line_terminator='\n', sep=";")
    return stats_df
################################################################################
def get_all_species(args):
    with open(args.species, "r") as species_file:
        species = species_file.readlines()
    return [s.strip() for s in species]
################################################################################
def choose_subset_of_all_exons(df):
    df = df[df["len_req_met"]]
    df = df[df["is_internal_and_thick"]]
    df = df[~df["is_alt_spliced"]]

    return df
################################################################################
def make_exon_data_sets_for_choosen_df(choosen_exons_df) -> None:

    import sys
    sys.path.insert(0, "..")
    from Viterbi import fasta_true_state_seq_and_optional_viterbi_guess_alignment

    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M")
    out_dir_path = f"{output_dir}/{date_string}"
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    path_to_subset_df = f"{out_dir_path}/choosen_exons_df.csv"
    args.stats_table = out_dir_path

    choosen_exons_df.to_csv(path_to_subset_df, sep = ";", header = True)

    all_species = get_all_species(args)

    total_rows = len(choosen_exons_df)
    res = 50
    num_tenths = total_rows//res

    for i,(index, row) in enumerate(choosen_exons_df.iterrows()):
        if num_tenths != 0 and (i + 1) % num_tenths == 0:
            print(f"create_exon_data_sets_for_choosen_df [{'#' *((i + 1) // num_tenths)}{' ' * (res - (i + 1) // num_tenths)}]", end = "\r")

        exon_dir = f"{out_dir_path}/exon_{row['seq']}_{row['start']}_{row['end']}"
        bed_output_dir = f"{exon_dir}/species_bed"
        seqs_dir = f"{exon_dir}/species_seqs"
        non_stripped_seqs_dir = f"{seqs_dir}/non_stripped"
        stripped_seqs_dir = f"{seqs_dir}/stripped"
        capitalzed_subs_seqs_dir = f"{exon_dir}/combined_fast_capitalized_{args.convert_short_acgt_to_ACGT}"
        extra_exon_data = {}

        for d in [exon_dir, bed_output_dir, seqs_dir, non_stripped_seqs_dir, stripped_seqs_dir, capitalzed_subs_seqs_dir]:
            if not os.path.exists(d):
                os.system(f"mkdir -p {d}")

        human_exon_to_be_lifted_path = f"{exon_dir}/human.bed"

        extra_exon_data["len_of_seq_substring_in_human"] = row["right_lift_start"] - row["left_lift_end"]
        extra_exon_data["human_strand"] = row["strand"]

        create_lift_over_query_bed_file(row = row, out_path = human_exon_to_be_lifted_path)

        for single_species in all_species:
            extra_seq_data = {}

            if not run_liftover(
                args = args,
                human_exon_to_be_lifted_path = human_exon_to_be_lifted_path,
                species_name = single_species,
                out_dir = bed_output_dir
            ):
                continue

            if not extract_info_and_check_bed_file(
                bed_dir_path = bed_output_dir,
                species_name = single_species,
                extra_seq_data = extra_seq_data,
                extra_exon_data = extra_exon_data
            ):
                continue


            # getting the seq, from human: [left exon    [litfed]] [intron] [exon] [intron] [[lifted]right exon]
            # the corresponding seq of [intron] [exon] [intron] in other species
            out_fa_path = f"{non_stripped_seqs_dir}/{single_species}.fa"
            # if not args.use_old_fasta:
            run_hal_2_fasta(args = args,
                            species_name = single_species, \
                            start = extra_seq_data["seq_start_in_genome"], \
                            len = extra_seq_data["len_of_seq_substring_in_single_species"], \
                            seq = extra_seq_data["seq_name"], \
                            outpath = out_fa_path)

            write_extra_data_to_fasta_description_and_reverse_complement(fa_path = out_fa_path, \
                                                    extra_seq_data = extra_seq_data,
                                                    row = row)

            stripped_fasta_file_path = re.sub("non_stripped","stripped", out_fa_path)
            strip_seqs(fasta_file = out_fa_path, \
                       row = row, \
                       out_path = stripped_fasta_file_path, \
                       extra_seq_data = extra_seq_data)

            # create alignment of fasta and true splice sites
            if single_species == "Homo_sapiens":
                print("homo sap ", stripped_fasta_file_path, exon_dir)
                fasta_true_state_seq_and_optional_viterbi_guess_alignment(stripped_fasta_file_path, out_dir_path = exon_dir)

        # gather all usable fasta seqs in a single file
        input_files = get_input_files_with_human_at_0(from_path = stripped_seqs_dir)

        output_file =f"{exon_dir}/combined.fasta"

        combine_fasta_files(output_file = output_file, input_files = input_files )

        output_file = f"{capitalzed_subs_seqs_dir}/combined.fasta"
        if args.convert_short_acgt_to_ACGT > 0:
            convert_short_acgt_to_ACGT(output_file, input_files, threshold = args.convert_short_acgt_to_ACGT)

    calc_stats_table(args)
################################################################################
def plot(dfs,
         column = None,
         plot_table = False,
         limit = 30000, \
         bins = 100,
         title = None, \
         xlabel = "value", \
         ylabel = "count" \
):
    if title is None:
        if column is None:
            title = "title"
        else:
            title = column

    title += " " + "".join(np.random.choice([c for c in "qwerasdfyxcvbnhjlzup"], size = 5))
    print(title)
    plt.close()
    if plot_table:
        assert type(dfs) is not list,"when using plot_table dfs must the original df"
        local_df = dfs
        dfs = []
        g = local_df[local_df["is_internal_and_thick"]]
        dfs.append(g)
        mask = ~g["is_alt_spliced"]
        dfs.append(g[mask])

        # dfs.append(g[g["len_req_met"]])

        mask = (~g["is_alt_spliced"]) & (g["len_req_met"])
        dfs.append(g[mask])

        mask = (~g["is_alt_spliced"]) & (g["len_req_met"]) & (g["ieilen"]<5000)
        dfs.append(g[mask])

        # df1["ieilen"]=df1["exon_len"] + df1["left_intron_len"] + df1["right_intron_len"]

    if type(dfs) is not list:
        print("not list")
        dfs = [dfs]

    minimum = float("inf")
    maximum = float("-inf")
    for i, local_df in enumerate(dfs):
        plt.title(title)
        if column is not None:
            data = local_df[column][local_df[column] < limit]
        else:
            data = local_df[local_df < limit]
        minimum = min(min(data), minimum)
        maximum = max(max(data), maximum)

    for i, loop_df in enumerate(dfs):
        plt.title(title)
        if column is not None:
            data = loop_df[column][loop_df[column] < limit]
        else:
            data = loop_df[loop_df < limit]

        # tab:blue
        plt.hist(data, bins = bins, \
                 label = f"{i} len(data) {len(data)}", \
                 alpha = .5, \
                 range = (minimum, maximum))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    # plt.savefig(f"{re.sub('_','-',column)}-{limit}.png")
    plt.savefig(f"cache.png")
################################################################################
# time halLiftover /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Homo_sapiens human_exon_to_be_lifted.bed Solenodon_paradoxus Solenodon_paradoxus.bed
# time hal2fasta /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Macaca_mulatta --start 66848804 --sequence CM002977.3 --length 15 --ucscSequenceNames > maxaxa_exon_left_seq.fa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='example python3 -i get_internal_exon.py --hal ../../../../CGP200/data/msa/241-mammalian-2020v2.hal --spe ../../../../CGP200/data/msa/species.names --hg ../../../iei-ranges/hg38-refseq.bed ')
    parser.add_argument('--hg38', help = 'path to hg38-refseq.bed')
    parser.add_argument('--hal', help = 'path to the .hal file')
    parser.add_argument('--species', help = 'path to species file, which are the target of liftover from human')
    parser.add_argument('--min_left_neighbour_exon_len', type = int, default = 10, help = 'min_left_neighbour_exon_len')
    parser.add_argument('--min_left_neighbour_intron_len', type = int, default = 20, help = 'min_left_neighbour_intron_len')
    parser.add_argument('--min_right_neighbour_exon_len', type = int, default = 10, help = 'min_right_neighbour_exon_len')
    parser.add_argument('--min_right_neighbour_intron_len', type = int, default = 20, help = 'min_right_neighbour_intron_len')
    parser.add_argument('--min_exon_len', type = int, default = 10, help = 'min_exon_len')
    parser.add_argument('--len_of_exon_middle_to_be_lifted', type = int, default = 10, help = 'the middle of the exon is also lifted, to check whether it is between left and right if target .bed')
    parser.add_argument('--len_of_left_to_be_lifted', type = int, default = 10, help = 'len_of_left_to_be_lifted')
    parser.add_argument('--len_of_right_to_be_lifted', type = int, default = 10, help = 'len_of_right_to_be_lifted')
    parser.add_argument('--path', default = "../../../cgp_data", help = 'working directory')
    parser.add_argument('-v', action = 'store_true', help = 'verbose')
    # parser.add_argument('--use_old_bed', action = 'store_true', help = 'use the old bed files and dont calculate new ones')
    # parser.add_argument('--use_old_fasta', action = 'store_true', help = 'use the old fasta files and dont calculate new ones')
    parser.add_argument('--discard_multiple_bed_hits', action = 'store_true', help = 'sometimes, halLiftover maps a single coordinate to 2 or more, if this flag is passed, the species is discarded, otherwise the largest of the hits is selected')
    # parser.add_argument('--stats_table', nargs = '?', const = True, help ='instead of getting all the exon data, get stats table of existing data. Specified path, or pass hg38, hal and species and same n')
    parser.add_argument('--stats_table', help ='path to dir to make the stats_table')
    parser.add_argument('--convert_short_acgt_to_ACGT', type = int, default = 0, help = 'convert shorter than --convert_short_acgt_to_ACGT')
    parser.add_argument('--dont_ask_for_overwrite', action = 'store_true', help = 'if this is passed you are not prompted to answer if the all exons df csv should be overwritten if it exists')
    args = parser.parse_args()

    assert args.len_of_left_to_be_lifted <= args.min_left_neighbour_exon_len, "len_of_left_to_be_lifted > min_left_neighbour_exon_len"
    assert args.len_of_right_to_be_lifted <= args.min_right_neighbour_exon_len, "len_of_right_to_be_lifted > min_right_neighbour_exon_len"
    assert args.len_of_exon_middle_to_be_lifted <= args.min_exon_len, "len_of_exon_middle_to_be_lifted > min_exon_len"

    if args.stats_table:
        stats_df = calc_stats_table(args)
        exit()

    assert args.hg38 and args.hal and args.species, "you must pass path to hg38, hal and species.lst"

    # preparing dirs
    output_dir = get_output_dir()
    csv_path = f"{output_dir}/exons_df.csv"
    print(f"parent dir = {csv_path}")
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))

    df = load_or_calc_df(args, csv_path)

    print("run this skript in interactive mode, select subset of df and pass it to make_data_from_df(args, df)")

