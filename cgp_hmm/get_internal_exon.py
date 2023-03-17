#!/usr/bin/env python3
import pandas as pd
import json
import argparse
import os
import time
import re
from Bio import SeqIO
import math

parser = argparse.ArgumentParser(description='Config module description')
parser.add_argument('--hg38', required = True, help = 'path to hg38-refseq.bed')
parser.add_argument('--hal', required = True, help = 'path to the .hal file')
parser.add_argument('--species', required = True, help = 'path to species file, which are the target of liftover from human')
parser.add_argument('--min_left_neighbour_exon_len', type = int, default = 20, help = 'min_left_neighbour_exon_len')
parser.add_argument('--min_left_neighbour_intron_len', type = int, default = 20, help = 'min_left_neighbour_intron_len')
parser.add_argument('--min_right_neighbour_exon_len', type = int, default = 20, help = 'min_right_neighbour_exon_len')
parser.add_argument('--min_right_neighbour_intron_len', type = int, default = 20, help = 'min_right_neighbour_intron_len')
parser.add_argument('--min_exon_len', type = int, default = 50, help = 'min_exon_len')
parser.add_argument('--len_of_exon_middle_to_be_lifted', type = int, default = 15, help = 'the middle of the exon is also lifted, to check whether it is between left and right if target .bed')
parser.add_argument('--len_of_left_to_be_lifted', type = int, default = 15, help = 'len_of_left_to_be_lifted')
parser.add_argument('--len_of_right_to_be_lifted', type = int, default = 15, help = 'len_of_right_to_be_lifted')
parser.add_argument('--path', default = ".", help = 'working directory')
parser.add_argument('-n', type = int, help = 'limit the number of exons to n')
parser.add_argument('-v', action = 'store_true', help = 'verbose')
parser.add_argument('--use_old_bed', action = 'store_true', help = 'use the old bed files and dont calculate new ones')
parser.add_argument('--use_old_fasta', action = 'store_true', help = 'use the old fasta files and dont calculate new ones')
args = parser.parse_args()

assert args.len_of_left_to_be_lifted < args.min_left_neighbour_exon_len, "len_of_left_to_be_lifted > min_left_neighbour_exon_len"
assert args.len_of_right_to_be_lifted < args.min_right_neighbour_exon_len, "len_of_right_to_be_lifted > min_right_neighbour_exon_len"
assert args.len_of_exon_middle_to_be_lifted < args.min_exon_len, "len_of_exon_middle_to_be_lifted > min_exon_len"

lengths_config_str = str(args.min_left_neighbour_exon_len)
lengths_config_str += "_" + str(args.len_of_left_to_be_lifted)
lengths_config_str += "_" + str(args.min_left_neighbour_intron_len)
lengths_config_str += "_" + str(args.min_exon_len)
lengths_config_str += "_" + str(args.len_of_exon_middle_to_be_lifted)
lengths_config_str += "_" + str(args.min_right_neighbour_intron_len)
lengths_config_str += "_" + str(args.len_of_right_to_be_lifted)
lengths_config_str += "_" + str(args.min_right_neighbour_exon_len)

# dirs
output_dir = f"{args.path}/out_{'' if not args.n else str(args.n) + 'Exons_'}{args.species.split('/')[-1]}_{lengths_config_str}_{args.hal.split('/')[-1]}"
if not os.path.exists(output_dir):
    os.system(f"mkdir -p {output_dir}")

# files
json_path = f"{output_dir}/filtered_internal_exons.json"


args.overwrite = False
if os.path.exists(json_path):
        print("There exists a file with previously exported filtered_internal_exons with same config")
        print("do you want to overwrite it? [y/n] ", end = "")
        while True:
            x = input().strip()
            if x == "y":
                args.overwrite = True
                break
            elif x == "n":
                args.overwrite = False
                break
            else:
                print("your answer must be either y or n")

def load_hg38_refseq_bed():
    start = time.perf_counter()
    print("started load_hg38_refseq_bed()")
    hg38_refseq_bed = pd.read_csv(args.hg38, delimiter = "\t", header = None)
    hg38_refseq_bed.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
    hg38_refseq_bed["blockSizes"] = hg38_refseq_bed["blockSizes"].apply(lambda s: [int(a) for a in s[:-1].split(",")])
    hg38_refseq_bed["blockStarts"] = hg38_refseq_bed["blockStarts"].apply(lambda s: [int(a) for a in s[:-1].split(",")])
    print("finished load_hg38_refseq_bed(). It took:", time.perf_counter() - start)
    return hg38_refseq_bed

def get_all_internal_exons(hg38_refseq_bed):
    start = time.perf_counter()
    print("started get_all_internal_exons()")
    internal_exons = {} # key is chromosom, start and stop of exon in genome, value is list of rows that mapped to this exon range
    for index, row in hg38_refseq_bed.iterrows():
        if args.n and index > args.n * 100: # cant just break at args.n since, exons arg filtered in an additional step. So i have to build some more here, such that after filtering sufficiently many remain.
            break
        if row["blockCount"] < 3:
            continue

        # print(row)
        for exon_id, (exon_len, exon_start) in enumerate(zip(row["blockSizes"][1:-1], row["blockStarts"][1:-1])):
            assert row["chromStart"] <= row["chromEnd"], 'row["chromStart"] <= row["chromEnd"]'
            assert row["thickStart"] <= row["thickEnd"], 'row["thickStart"] <= row["thickEnd"]'
            chromosom = row["chrom"]
            exon_start_in_genome = row["chromStart"] + exon_start
            exon_end_in_genome = row["chromStart"] + exon_start + exon_len # the end id is not included
            key = (chromosom, exon_start_in_genome, exon_end_in_genome)
            if key in internal_exons:
                internal_exons[key].append(row)
            else:
                internal_exons[key] = [row]
    if args.v:
        print("since the same exon occures in multiple genes, which may just be spliced differently")
        for i, key in enumerate(sorted(internal_exons.keys())):
            if len(internal_exons[key]) > 3:
                print(f"key is exon = {key}")
                print("value is list of df_rows:")
                for df_row in internal_exons[key]:
                    print(df_row)
                break
    print("finished get_all_internal_exons(). It took:", time.perf_counter() - start)
    return internal_exons


# minexon_len = float("inf")
# for i, key in enumerate(internal_exons.keys()):
#    for row in internal_exons[key]:
#        if  (x:= min(row["blockSizes"])) < minexon_len:
#            print(row)
#            minexon_len = x
#            print(minexon_len)
# print(minexon_len)

def filter_and_choose_exon_neighbours(all_internal_exons):
    start = time.perf_counter()
    print("started filter_and_choose_exon_neighbours()")
    filtered_internal_exons = []
    for i, key in enumerate(all_internal_exons.keys()):
        if args.n and len(filtered_internal_exons) >= args.n:
            break
        for row in all_internal_exons[key]:
            exon_start_in_gene = key[1] - row["chromStart"]
            exon_id = row["blockStarts"].index(exon_start_in_gene)
            assert row["blockSizes"][exon_id] == key[2] - key[1], "calculated id exon len is not same as stop - start in genome"
            if row["blockSizes"][exon_id] < args.min_exon_len:
                continue
            if row["blockSizes"][exon_id - 1] < args.min_left_neighbour_exon_len:
                continue
            left_intron_len = row["blockStarts"][exon_id] - row["blockStarts"][exon_id-1] - row["blockSizes"][exon_id - 1]
            if left_intron_len < args.min_left_neighbour_intron_len:
                continue
            right_intron_len = row["blockStarts"][exon_id + 1] - row["blockStarts"][exon_id] - row["blockSizes"][exon_id]
            if right_intron_len < args.min_right_neighbour_intron_len:
                continue
            if row["blockSizes"][exon_id + 1] < args.min_right_neighbour_exon_len:
                continue

            # getting coordinates from rightmost end of left exon that will be lifted to the other genomes
            # getting coordinates from leftmost end of right exon that will be lifted to the other genome
            left_lift_start = key[1] - left_intron_len - args.len_of_left_to_be_lifted
            left_lift_end = left_lift_start + args.len_of_left_to_be_lifted
            right_lift_start = key[2] + right_intron_len
            right_lift_end = right_lift_start + args.len_of_right_to_be_lifted

            di = {"seq" : key[0], \
                  "start_in_genome" : key[1], \
                  "stop_in_genome" : key[2], \
                  "exon_id" : exon_id, \
                  "left_lift_start": left_lift_start, \
                  "left_lift_end" : left_lift_end, \
                  "right_lift_start" : right_lift_start, \
                  "right_lift_end" : right_lift_end, \
                  "left_intron_len" : left_intron_len, \
                  "right_intron_len" : right_intron_len, \
                  "row" : dict(row)}
            filtered_internal_exons.append(di)
            break

    if args.v:
        for i in range(3):
            print()
            for key in sorted(filtered_internal_exons[i].keys()):
                print(key, filtered_internal_exons[i][key])
    print("finished filter_and_choose_exon_neighbours(). It took:", time.perf_counter() - start)
    return filtered_internal_exons

def write_filtered_internal_exons(filtered_internal_exons):
    start = time.perf_counter()
    print("started write_filtered_internal_exons()")
    with open(json_path, "w") as json_out:
        json.dump(filtered_internal_exons, json_out)
    print("finished write_filtered_internal_exons(). It took:", time.perf_counter() - start)

def get_to_be_lifted_exons(hg38_refseq_bed):
    if os.path.exists(json_path) and not args.overwrite:
        print(f"the file {json_path} exists, so it isnt calculated again")
        start = time.perf_counter()
        print("started json.load(filtered_internal_exons.json)")
        with open(json_path) as file:
            filtered_internal_exons = json.load(file)
        print("finished json.load(filtered_internal_exons.json). It took:", time.perf_counter() - start)
    else:
        all_internal_exons = get_all_internal_exons(hg38_refseq_bed)
        filtered_internal_exons = filter_and_choose_exon_neighbours(all_internal_exons)
        write_filtered_internal_exons(filtered_internal_exons)
    if args.n:
        filtered_internal_exons = filtered_internal_exons[:args.n]
    return filtered_internal_exons

hg38_refseq_bed = load_hg38_refseq_bed()
filtered_internal_exons = get_to_be_lifted_exons(hg38_refseq_bed)

def create_exon_data_sets(filtered_internal_exons):
    def get_all_species():
        with open(args.species, "r") as species_file:
            species = species_file.readlines()
        return [s.strip() for s in species]
    all_species = get_all_species()

    for exon in filtered_internal_exons:
        exon_dir = f"{output_dir}/exon_{exon['seq']}_{exon['start_in_genome']}_{exon['stop_in_genome']}"
        bed_output_dir = f"{exon_dir}/species_bed"
        seqs_dir = f"{exon_dir}/species_seqs"
        non_stripped_seqs_dir = f"{seqs_dir}/non_stripped"
        stripped_seqs_dir = f"{seqs_dir}/stripped"

        for d in [exon_dir, bed_output_dir, seqs_dir, non_stripped_seqs_dir, stripped_seqs_dir]:
            if not os.path.exists(d):
                os.system(f"mkdir -p {d}")


        human_exon_to_be_lifted_path = f"{exon_dir}/human_exons.bed"
        len_of_seq_substring_in_human = exon["right_lift_start"] - exon["left_lift_end"]

        # seq     start           stop            name    score   strand
        # chr1    67093589        67093604        left    0       -
        with open(human_exon_to_be_lifted_path, "w") as bed_file:
                for left_or_right in ["left","right"]:
                    bed_file.write(exon["seq"] + "\t")
                    bed_file.write(str(exon[f"{left_or_right}_lift_start"]) + "\t")
                    bed_file.write(str(exon[f"{left_or_right}_lift_end"]) + "\t")
                    bed_file.write(f"exon_{exon['start_in_genome']}_{exon['stop_in_genome']}_{exon['exon_id']}_{left_or_right}" + "\t")
                    bed_file.write("0" + "\t")
                    bed_file.write(exon["row"]["strand"] + "\n")
                bed_file.write(exon["seq"] + "\t")
                left_middle = (exon["stop_in_genome"] + exon['start_in_genome'] - args.len_of_exon_middle_to_be_lifted)//2
                right_middle = left_middle + args.len_of_exon_middle_to_be_lifted # this index does not part of the area to be lifted
                bed_file.write(str(left_middle) + "\t")
                bed_file.write(str(right_middle) + "\t")
                bed_file.write(f"exon_{exon['start_in_genome']}_{exon['stop_in_genome']}_{exon['exon_id']}_middle" + "\t")
                bed_file.write("0" + "\t")
                bed_file.write(exon["row"]["strand"] + "\n")

        for single_species in all_species:
            bed_file_path = f"{bed_output_dir}/{single_species}.bed"
            if not args.use_old_bed:
                command = f"time halLiftover {args.hal} Homo_sapiens {human_exon_to_be_lifted_path} {single_species} {bed_file_path}"
                print("running:", command)
                os.system(command)
            else:
                bed_files = [f for f in os.listdir(bed_output_dir) if f.endswith(".bed")]
                for bed_file in bed_files:
                    if bed_file.startswith(single_species):
                        bed_file_path = f"{bed_output_dir}/{bed_file}"
                        break # found an existing bed file
                else:
                    continue

            species_bed = pd.read_csv(bed_file_path, delimiter = "\t", header = None)
            species_bed.columns = ["seq", "start", "stop", "name", "score", "strand"]
            if len(species_bed.index) != 3:
                os.system(f"mv {bed_output_dir}/{single_species}.bed {bed_output_dir}/{single_species}_errorcode_more_than_3_lines.bed")
                continue
            l_m_r = {}
            for index, row in species_bed.iterrows():
                x = re.search(r"\d+_\d+_(.+)",row["name"])
                try:
                    l_m_r[x.group(1)] = row
                except:
                    print("l_m_r[x.group(1)] didnt work")
                    print("row['name']", row["name"])
                    exit()
            if len(l_m_r) != 3:
                os.system(f"mv {bed_output_dir}/{single_species}.bed {bed_output_dir}/{single_species}_errorcode_not_all_l_m_r.bed")
                continue

            if l_m_r["left"]["strand"] != l_m_r["right"]["strand"] or l_m_r["left"]["strand"] != l_m_r["middle"]["strand"]:
                os.system(f"mv {bed_output_dir}/{single_species}.bed {bed_output_dir}/{single_species}_errorcode_unequal_strands.bed")
                continue
            if l_m_r["left"]["seq"] != l_m_r["left"]["seq"] or l_m_r["left"]["seq"] != l_m_r["middle"]["seq"]:
                os.system(f"mv {bed_output_dir}/{single_species}.bed {bed_output_dir}/{single_species}_errorcode_unequal_seqs.bed")
                continue

            if exon["row"]["strand"] == l_m_r["left"]["strand"]:
                left_stop = l_m_r["left"]["stop"]
                middle_start = l_m_r["middle"]["start"]
                middle_end = l_m_r["middle"]["stop"]
                right_start = l_m_r["right"]["start"]
            else:
                left_stop = l_m_r["right"]["start"]
                middle_start = l_m_r["middle"]["stop"]
                middle_end = l_m_r["middle"]["start"]
                right_start = l_m_r["left"]["stop"]

            assert left_stop < middle_start, "left_stop > middle_start"
            assert middle_start < middle_stop, "middle_start > middle_stop"
            assert middle_stop < right_start, "middle_stop > right_start"

            # exit when this is in a different order of magnitude than len_of_seq_substring_in_human
            len_of_seq_substring_in_single_species = right_start - left_stop

            threshold = 1
            if abs(math.log10(len_of_seq_substring_in_single_species) - math.log10(len_of_seq_substring_in_human)) >= threshold:
                os.system(f"mv {bed_output_dir}/{single_species}.bed {bed_output_dir}/{single_species}_errorcode_lengths_differ_substantially .bed")
                continue

            # getting the seq, from humand: [left exon    [litfed]] [intron] [exon] [intron] [[lifted]right exon]
            # the corresponding seq of [intron] [exon] [intron] in other species
            out_fa_path = f"{non_stripped_seqs_dir}/{single_species}.fa"
            if not args.use_old_fasta:
                command = f"time hal2fasta {args.hal} {single_species} --start {left_stop} --length {len_of_seq_substring_in_single_species} --sequence {l_m_r['left']['seq']} --ucscSequenceNames --outFaPath {out_fa_path}"
                print("running:", command)
                os.system(command)
                os.system(f"head {out_fa_path}q	")

                # checking if fasta out only contains one seq
                # if strand is -, convert to reverse_complement
                for i, record in enumerate(SeqIO.parse(out_fa_path, "fasta")):
                    assert i == 0, f"found more than one seq in fasta file {out_fa_path}"
                    if l_m_r["left"]["strand"] == "-":
                        reverse_seq = record.seq.reverse_complement()
                        record.seq = reverse_seq
                        with open(out_fa_path, "w") as out_file:
                            SeqIO.write(record, out_file, "fasta")

            # strip seqs
            if os.path.exists(out_fa_path):
                for i, record in enumerate(SeqIO.parse(out_fa_path, "fasta")):
                    stripped_fa_path = re.sub("non_stripped","stripped",out_fa_path)
                    with open(stripped_fa_path, "w") as stripped_seq_file:
                        record.seq = record.seq[int(args.min_left_neighbour_exon_len /2) : - int(args.min_right_neighbour_exon_len/2)]
                        SeqIO.write(record, stripped_seq_file, "fasta")

        # gather all usable fasta seqs in a single file
        output_file =f"{exon_dir}/combined.fasta"
        input_files = [f"{stripped_seqs_dir}/{f}" for f in os.listdir(stripped_seqs_dir) if f.endswith(".fa")]

        with open(output_file, "w") as out:
            for input_file in input_files:
                for seq_record in SeqIO.parse(input_file, "fasta"):
                    SeqIO.write(seq_record, out, "fasta")


create_exon_data_sets(filtered_internal_exons)
# time halLiftover /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Homo_sapiens human_exon_to_be_lifted.bed Solenodon_paradoxus Solenodon_paradoxus.bed
# time hal2fasta /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Macaca_mulatta --start 66848804 --sequence CM002977.3 --length 15 --ucscSequenceNames > maxaxa_exon_left_seq.fa
