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

################################################################################
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
def load_hg38_refseq_bed(load_hg38_refseq_bed_file_path : str) -> pd.DataFrame:
    start = time.perf_counter()
    print("started load_hg38_refseq_bed()")
    hg38_refseq_bed = pd.read_csv(load_hg38_refseq_bed_file_path, delimiter = "\t", header = None)
    hg38_refseq_bed.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
    hg38_refseq_bed["blockSizes"] = hg38_refseq_bed["blockSizes"].apply(lambda s: [int(a) for a in s[:-1].split(",")])
    hg38_refseq_bed["blockStarts"] = hg38_refseq_bed["blockStarts"].apply(lambda s: [int(a) for a in s[:-1].split(",")])
    print("finished load_hg38_refseq_bed(). It took:", time.perf_counter() - start)
    return hg38_refseq_bed
################################################################################
def get_internal_conding_exons(hg38_refseq_bed : pd.DataFrame) -> dict[tuple[str,int,int, str], pd.Series]:
    '''
    gets exons that have at least one neigbhouring exon upsteam and downstream
    these exons are also required to be between thick start and thick end
    so they dont include ATG or Stop

    returns:
        dict{
            key = (chr, start_in_genome, stop_in_genome) :
            value = row in RefSeq.bed
        }
    '''

    NM_df = hg38_refseq_bed[hg38_refseq_bed['name'].str.startswith('NM_')]
    # if there are less then 3 exons, there cant be a middle one
    NM_df = NM_df[NM_df['blockCount'].apply(lambda x: x >= 3)]
    NM_df = NM_df[NM_df['chrom'].apply(lambda x: not re.search("_", x))]

    NM_df = NM_df.reset_index(drop=True)

    all_exon_intervalls = {}
    start = time.perf_counter()
    print("started get_internal_conding_exons()")

    total_rows = len(NM_df)
    num_tenths = total_rows//10

    internal_exons = {} # key is chromosom, start and stop of exon in genome, value is list of rows that mapped to this exon range
    for index, row in NM_df.iterrows():
        # if index > 200:
        #     break
        if (index + 1) % num_tenths == 0:
            progress = (index + 1) // num_tenths
            print(f"{progress}0% of the get_internal_conding_exons loop completed")

        # for exon_id, (exon_len, exon_start) in enumerate(zip(row["blockSizes"][1:-1], row["blockStarts"][1:-1])):
        # this was under the assumption that the first exons contains ATG and the last one STOP
        # but it could be the case that the first or last exons are in UTR
        for exon_id, (exon_len, exon_start) in enumerate(zip(row["blockSizes"], row["blockStarts"])):
            exon_start_in_genome = row["chromStart"] + exon_start
            exon_end_in_genome = row["chromStart"] + exon_start + exon_len # the end id is not included

            all_exon_intervalls.add((exon_start_in_genome, exon_end_in_genome))
            # getting exons that are actually between ATG and Stop
            # these might still include the exons with start and stop codon
            # these still need to be exluded in the filtering step
            # (i require exonstart > thickstart
            # there might not be the necessity to filter them further)
            if row["chromStart"] + exon_start <= row["thickStart"]:
                # print("skipping exon bc exon_start <= thickStart with id", exon_id, "in", dict(row))
                continue

            if row["chromStart"] + exon_start + exon_len >= row["thickEnd"]:
                # print("skipping exon bc exon_end >= thickEnd with id", exon_id, "in", dict(row))
                continue

            assert row["chromStart"] <= row["chromEnd"], 'row["chromStart"] > row["chromEnd"]'
            assert row["thickStart"] <= row["thickEnd"], 'row["thickStart"] > row["thickEnd"]'
            chromosom = row["chrom"]
            key = (chromosom, exon_start_in_genome, exon_end_in_genome, row["strand"])
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

    print("finished get_internal_conding_exons(). It took:", time.perf_counter() - start)

    print("len get_internal_conding_exons()", len(internal_exons))

    def two_different_spliced_froms(interval1, interval2) -> bool:
        if interval1['start'] == interval2['start'] and interval1['stop'] == interval2['stop']:
            return False
        if interval1['start'] <= interval2['stop'] and interval1['stop'] >= interval2['start']:
            return True
        else:
            return False


    exons_keys_to_be_removed = []

    for key, row in internal_exons.items():
        for other_exon_interval in all_exon_intervalls:
            if two_different_spliced_froms((key[1], key[2]), other_exon_interval):
                exons_keys_to_be_removed.append(key)

    print("exons_keys_to_be_removed bc it is alternatively spliced:", exons_keys_to_be_removed)
    for key in exons_keys_to_be_removed:
        internal_exons.pop(key)


    return internal_exons
################################################################################
def choose_exon_of_all_its_duplicates(exons: dict[tuple, list[pd.Series]]) -> list[tuple, pd.Series]:
    '''
    bc of alternative spliced genes, an exon might have multiple exons neigbhours to choose one.
    '''
    start = time.perf_counter()
    print("started choose_exon_of_all_its_duplicates()")
    exons_with_out_duplicates = []

    # exon1 has left closest neighbour, exon2 has right closest neighbour

    total_rows = len(exons)
    num_tenths = total_rows//10
    for i, (exon_key, exon_rows)in enumerate(exons.items()):
        if (i + 1) % num_tenths == 0:
            progress = (i + 1) // num_tenths
            print(f"{progress}0% of the choose_exon_of_all_its_duplicates loop completed")

        def get_dist_to_left_exon(exon_key, exon_row : pd.Series) -> int:
            exon_start_in_gene = exon_key[1] - exon_row["chromStart"]
            exon_id = exon_row["blockStarts"].index(exon_start_in_gene)
            assert exon_row["blockSizes"][exon_id] == exon_key[2] - exon_key[1], "calculated id exon len is not same as stop - start in genome"

            end_of_neighbouring_exon = exon_row["blockStarts"][exon_id - 1] + exon_row["blockSizes"][exon_id - 1]
            dist = exon_start_in_gene - end_of_neighbouring_exon
            return dist

        def get_dist_to_right_exon(exon_key, exon_row : pd.Series) -> int:
            exon_start_in_gene = exon_key[1] - exon_row["chromStart"]
            exon_id = exon_row["blockStarts"].index(exon_start_in_gene)
            assert exon_row["blockSizes"][exon_id] == exon_key[2] - exon_key[1], "calculated id exon len is not same as stop - start in genome"

            start_of_neighbouring_exon = exon_row["blockStarts"][exon_id + 1] + exon_row["blockSizes"][exon_id + 1]
            end_of_current_exon = exon_row["blockStarts"][exon_id] + exon_row["blockSizes"][exon_id]
            assert end_of_current_exon == exon_start_in_gene + exon_row["blockSizes"][exon_id]

            dist = start_of_neighbouring_exon - end_of_current_exon
            return dist

        # since exon duplicates all have thier own row,
        # and for every row, there is a exon neighbour,
        # of which the distance can be calculated


        dists_to_left_neighbour =  [get_dist_to_left_exon(exon_key, exon_row)  for exon_row in exon_rows]
        dists_to_right_neighbour = [get_dist_to_right_exon(exon_key, exon_row) for exon_row in exon_rows]

        # print("dists_to_left_neighbour", dists_to_left_neighbour)
        # print("dists_to_right_neighbour", dists_to_right_neighbour)

        min_over_left_exons_of_dist_to_left_exon = min(dists_to_left_neighbour)
        min_over_right_exons_of_dist_to_right_exon = min(dists_to_right_neighbour)

        for j in range(len(exon_rows)):
            if dists_to_left_neighbour[j] == min_over_left_exons_of_dist_to_left_exon and \
                dists_to_right_neighbour[j] == min_over_right_exons_of_dist_to_right_exon:
                exons_with_out_duplicates.append((exon_key, exon_rows[j]))
                break
                # print(f"found optimal neighbours, exon id = {exon_key[4]}")
                # for row in exon_rows:
                #     print(exon_key[1] - row["chromStart"])
                #     print("row")
                #     print("starts", list(row["blockStarts"]))
                #     print("sizes", list(row["blockSizes"]))
                # exit()
        # else:
        #     print(f"found not optimal neighbours, exon id = {exon_key[4]}")
        #     for row in exon_rows:
        #         print(exon_key[1] - row["chromStart"])
        #         print("row")
        #         print("starts", list(row["blockStarts"]))
        #         print("sizes", list(row["blockSizes"]))
        #     break
    print("exons_with_out_duplicates size =", len(exons_with_out_duplicates))
    print("finished choose_exon_of_all_its_duplicates(). It took:", time.perf_counter() - start)
    return exons_with_out_duplicates
################################################################################
def filter_exons_based_in_min_segment_lengths(exons :  list[tuple, pd.Series]) -> list[tuple, pd.Series]:
    def my_filter(exon) -> bool:
        exon_key, exon_row = exon
        exon_start_in_gene = exon_key[1] - exon_row["chromStart"]
        exon_id = exon_row["blockStarts"].index(exon_start_in_gene)
        assert exon_row["blockSizes"][exon_id] == exon_key[2] - exon_key[1], "calculated id exon len is not same as stop - start in genome"

        if exon_row["blockSizes"][exon_id] < args.min_exon_len:
            return True

        if exon_row["blockSizes"][exon_id - 1] < args.min_left_neighbour_exon_len:
            return True

        left_intron_len = exon_row["blockStarts"][exon_id] - exon_row["blockStarts"][exon_id-1] - exon_row["blockSizes"][exon_id - 1]
        if left_intron_len < args.min_left_neighbour_intron_len:
            return True

        right_intron_len = exon_row["blockStarts"][exon_id + 1] - exon_row["blockStarts"][exon_id] - exon_row["blockSizes"][exon_id]
        if right_intron_len < args.min_right_neighbour_intron_len:
            return True

        if exon_row["blockSizes"][exon_id + 1] < args.min_right_neighbour_exon_len:
            return True

        return False

    start = time.perf_counter()
    print("started get_ref_seqs_to_be_lifted(): without progress info")
    exons = [exon for exon in exons if not my_filter(exon)]
    print("after filter_exons_based_in_min_segment_lengths", len(exons))
    print("finished get_ref_seqs_to_be_lifted(). It took:", time.perf_counter() - start)
    return exons


################################################################################
def get_ref_seqs_to_be_lifted(exons : list[tuple, pd.Series]) -> list[dict]:
    '''
    calculates the infos necessary for liftover
    '''

    start = time.perf_counter()
    print("started get_ref_seqs_to_be_lifted()")
    total_rows = len(exons)
    num_tenths = total_rows//10

    seqs_to_be_lifted = []
    for i, exon in enumerate(exons):
        if (i + 1) % num_tenths == 0:
            progress = (i + 1) // num_tenths
            print(f"{progress}0% of the get_ref_seqs_to_be_lifted loop completed")
        exon_key, exon_row = exon
        exon_start_in_gene = exon_key[1] - exon_row["chromStart"]
        exon_id = exon_row["blockStarts"].index(exon_start_in_gene)
        # getting coordinates from rightmost end of left exon that will be lifted to the other genomes
        # getting coordinates from leftmost end of right exon that will be lifted to the other genome
        left_intron_len = exon_row["blockStarts"][exon_id] - exon_row["blockStarts"][exon_id-1] - exon_row["blockSizes"][exon_id - 1]
        right_intron_len = exon_row["blockStarts"][exon_id + 1] - exon_row["blockStarts"][exon_id] - exon_row["blockSizes"][exon_id]

        left_lift_start = exon_key[1] - left_intron_len - args.len_of_left_to_be_lifted
        left_lift_end = left_lift_start + args.len_of_left_to_be_lifted
        right_lift_start = exon_key[2] + right_intron_len
        right_lift_end = right_lift_start + args.len_of_right_to_be_lifted

        seq_dict = {"seq" : exon_key[0], \
                "start_in_genome" : exon_key[1], \
                "stop_in_genome" : exon_key[2], \
                "exon_id" : exon_id, \
                "left_lift_start": left_lift_start, \
                "left_lift_end" : left_lift_end, \
                "right_lift_start" : right_lift_start, \
                "right_lift_end" : right_lift_end, \
                "left_intron_len" : left_intron_len, \
                "left_exon_len" : exon_row["blockSizes"][exon_id -1 ], \
                "right_intron_len" : right_intron_len, \
                "right_exon_len" : exon_row["blockSizes"][exon_id +1 ], \
                "key" : exon_key ,\
                "row" : dict(exon_row)}

        seqs_to_be_lifted.append(seq_dict)
    print("finished get_ref_seqs_to_be_lifted(). It took:", time.perf_counter() - start)
    return seqs_to_be_lifted

################################################################################
def write_seqs_to_be_lifted(seqs_to_be_lifted, path):
    start = time.perf_counter()
    print("started write_seqs_to_be_lifted()")
    with open(path, "w") as json_out:
        json.dump(seqs_to_be_lifted, json_out)
    print("finished write_seqs_to_be_lifted(). It took:", time.perf_counter() - start)

################################################################################
def get_seqs_to_be_lifted_df(seqs_to_be_lifted : list[dict]) -> pd.DataFrame:
    ''' keys are:
            "key" = exon_key = (seq, start, stop, strand, exon_id
            "row" = dict(["chrom", "chromStart", "chromEnd", "name", "score", "strand", "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"])
            "left_lift_end"
            "right_lift_start"
            "left_intron_len"
            "right_intron_len"

        i want to add cols:
            exon_len
            seq_len
            exon_len_to_seq_len_ratio

        i want to run this method to get a dict with data describing
        the human exons, which can be lifted over
        in interactive pyhton mode i want to select a subset of all exons
        and pass these to the further processing pipeline
        these should create a new data dir, with a csv of the selected exons
    '''

    start = time.perf_counter()
    print("started get_seqs_to_be_lifted_df()")

    total_rows = len(seqs_to_be_lifted)
    num_tenths = total_rows//10
    list_for_df = []
    # append new row to df with for loop
    for i, seqs_dict in enumerate(seqs_to_be_lifted):
        if (i + 1) % num_tenths == 0:
            progress = (i + 1) // num_tenths
            print(f"{progress}0% of the get_seqs_to_be_lifted_df loop completed")
        exon_len = seqs_dict["stop_in_genome"] - seqs_dict["start_in_genome"]
        before_strip_seq_len = seqs_dict["right_lift_start"] - seqs_dict["left_lift_end"]
        new_row_dict = {"seq" : seqs_dict["key"][0], \
                        "start" : seqs_dict["key"][1], \
                        "stop" : seqs_dict["key"][2], \
                        "strand" : seqs_dict["key"][3], \
                        "exon_len" :exon_len, \
                        "before_strip_seq_len" : before_strip_seq_len, \
                        "exon_len_to_seq_len_ratio" : exon_len / before_strip_seq_len, \
                        "left_intron_len" : seqs_dict["left_intron_len"], \
                        "left_exon_len" : seqs_dict["left_exon_len"], \
                        "right_intron_len" : seqs_dict["right_intron_len"], \
                        "right_exon_len" : seqs_dict["right_exon_len"]
                        }
        list_for_df.append(new_row_dict)
        # new_row = pd.DataFrame(new_row_dict, index=[0])
        # df = pd.concat([df, new_row], axis = 0, ignore_index = True)
    seqs_df = pd.DataFrame(list_for_df)

    print("finished get_seqs_to_be_lifted_df(). It took:", time.perf_counter() - start)
    return seqs_df
################################################################################
def get_to_be_lifted_seqs(args, json_path) -> list[dict]:
    '''
    gets the seqs_to_be_lifted
    either by calculating or loading it
    '''
    # seqs_to_be_lifted_file exists
    if os.path.exists(json_path) and not args.overwrite:
        print(f"the file {json_path} exists, so it isnt calculated again")
        start = time.perf_counter()
        print("started json.load(seqs_to_be_lifted.json)")

        with open(json_path) as file:
            seqs_to_be_lifted = json.load(file)
            seqs_to_be_lifted_df = get_seqs_to_be_lifted_df(seqs_to_be_lifted)
            unfiltered_seqs_to_be_lifted_df = seqs_to_be_lifted_df
            print("attention unfiltered_seqs_to_be_lifted_df is same as seqs_to_be_lifted_df if loaded and not calculated")
        print("finished json.load(seqs_to_be_lifted.json). It took:", time.perf_counter() - start)
    # seqs_to_be_lifted_file doesnt exist
    else:

        hg38_refseq_bed_df = load_hg38_refseq_bed(args.hg38)
        internal_exons = get_internal_conding_exons(hg38_refseq_bed_df)
        exons_with_out_duplicates = choose_exon_of_all_its_duplicates(internal_exons)
        filtered_exons = filter_exons_based_in_min_segment_lengths(exons_with_out_duplicates)
        seqs_to_be_lifted = get_ref_seqs_to_be_lifted(filtered_exons)
        write_seqs_to_be_lifted(seqs_to_be_lifted, json_path)
        seqs_to_be_lifted_df = get_seqs_to_be_lifted_df(seqs_to_be_lifted)

        unfiltered_seqs_to_be_lifted = get_ref_seqs_to_be_lifted(exons_with_out_duplicates)
        write_seqs_to_be_lifted(unfiltered_seqs_to_be_lifted, json_path + ".unfiltered")
        unfiltered_seqs_to_be_lifted_df = get_seqs_to_be_lifted_df(unfiltered_seqs_to_be_lifted)
    return seqs_to_be_lifted, seqs_to_be_lifted_df, unfiltered_seqs_to_be_lifted_df
################################################################################
################################################################################
################################################################################
def create_lift_over_query_bed_file(seq_dict : dict = None, \
                                    out_path : str = None) -> None:

    '''
    create the bed file of reference sequence that should be lifted over
    '''
    # seq     start           stop            name    score   strand
    # chr1    67093589        67093604        left    0       -
    with open(out_path, "w") as bed_file:
        def add_bed_line(start, stop, name , seq = seq_dict["seq"], score = "0", strand = seq_dict["row"]["strand"]):
            bed_file.write(seq + "\t")
            bed_file.write(start + "\t")
            bed_file.write(stop + "\t")
            bed_file.write(name + "\t")
            bed_file.write(score + "\t")
            bed_file.write(strand + "\n")

        base_name = f"exon_{seq_dict['start_in_genome']}_{seq_dict['stop_in_genome']}_{seq_dict['exon_id']}"

        # left and right neighbouring exon
        for left_or_right in ["left","right"]:
            add_bed_line(start = str(seq_dict[f"{left_or_right}_lift_start"]), \
                         stop = str(seq_dict[f"{left_or_right}_lift_end"]), \
                         name = f"{base_name}_{left_or_right}")

        # middle of exon
        left_middle = (seq_dict["stop_in_genome"] + seq_dict['start_in_genome'] - args.len_of_exon_middle_to_be_lifted)//2
        right_middle = left_middle + args.len_of_exon_middle_to_be_lifted # this index is not part of the area to be lifted
        add_bed_line(start = str(left_middle), \
                     stop = str(right_middle), \
                     name = f"{base_name}_middle")

        # start and stop of exon

        # add_bed_line(start = str(exon["start_in_genome"]), \
        #              stop = str(exon["start_in_genome"] + args.len_of_exon_middle_to_be_lifted), \
        #              name = f"{base_name}_exonstart")
        # add_bed_line(start = str(exon["stop_in_genome"] -  args.len_of_exon_middle_to_be_lifted), \
        #              stop = str(exon["stop_in_genome"]), \
        #              name = f"{base_name}_exonend")
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
def write_extra_data_to_fasta_description_and_reverse_complement(fa_path : str = None, \
                                                                 extra_seq_data : dict = None, \
                                                                 seq_dict : dict = None):
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
                           "exon_start_in_human_genome_+_strand": seq_dict['start_in_genome'], \
                           "exon_stop_in_human_genome_+_strand" : seq_dict['stop_in_genome'], \
                           "seq_start_in_genome_cd_strand" : extra_seq_data["seq_start_in_genome"] if not extra_seq_data["on_reverse_strand"] else extra_seq_data["seq_stop_in_genome"], \
                           "seq_stop_in_genome_cd_strand" : extra_seq_data["seq_start_in_genome"] if extra_seq_data["on_reverse_strand"] else extra_seq_data["seq_stop_in_genome"], \
                           "exon_start_in_human_genome_cd_strand" : seq_dict["start_in_genome"] if not extra_seq_data["on_reverse_strand"] else seq_dict["stop_in_genome"], \
                           "exon_stop_in_human_genome_cd_strand" : seq_dict["stop_in_genome"] if not extra_seq_data["on_reverse_strand"] else seq_dict["start_in_genome"], \
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
def strip_seqs(fasta_file = None, seq_dict = None, out_path = None, extra_seq_data = None):
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
                               "exon_start_in_human_genome_+_strand": seq_dict['start_in_genome'], \
                               "exon_stop_in_human_genome_+_strand" : seq_dict['stop_in_genome'], \
                               "seq_start_in_genome_cd_strand" : seq_start_in_genome if not extra_seq_data["on_reverse_strand"] else seq_stop_in_genome, \
                               "seq_stop_in_genome_cd_strand" : seq_start_in_genome if extra_seq_data["on_reverse_strand"] else seq_stop_in_genome, \
                               "exon_start_in_human_genome_cd_strand" : seq_dict["start_in_genome"] if not extra_seq_data["on_reverse_strand"] else seq_dict["stop_in_genome"], \
                               "exon_stop_in_human_genome_cd_strand" : seq_dict["stop_in_genome"] if not extra_seq_data["on_reverse_strand"] else seq_dict["start_in_genome"], \
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
# def write_human_gff_for_exon(seq_dict = None, path = None):
#     '''seq_dict = {"seq" : exon_key[0], \
#                 "start_in_genome" : exon_key[1], \
#                 "stop_in_genome" : exon_key[2], \
#                 "exon_id" : exon_id, \
#                 "left_lift_start": left_lift_start, \
#                 "left_lift_end" : left_lift_end, \
#                 "right_lift_start" : right_lift_start, \
#                 "right_lift_end" : right_lift_end, \
#                 "left_intron_len" : left_intron_len, \
#                 "left_exon_len" : exon_row["blockSizes"][exon_id -1 ], \
#                 "right_intron_len" : right_intron_len, \
#                 "right_exon_len" : exon_row["blockSizes"][exon_id +1 ], \
#                 "key" : exon_key ,\
#                 "row" : dict(exon_row)}
#     '''
#     # ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
#     seqname = seq_dict["seq"]
#     source = "hg38refseq"
#     feature = "exon"
#     strand = seq_dict["row"]["strand"]
#     if strand == "+":
#         start = seq_dict["row"]["blockStars"][seq_dict["exon_id"]]
#         end =   seq_dict["row"]["blockStarts"][seq_dict["exon_id"]] + seq_dict["row"]["blockSizes"][seq_dict["exon_id"]]
#     elif strand == "-":
#         start = seq_dict["row"]["blockStars"][seq_dict["exon_id"]]
#         end =   seq_dict["row"]["blockStarts"][seq_dict["exon_id"]] + seq_dict["row"]["blockSizes"][seq_dict["exon_id"]]
#     else:
#         print("strand is neither + or - ,   238t493dgzuh")
#         exit(1)



################################################################################
def create_exon_data_sets(args, seqs_to_be_lifted, output_dir) -> None:
    import sys
    sys.path.insert(0, "..")
    from Viterbi import fasta_true_state_seq_and_optional_viterbi_guess_alignment

    def get_all_species():
        with open(args.species, "r") as species_file:
            species = species_file.readlines()
        return [s.strip() for s in species]
    all_species = get_all_species()
    for seq_dict in seqs_to_be_lifted:
        print("seq_dict", seq_dict)
        exon_dir = f"{output_dir}/exon_{seq_dict['seq']}_{seq_dict['start_in_genome']}_{seq_dict['stop_in_genome']}"
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

        extra_exon_data["len_of_seq_substring_in_human"] = seq_dict["right_lift_start"] - seq_dict["left_lift_end"]
        extra_exon_data["human_strand"] = seq_dict["row"]["strand"]

        create_lift_over_query_bed_file(seq_dict = seq_dict, out_path = human_exon_to_be_lifted_path)


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
                                                    seq_dict = seq_dict)

            stripped_fasta_file_path = re.sub("non_stripped","stripped", out_fa_path)
            strip_seqs(fasta_file = out_fa_path, \
                       seq_dict = seq_dict, \
                       out_path = stripped_fasta_file_path, \
                       extra_seq_data = extra_seq_data)

            # create alignment of fasta and true splice sites
            if single_species == "Homo_sapiens":
                fasta_true_state_seq_and_optional_viterbi_guess_alignment(stripped_fasta_file_path, out_dir_path = exon_dir)

        # gather all usable fasta seqs in a single file
        input_files = get_input_files_with_human_at_0(from_path = stripped_seqs_dir)

        output_file =f"{exon_dir}/combined.fasta"

        combine_fasta_files(output_file = output_file, input_files = input_files )

        output_file = f"{capitalzed_subs_seqs_dir}/combined.fasta"
        if args.convert_short_acgt_to_ACGT > 0:
            convert_short_acgt_to_ACGT(output_file, input_files, threshold = args.convert_short_acgt_to_ACGT)
################################################################################
def make_stats_table(args):
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

# time halLiftover /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Homo_sapiens human_exon_to_be_lifted.bed Solenodon_paradoxus Solenodon_paradoxus.bed
# time hal2fasta /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Macaca_mulatta --start 66848804 --sequence CM002977.3 --length 15 --ucscSequenceNames > maxaxa_exon_left_seq.fa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python3 -i get_internal_exon.py --hal ../../../../CGP200/data/msa/241-mammalian-2020v2.hal --spe ../../../../CGP200/data/msa/species.names --hg ../../../iei-ranges/hg38-refseq.bed ')
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
    args = parser.parse_args()

    assert args.len_of_left_to_be_lifted <= args.min_left_neighbour_exon_len, "len_of_left_to_be_lifted > min_left_neighbour_exon_len"
    assert args.len_of_right_to_be_lifted <= args.min_right_neighbour_exon_len, "len_of_right_to_be_lifted > min_right_neighbour_exon_len"
    assert args.len_of_exon_middle_to_be_lifted <= args.min_exon_len, "len_of_exon_middle_to_be_lifted > min_exon_len"
    # TODO im using the above also for the start and end of th middle exon, not only the middle of the middle/current exon

    if args.stats_table:
        stats_df = make_stats_table(args)
        exit()


    assert args.hg38 and args.hal and args.species, "you must pass path to hg38, hal and species.lst"
    output_dir = get_output_dir()

    # files
    json_path = f"{output_dir}/seqs_to_be_lifted.json"
    print(f"writing to: {json_path}")
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path))

    args.overwrite = False
    if os.path.exists(json_path):
        print("There exists a file with previously exported seqs_to_be_lifted with same config")
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

    seqs_to_be_lifted, seqs_df, unfiltered_seqs_df = get_to_be_lifted_seqs(args, json_path)




    print("run this skript in interactive mode, select subset of seqs_df and pass it to make_data_from_df(args, seqs_df, seqs_to_be_lifted)")

def make_data_from_df(args, seqs_df, seqs_to_be_lifted):
    '''
    parameters:
        args: for .hal file
        seqs_df: was created by get_seqs_to_be_lifted_seqs_df, and should be passed to here, optionally sorted and subsetted
        seqs_to_be_lifted: seqs_to_be_lifted is list, indices from seqs_df are used to extract seqs from here

        seqs_df.sort_values("before_strip_seq_len")[:20000].sort_values("exon_len")[:7000].sample(40)
        plt.hist(seqs_df["exon_len"][seqs_df["exon_len"] < 600], bins = 20); plt.savefig("hist.png")
    '''
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M")
    out_dir_path = f"{output_dir}/{date_string}"
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    path_to_subset_df = f"{out_dir_path}/seqs_to_be_lifted_subset_df.csv"
    seqs_df.to_csv(path_to_subset_df, sep = ";", header = True)


    seqs_to_be_lifted = [seqs_to_be_lifted[i] for i in seqs_df.index.tolist()]
    create_exon_data_sets(args, seqs_to_be_lifted, out_dir_path)
    args.stats_table = out_dir_path
    make_stats_table(args)

def plot(column = "left_intron_len", \
         limit = 30000, \
         bins = 100,
         title = "intron length distribution", \
         xlabel = "intron length", \
         ylabel = "count" \
):
    plt.title(title)
    unfiltered_data = unfiltered_seqs_df[column][unfiltered_seqs_df[column] < limit]
    plt.hist(unfiltered_data, bins = bins, color = "tab:blue", label = "unfiltered")
    plt.hist(seqs_df[column][seqs_df[column] < limit], bins = bins, color = "tab:orange", label = "filtered", range = (min(unfiltered_data), max(unfiltered_data)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"{column}_{limit}.png")
    plt.savefig(f"hist.png")