#!/usr/bin/env python3
import pandas as pd
import os
import json
import time
import re
import numpy as np

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
def get_all_exons_df(hg38_refseq_bed : pd.DataFrame) -> dict[tuple[str,int,int, str], list[pd.Series]]:

    NM_df = hg38_refseq_bed[hg38_refseq_bed['name'].str.startswith('NM_')]
    # if there are less then 3 exons, there cant be a middle one
    # NM_df = NM_df[NM_df['blockCount'].apply(lambda x: x >= 3)]
    NM_df = NM_df[NM_df['chrom'].apply(lambda x: not re.search("_", x))]

    NM_df = NM_df.reset_index(drop=True)

    # all_exon_intervalls = {}
    # start = time.perf_counter()

    total_rows = len(NM_df)
    res = 50
    num_tenths = total_rows//res

    internal_exons = {} # key is chromosom, start and stop of exon in genome, value is list of rows that mapped to this exon range
    for i, (index, row) in enumerate(NM_df.iterrows()):
        # if index == 100:
        #     break
        if num_tenths != 0 and (i + 1) % num_tenths == 0:
            print(f"get_exons [{'#' *((i + 1) // num_tenths)}{' ' * (res - (i + 1) // num_tenths)}]", end = "\r")


        # for exon_id, (exon_len, exon_start) in enumerate(zip(row["blockSizes"][1:-1], row["blockStarts"][1:-1])):
        # this was under the assumption that the first exons contains ATG and the last one STOP
        # but it could be the case that the first or last exons are in UTR
        for exon_id, (exon_len, exon_start) in enumerate(zip(row["blockSizes"], row["blockStarts"])):
            exon_start_in_genome = row["chromStart"] + exon_start
            exon_end_in_genome = row["chromStart"] + exon_start + exon_len # the end id is not included
            chromosom = row["chrom"]
            # new_interval = (exon_start_in_genome, exon_end_in_genome)
            # if chromosom not in all_exon_intervalls:
            #     all_exon_intervalls[chromosom] = set([new_interval])
            # else:
            #      all_exon_intervalls[chromosom].add(new_interval)

            key = (chromosom, exon_start_in_genome, exon_end_in_genome, row["strand"])
            if key in internal_exons:
                internal_exons[key].append(dict(row))
            else:
                internal_exons[key] = [dict(row)]

    # print("finished get_internal_conding_exons(). It took:", time.perf_counter() - start)

    # print("len get_internal_conding_exons()", len(internal_exons))


    list_from_dict = [{"seq": exon_key[0], "start" : exon_key[1], "end" : exon_key[2], "strand" : exon_key[3], "bed_rows" : rows_list, "exon_key" : exon_key} for exon_key, rows_list in internal_exons.items()]
    df = pd.DataFrame(list_from_dict)

    print()
    return(df)
################################################################################
def choose_exon_of_all_its_duplicates_to_new_col(df):
    '''
    bc of alternative spliced genes, an exon might have multiple exons neigbhours to choose one.
    '''

    # exon1 has left closest neighbour, exon2 has right closest neighbour

    df["exon_row"] = None
    df["exon_id"] = None
    df["start_in_gene"] = None
    df["end_in_gene"] = None
    df["min_left_dist"] = None
    df["min_right_dist"] = None
    df["neighbours"] = None

    total_rows = len(df)
    res = 50
    num_tenths = total_rows//res

    for i,(index, row) in enumerate(df.iterrows()):
        exon_key = row["exon_key"]
        exon_bed_rows = row["bed_rows"]
    # for i, (exon_key, exon_bed_rows)in enumerate(exons.items()):
        if num_tenths != 0 and (i + 1) % num_tenths == 0:
            print(f"choose_exon_of_all_its_duplicates [{'#' *((i + 1) // num_tenths)}{' ' * (res - (i + 1) // num_tenths)}]", end = "\r")

        def get_dist_to_left_exon(exon_key, exon_row : pd.Series) -> int:
            try:
                exon_start_in_gene = exon_key[1] - exon_row["chromStart"]
                exon_id = exon_row["blockStarts"].index(exon_start_in_gene)
                assert exon_row["blockSizes"][exon_id] == exon_key[2] - exon_key[1], "calculated id exon len is not same as stop - start in genome"
                assert exon_id > 0, "exon_id is zero, which is oke, but not for dist to left exon"
                end_of_neighbouring_exon = exon_row["blockStarts"][exon_id - 1] + exon_row["blockSizes"][exon_id - 1]
                distance = exon_start_in_gene - end_of_neighbouring_exon
                return distance
            except:
                # print(f"exon key {exon_key}, row {exon_row}, return left inf")
                return float("inf")

        def get_dist_to_right_exon(exon_key, exon_row : pd.Series) -> int:
            try:
                exon_start_in_gene = exon_key[1] - exon_row["chromStart"]
                exon_id = exon_row["blockStarts"].index(exon_start_in_gene)
                assert exon_row["blockSizes"][exon_id] == exon_key[2] - exon_key[1], "calculated id exon len is not same as stop - start in genome"

                start_of_neighbouring_exon = exon_row["blockStarts"][exon_id + 1] + exon_row["blockSizes"][exon_id + 1]
                end_of_current_exon = exon_row["blockStarts"][exon_id] + exon_row["blockSizes"][exon_id]
                assert end_of_current_exon == exon_start_in_gene + exon_row["blockSizes"][exon_id]

                distance = start_of_neighbouring_exon - end_of_current_exon
                return distance
            except:
                # print(f"exon key {exon_key}, row {exon_row}, return right inf")
                return float("inf")
        # print()
        # since exon duplicates all have thier own row,
        # and for every row, there is a exon neighbour,
        # of which the distance can be calculated

        dists_to_left_neighbour =  [get_dist_to_left_exon(exon_key, exon_row)  for exon_row in exon_bed_rows]
        dists_to_right_neighbour = [get_dist_to_right_exon(exon_key, exon_row) for exon_row in exon_bed_rows]


        # print("dists_to_left_neighbour", dists_to_left_neighbour)
        # print("dists_to_right_neighbour", dists_to_right_neighbour)

        min_over_left_exons_of_dist_to_left_exon = min(dists_to_left_neighbour)
        min_over_right_exons_of_dist_to_right_exon = min(dists_to_right_neighbour)
        row["min_left_dist"] = min_over_left_exons_of_dist_to_left_exon
        row["min_right_dist"] = min_over_right_exons_of_dist_to_right_exon

        if min_over_left_exons_of_dist_to_left_exon == float("inf"):
            row["neighbours"] = "inf"
            df.iloc[index,:] = row
            continue
        if min_over_right_exons_of_dist_to_right_exon == float("inf"):
            row["neighbours"] = "inf"
            df.iloc[index,:] = row
            continue


        for j in range(len(exon_bed_rows)):
            if dists_to_left_neighbour[j] == min_over_left_exons_of_dist_to_left_exon and \
                dists_to_right_neighbour[j] == min_over_right_exons_of_dist_to_right_exon:
                # exons_with_out_duplicates.append((exon_key, exon_bed_rows[j]))
                row["exon_row"] = exon_bed_rows[j]
                start_in_gene =  exon_key[1] - row["exon_row"]["chromStart"]
                end_in_gene = exon_key[2] - row["exon_row"]["chromStart"]
                exon_id = row["exon_row"]["blockStarts"].index(start_in_gene)
                assert end_in_gene == row["exon_row"]["blockStarts"][exon_id] + row["exon_row"]["blockSizes"][exon_id],f'end_in_gene != row["exon_row"]["blockStarts"][exon_id] + row["exon_row"]["blockSizes"][exon_id]: {end_in_gene} == {row["exon_row"]["blockStarts"][exon_id]} + {row["exon_row"]["blockSizes"][exon_id]}'
                row["start_in_gene"] = start_in_gene
                row["end_in_gene"] = end_in_gene
                row["exon_id"] = exon_id
                row["neighbours"] = "opti"
                # print("found no pareto opti     ")
                # print(row)
                # print(df)
                break
        else:
            row["neighbours"] = "only_pareto"

        df.iloc[index,:] = row

    print()
    return df
################################################################################
def add_alternatively_spliced_flag_col(df):
    def two_different_spliced_froms(interval1, interval2) -> bool:
        if interval1[0] == interval2[0] and interval1[1] == interval2[1]:
            return False
        if interval1[0] <= interval2[1] and interval1[1] >= interval2[0]:
            return True
        else:
            return False
    total_rows = len(df)
    res = 50
    num_tenths = total_rows//res
    all_exon_intervalls = {}
    for i,(index, row) in enumerate(df.iterrows()):
        if num_tenths != 0 and (i + 1) % num_tenths == 0:
            print(f"alt spliced first pass [{'#' *((i + 1) // num_tenths)}{' ' * (res - (i + 1) // num_tenths)}]", end = "\r")
        seq = row["seq"]
        new_interval = (row["start"], row["end"])
        if seq not in all_exon_intervalls:
            all_exon_intervalls[seq] = set([new_interval])
        else:
                all_exon_intervalls[seq].add(new_interval)
    print()
    df["is_alt_spliced"] = False
    for i,(index, row) in enumerate(df.iterrows()):
        print(f"alt spliced second pass [{'#' *((i + 1) // num_tenths)}{' ' * (res - (i + 1) // num_tenths)}]", end = "\r")
        seq = row["seq"]
        key = row["exon_key"]
        for other_exon_interval in all_exon_intervalls[seq]:
            if two_different_spliced_froms((key[1], key[2]), other_exon_interval):
                # print((key[1], key[2]), other_exon_interval)
                row["is_alt_spliced"] = True
        df.iloc[index,:] = row
    # print("add_alternatively_spliced_flag")
    print()
    return df
################################################################################
def add_internal_thick_exon_flag_col(df):

    # getting exons that are actually between ATG and Stop
    # these might still include the exons with start and stop codon
    # these still need to be exluded in the filtering step
    # (i require exonstart > thickstart
    # there might not be the necessity to filter them further)
    total_rows = len(df)
    res = 50
    num_tenths = total_rows//res

    if "is_internal_and_thick" in df.columns:
        df.loc[:,"is_internal_and_thick"] = False
    else:
        df["is_internal_and_thick"] = False

    for i,(index, row) in enumerate(df.iterrows()):
        if num_tenths != 0 and (i + 1) % num_tenths == 0:
            print(f"add_internal_thick_exon_flag [{'#' *((i + 1) // num_tenths)}{' ' * (res - (i + 1) // num_tenths)}]", end = "\r")
        exon_row = row["exon_row"]
        if type(exon_row) is str:
            if exon_row in ["NaN","None"]:
                continue
            # print("exon_row", exon_row)
            exon_row = json.loads(re.sub("'",'"',exon_row))

        if exon_row is None or pd.isna(exon_row):
            continue

        if row["start"] > exon_row["thickStart"] and row["end"] < exon_row["thickEnd"] and row["exon_id"] != 0 and row["exon_id"] != exon_row["blockCount"]:
            row["is_internal_and_thick"] = True

        df.iloc[index,:] = row
    # print("add_internal_thick_exon_flag")
    print()
    return df
################################################################################
def add_len_cols(df :  list[tuple, pd.Series]) -> list[tuple, pd.Series]:
    total_rows = len(df)
    res = 50
    num_tenths = total_rows//res

    df["exon_len"] = df["end"] - df["start"]
    df["left_exon_len"] = None
    df["left_intron_len"] = None
    df["right_exon_len"] = None
    df["right_intron_len"] = None

    for i,(index, row) in enumerate(df.iterrows()):
        if num_tenths != 0 and (i + 1) % num_tenths == 0:
            print(f"filter_exons_based_in_min_segment_lengths [{'#' *((i + 1) // num_tenths)}{' ' * (res - (i + 1) // num_tenths)}]", end = "\r")
        exon_id = row["exon_id"]
        exon_row = row["exon_row"]
        if exon_row is None:
            continue
        row["left_exon_len"] = exon_row["blockSizes"][exon_id - 1]
        row["right_exon_len"] = exon_row["blockSizes"][exon_id + 1]
        row["left_intron_len"] = exon_row["blockStarts"][exon_id] - exon_row["blockStarts"][exon_id-1] - exon_row["blockSizes"][exon_id - 1]
        row["right_intron_len"] = exon_row["blockStarts"][exon_id + 1] - exon_row["blockStarts"][exon_id] - exon_row["blockSizes"][exon_id]

        df.iloc[index,:] = row
    print()
    return df
################################################################################
def add_col_whether_len_requirements_are_met(df, args):
    print("adding len_requirements_are_met col")
    df["len_req_met"] = (df["exon_len"] > args.min_exon_len) \
                    & (df["left_exon_len"] > args.min_left_neighbour_exon_len) \
                    & (df["right_exon_len"] > args.min_right_neighbour_exon_len) \
                    & (df["left_intron_len"] > args.min_left_neighbour_intron_len) \
                    & (df["right_intron_len"] > args.min_right_neighbour_intron_len)

    # print("add_col_whether_len_requirements_are_met")
    # print(df)
    return df
################################################################################
def add_ref_seqs_to_be_lifted_cols(df, args):
    '''
    calculates the infos necessary for liftover
    '''


    df["left_lift_start"] = None
    df["left_lift_end"] = None
    df["right_lift_start"] = None
    df["right_lift_end"] = None
    df["before_strip_seq_len"] = None

    total_rows = len(df)
    res = 50
    num_tenths = total_rows//res

    for i,(index, row) in enumerate(df.iterrows()):
        if num_tenths != 0 and (i + 1) % num_tenths == 0:
            print(f"get_ref_seqs_to_be_lifted [{'#' *((i + 1) // num_tenths)}{' ' * (res - (i + 1) // num_tenths)}]", end = "\r")

        if not row["len_req_met"] or row["len_req_met"] is None:
            continue

        exon_key = row["exon_key"]
        left_intron_len = row["left_intron_len"]
        right_intron_len = row["right_intron_len"]

        left_lift_start = row["start"] - left_intron_len - args.len_of_left_to_be_lifted
        left_lift_end = left_lift_start + args.len_of_left_to_be_lifted
        right_lift_start = row["end"] + right_intron_len
        right_lift_end = right_lift_start + args.len_of_right_to_be_lifted

        row["left_lift_start"] = left_lift_start
        row["left_lift_end"] = left_lift_end
        row["right_lift_start"] = right_lift_start
        row["right_lift_end"] = right_lift_end

        row["before_strip_seq_len"] = right_lift_start - left_lift_end

        df.iloc[index,:] = row


    df["exon_len_to_seq_len_ratio"] = df["exon_len"] / df["before_strip_seq_len"]
    return df
################################################################################
def load_or_calc_df(args, csv_path):
    '''
    gets the seqs_to_be_lifted
    either by calculating or loading it
    '''
    # seqs_to_be_lifted_file exists
    if os.path.exists(csv_path) :
        print(f"The df {csv_path} exists")
        if args.dont_ask_for_overwrite:
            should_get_calculated = False
        else:
            print("Do you want to overwrite it? [y/n] ", end = "")
            while True:
                x = input().strip()
                if x == "y":
                    should_get_calculated = True
                    break
                elif x == "n":
                    should_get_calculated = False
                    break
                else:
                    print("your answer must be either y or n")
    else:
        should_get_calculated = True

    print("should_get_calculated is hard coded to False")
    should_get_calculated = False

    if should_get_calculated:
        df = load_hg38_refseq_bed(args.hg38)
        df = get_all_exons_df(df)
        df = choose_exon_of_all_its_duplicates_to_new_col(df)
        df = add_alternatively_spliced_flag_col(df)
        df = add_internal_thick_exon_flag_col(df)
        df = add_len_cols(df)
        df = add_col_whether_len_requirements_are_met(df, args)
        df = add_ref_seqs_to_be_lifted_cols(df, args)
        df.to_csv(csv_path)

    else:
        print("loading df")
        df = pd.read_csv(csv_path)

    return df