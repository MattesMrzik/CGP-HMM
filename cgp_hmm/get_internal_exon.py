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
parser.add_argument('--discard_multiple_bed_hits', action = 'store_true', help = 'sometimes, halLiftover maps a single coordinate to 2 or more, if this flag is passed, the species is discarded, otherwise the largest of the hits is selected')
args = parser.parse_args()

assert args.len_of_left_to_be_lifted < args.min_left_neighbour_exon_len, "len_of_left_to_be_lifted > min_left_neighbour_exon_len"
assert args.len_of_right_to_be_lifted < args.min_right_neighbour_exon_len, "len_of_right_to_be_lifted > min_right_neighbour_exon_len"
assert args.len_of_exon_middle_to_be_lifted < args.min_exon_len, "len_of_exon_middle_to_be_lifted > min_exon_len"
# TODO im using the above also for the start and end of th middle exon, not only the middle of the middle/current exon


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

def fasta_true_state_seq_and_optional_viterbi_guess_alignment(fasta_path, viterbi_path = None, out_dir_path = "."):
    # TODO: maybe also implement model.state_id_to_description_single_letter()

    # assumes viterbi only contains prediction for human

    import os
    from Bio import SeqIO, AlignIO
    from Bio.Align import MultipleSeqAlignment
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    import re

    try:
        fasta_data = SeqIO.parse(fasta_path, "fasta")
        for record in fasta_data:
            if re.search("Homo_sapiens", record.id):
                human_fasta = record
                # if nothing is found this will call except block
            try:
                human_fasta.id
            except:
                print("no human id found")
                return
    except:
        print("seqIO could not parse", fasta_path)
        return

    coords = json.loads(re.search("({.*})", human_fasta.description).group(1))

    l = []
    if viterbi_path != None:
        try:
            file = open(viterbi_path)
        except:
            print("could not open", file)
            return
        try:
            json_data = json.load(file)
        except:
            print("json could not parse", file)
            return

        if type(json_data[0]) is list: #[[0,1,2],[0,0,1],[1,2,3,4,5]]
            description_seq = []
            for seq_id, seq in enumerate(json_data):
                for nth_state, state in enumerate(seq):
                    description = self.state_id_to_str(state)
                    description_seq.append((state,description))
                l.append(description_seq)
        else: # [0,0,0,01,2,3,4,4,4,4]
            for nth_state, state in enumerate(json_data):
                description = self.state_id_to_str(state)
                l.append((state,description))


################################################################################
    viterbi_as_fasta = ""
    if viterbi_path == None:
        viterbi_as_fasta = " " * len(human_fasta.seq)
    else:
        for state_id, description in l[0]:
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
            else:
                viterbi_as_fasta += "-"

        # removing terminal
        viterbi_as_fasta = viterbi_as_fasta[:-1]
        assert l[0][-1][1] == "ter", "Model.py last not terminal"

    viterbi_record = SeqRecord(seq = Seq(viterbi_as_fasta), id = "viterbi_guess")
################################################################################
    on_reverse_strand = coords["exon_start_in_human_genome_cd_strand"] != coords["exon_start_in_human_genome_+_strand"]
    if not on_reverse_strand:
        true_seq = "l" * (coords["exon_start_in_human_genome_+_strand"] - coords["seq_start_in_genome_+_strand"])
        true_seq += "E" * (coords["exon_stop_in_human_genome_+_strand"] - coords["exon_start_in_human_genome_+_strand"])
        true_seq += "r" * (coords["seq_stop_in_genome_+_strand"] - coords["exon_stop_in_human_genome_+_strand"])
    else:
        true_seq = "l" * (coords["seq_start_in_genome_cd_strand"] - coords["exon_start_in_human_genome_cd_strand"])
        true_seq += "E" * (coords["exon_start_in_human_genome_cd_strand"] - coords["exon_stop_in_human_genome_cd_strand"])
        true_seq += "r" * (coords["exon_stop_in_human_genome_cd_strand"] - coords["seq_stop_in_genome_cd_strand"])
    true_seq_record = SeqRecord(seq = Seq(true_seq), id = "true_seq")
################################################################################
    len_of_line_in_clw = 50
    numerate_line = ""
    for i in range(len(viterbi_as_fasta)):
        i_line = i % len_of_line_in_clw
        if i_line % 10 == 0:
            numerate_line += "|"
        else:
            numerate_line += " "

    numerate_line_record =  SeqRecord(seq = Seq(numerate_line), id = "numerate_line")
################################################################################
    coords_fasta = ""
    for line_id in range(len(viterbi_as_fasta)//len_of_line_in_clw):
        in_fasta = line_id*len_of_line_in_clw
        if not on_reverse_strand:
            coords_line = f"in this fasta {in_fasta}, in genome {in_fasta + coords['seq_start_in_genome_+_strand']}"
        else:
            coords_line = f"in this fasta {in_fasta}, in genome {coords['seq_start_in_genome_cd_strand']- in_fasta}"
        coords_fasta += coords_line + " " * (len_of_line_in_clw - len(coords_line))

    last_line_len = len(viterbi_as_fasta) - len(coords_fasta)
    coords_fasta += " " * last_line_len

    coords_fasta_record = SeqRecord(seq = Seq(coords_fasta), id = "coords_fasta")

################################################################################
    if viterbi_path == None:
        records = [coords_fasta_record, numerate_line_record, human_fasta, true_seq_record]
    else:
        records = [coords_fasta_record, numerate_line_record, human_fasta, true_seq_record, viterbi_record]

    exon_contains_ambiguous_bases = ""
    for base, e_or_i in zip(human_fasta.seq, true_seq_record.seq):
        if e_or_i == "E" and base in "acgtnN":
            exon_contains_ambiguous_bases = "_exon_contains_ambiguous_bases"
    alignment = MultipleSeqAlignment(records)

    alignment_out_path = f"{os.path.dirname(viterbi_path)}/true_alignment{exon_contains_ambiguous_bases}.txt" if viterbi_path != None else f"{out_dir_path}/true_alignment{exon_contains_ambiguous_bases}.txt"
    with open(alignment_out_path, "w") as output_handle:
        AlignIO.write(alignment, output_handle, "clustal")
    print("wrote alignment to", alignment_out_path)

    return l
################################################################################
################################################################################
################################################################################
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
            def add_bed_line(seq = exon["seq"], start = None, stop = None, name = None, score = "0", strand = exon["row"]["strand"]):
                bed_file.write(seq + "\t")
                bed_file.write(start + "\t")
                bed_file.write(stop + "\t")
                bed_file.write(name + "\t")
                bed_file.write(score + "\t")
                bed_file.write(strand + "\n")

            base_name = f"exon_{exon['start_in_genome']}_{exon['stop_in_genome']}_{exon['exon_id']}"

            # left and right neighbouring exon
            for left_or_right in ["left","right"]:
                add_bed_line(start = str(exon[f"{left_or_right}_lift_start"]), \
                             stop = str(exon[f"{left_or_right}_lift_end"]), \
                             name = f"{base_name}_{left_or_right}")

            # middle of exon
            left_middle = (exon["stop_in_genome"] + exon['start_in_genome'] - args.len_of_exon_middle_to_be_lifted)//2
            right_middle = left_middle + args.len_of_exon_middle_to_be_lifted # this index does not part of the area to be lifted
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
            if len(species_bed.index) != 3 and args.discard_multiple_bed_hits:
                os.system(f"mv {bed_output_dir}/{single_species}.bed {bed_output_dir}/{single_species}_errorcode_more_than_3_lines.bed")
                continue
            l_m_r = {}
            for index, row in species_bed.iterrows():
                x = re.search(r"\d+_\d+_\d+_(.+)",row["name"])
                try:
                    if (y := x.group(1)) in l_m_r: # y = left, right or middle
                        len_of_previous_bed_hit = l_m_r[y]["stop"] - l_m_r[y]["start"]
                        len_of_current_bed_hit = row["stop"] - row["start"]
                        if len_of_current_bed_hit > len_of_previous_bed_hit:
                            l_m_r[y] = row
                    else:
                        l_m_r[y] = row
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
                right_start = l_m_r["right"]["start"]
            else:
                # i think        [left_start, left_stop] ... [middle_start, middle_stop] ... [right_start, right_stop]
                # gets mapped to [right_start, right_stop] ... [middle_start, middle_stop] ... [left_start, left_stop]
                left_stop = l_m_r["right"]["stop"]
                right_start = l_m_r["left"]["start"]
            middle_start = l_m_r["middle"]["start"]
            middle_stop = l_m_r["middle"]["stop"]

            if left_stop >= middle_start:
                os.system(f"mv {bed_output_dir}/{single_species}.bed {bed_output_dir}/{single_species}_errorcode_left_greater_middle.bed")
                continue
            if middle_stop >= right_start:
                os.system(f"mv {bed_output_dir}/{single_species}.bed {bed_output_dir}/{single_species}_errorcode_right_less_middle.bed")
                continue

            # exit when this is in a different order of magnitude than len_of_seq_substring_in_human
            len_of_seq_substring_in_single_species = right_start - left_stop

            threshold = 1
            if abs(math.log10(len_of_seq_substring_in_single_species) - math.log10(len_of_seq_substring_in_human)) >= threshold:
                os.system(f"mv {bed_output_dir}/{single_species}.bed {bed_output_dir}/{single_species}_errorcode_lengths_differ_substantially .bed")
                continue

            # getting the seq, from humand: [left exon    [litfed]] [intron] [exon] [intron] [[lifted]right exon]
            # the corresponding seq of [intron] [exon] [intron] in other species
            out_fa_path = f"{non_stripped_seqs_dir}/{single_species}.fa"
            on_reverse_strand = l_m_r["left"]["strand"] == "-"
            if not args.use_old_fasta:
                command = f"time hal2fasta {args.hal} {single_species} --start {left_stop} --length {len_of_seq_substring_in_single_species} --sequence {l_m_r['left']['seq']} --ucscSequenceNames --outFaPath {out_fa_path}"
                print("running:", command)
                os.system(command)
                os.system(f"head {out_fa_path}")

                # checking if fasta out only contains one seq
                # if strand is -, convert to reverse_complement
                for i, record in enumerate(SeqIO.parse(out_fa_path, "fasta")):
                    assert i == 0, f"found more than one seq in fasta file {out_fa_path}"

                    # write coordinates in genome to seq description
                    with open(out_fa_path, "w") as out_file:
                        start_in_genome = left_stop
                        stop_in_genome = left_stop + len_of_seq_substring_in_single_species
                        assert len(record.seq) == stop_in_genome - start_in_genome, "non stripped: actual seq len and calculated coordinate len differ"

                        # if exon is on - strand
                        # extracetd fasta is from + strand
                        # TAA -> gets converted to TTA
                        # these coords are from bed file and hg38:
                        # 1 start in genome +     2 exon start +      3 exon stop +      4 stop in genome +
                        # these are adjusted to the reversed and complemented fasta
                        # 4 start in genomce cd

                        description = {"seq_start_in_genome_+_strand" : start_in_genome, \
                                       "seq_stop_in_genome_+_strand" : stop_in_genome, \
                                       "exon_start_in_human_genome_+_strand": exon['start_in_genome'], \
                                       "exon_stop_in_human_genome_+_strand" : exon['stop_in_genome'], \
                                       "seq_start_in_genome_cd_strand" : start_in_genome if not on_reverse_strand else stop_in_genome, \
                                       "seq_stop_in_genome_cd_strand" : start_in_genome if on_reverse_strand else stop_in_genome, \
                                       "exon_start_in_human_genome_cd_strand" : exon["start_in_genome"] if not on_reverse_strand else exon["stop_in_genome"], \
                                       "exon_stop_in_human_genome_cd_strand" : exon["stop_in_genome"] if not on_reverse_strand else exon["start_in_genome"]}
                        record.description = json.dumps(description)
                        SeqIO.write(record, out_file, "fasta")

                    # reverse complement if on reverse strand
                    if on_reverse_strand:
                        reverse_seq = record.seq.reverse_complement()
                        record.seq = reverse_seq
                        with open(out_fa_path, "w") as out_file:
                            SeqIO.write(record, out_file, "fasta")

            # strip seqs
            if os.path.exists(out_fa_path):
                for i, record in enumerate(SeqIO.parse(out_fa_path, "fasta")):
                    stripped_fa_path = re.sub("non_stripped","stripped",out_fa_path)
                    with open(stripped_fa_path, "w") as stripped_seq_file:
                        left_strip_len = int(args.min_left_neighbour_exon_len /2)
                        right_strip_len = int(args.min_right_neighbour_exon_len/2)
                        record.seq = record.seq[left_strip_len : - right_strip_len]
                        start_in_genome = left_stop + left_strip_len
                        stop_in_genome = start_in_genome + len_of_seq_substring_in_single_species - left_strip_len - right_strip_len
                        assert stop_in_genome - start_in_genome == len(record.seq), "stripped: actual seq len and calculated coordinate len differ"
                        description = {"seq_start_in_genome_+_strand" : start_in_genome, \
                                       "seq_stop_in_genome_+_strand" : stop_in_genome, \
                                       "exon_start_in_human_genome_+_strand": exon['start_in_genome'], \
                                       "exon_stop_in_human_genome_+_strand" : exon['stop_in_genome'], \
                                       "seq_start_in_genome_cd_strand" : start_in_genome if not on_reverse_strand else stop_in_genome, \
                                       "seq_stop_in_genome_cd_strand" : start_in_genome if on_reverse_strand else stop_in_genome, \
                                       "exon_start_in_human_genome_cd_strand" : exon["start_in_genome"] if not on_reverse_strand else exon["stop_in_genome"], \
                                       "exon_stop_in_human_genome_cd_strand" : exon["stop_in_genome"] if not on_reverse_strand else exon["start_in_genome"]}
                        record.description = json.dumps(description)
                        record.description = json.dumps(description)
                        SeqIO.write(record, stripped_seq_file, "fasta")

            # create alignment of fasta and true splice sites
            fasta_true_state_seq_and_optional_viterbi_guess_alignment(stripped_fa_path, out_dir_path = exon_dir)



        # gather all usable fasta seqs in a single file
        output_file =f"{exon_dir}/combined.fasta"
        input_files = [f"{stripped_seqs_dir}/{f}" for f in os.listdir(stripped_seqs_dir) if f.endswith(".fa")]
        input_files = sorted(input_files, key = lambda x: 0 if re.search("Homo_sapiens", x) else 1)
        assert re.search("Homo_sapiens", input_files[0]), "homo sapiens not in first pos of combined.fasta"

        with open(output_file, "w") as out:
            for input_file in input_files:
                for seq_record in SeqIO.parse(input_file, "fasta"):
                    SeqIO.write(seq_record, out, "fasta")


create_exon_data_sets(filtered_internal_exons)
# time halLiftover /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Homo_sapiens human_exon_to_be_lifted.bed Solenodon_paradoxus Solenodon_paradoxus.bed
# time hal2fasta /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Macaca_mulatta --start 66848804 --sequence CM002977.3 --length 15 --ucscSequenceNames > maxaxa_exon_left_seq.fa
