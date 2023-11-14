#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import re

parser = argparse.ArgumentParser(description='Config module description')
parser.add_argument("-p", "--path_to_stats_table", help = "path_to stats table")
args = parser.parse_args()

pd.set_option('display.max_columns', None)
pd.options.display.width = 0

df = pd.read_csv(args.path_to_stats_table, sep = ";")

# some commands that are useful
print("pd.options.display.width = 0")
print('pd.set_option("display.max_rows", 10)')
print('df.sort_values(by = "exon_len", ascending = 1)')
print('df1 = df.groupby("ambiguous").apply(lambda x:x.sort_values("human_seq_len", ascending = 1))[:20]')

def copy_to_new_good_exons_dir(df) -> None:
    '''
    create df with desired exons in interactive mode and pass it to this function
    this will create a new dir in the same dir as where args.p points to
    '''
    dir_path = os.path.dirname(args.path_to_stats_table)
    print("dir_path", dir_path)
    highest_id_for_good_exon_dir = 0
    for subdir in os.listdir(dir_path):
        # check if the subdirectory is a directory
        sub_path = os.path.join(dir_path, subdir)
        if x := re.search(r"good_exons.{0,1}(\d+)", subdir):
            print("found prev good exon")
            highest_id_for_good_exon_dir = max(highest_id_for_good_exon_dir, int(x.group(1)))

    highest_id_for_good_exon_dir += 1

    new_good_exons_dir = f"{dir_path}/good_exons_{highest_id_for_good_exon_dir}"

    os.makedirs(new_good_exons_dir)
    for i, entry in df.iterrows():
        print("entry", entry)
        exon_name = os.path.basename(entry["path"])
        command = f"cp -r {dir_path}/{exon_name} {new_good_exons_dir}"
        print(command)
        os.system(command)

    # making the stats_table for the selected exons
    sub_stats_table_path = f"{new_good_exons_dir}/stats_table.csv"
    print("writing stats to", sub_stats_table_path)
    df.to_csv(sub_stats_table_path, index=True, header=True, line_terminator='\n', sep=";")

