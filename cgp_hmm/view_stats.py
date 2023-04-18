#!/usr/bin/python3
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Config module description')
parser.add_argument("-p", "--path_to_stats_table", help = "path_to stats table")
args = parser.parse_args()

pd.set_option('display.max_columns', None)
pd.options.display.width = 0
df = pd.read_csv(args.path_to_stats_table, sep = ";")
print("pd.options.display.width = 0")
print('pd.set_option("display.max_rows", 10)')
print('df.sort_values(by = "exon_len", ascending = 1)')
