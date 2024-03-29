#!/usr/bin/env python3

# If only json files exist, which are created from Viterbi.py,
# then call add_state_and_emmision_str_to_matrices first.
# Then call this script

import os
import pandas as pd

from add_state_and_emission_str_to_matrices import call_for_every_A_and_B_found_in_subdirs
def diff_of_csv_with_description(before_path, after_path, output_path):

    # Read in the two input CSV files
    df1 = pd.read_csv(before_path, index_col = 0, sep = ";")
    df2 = pd.read_csv(after_path, index_col = 0, sep = ";")

    # Compute the difference between the two dataframes
    df_diff = df1 - df2
    df_diff[df1.isin([0, 1])] = ""

    # Write the output to a new CSV file
    df_diff.to_csv(output_path)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../src")
    from Config import Config
    config = Config()
    config.init_for_matrix_diff()

    for transition_or_emissions in "AB":
        before = f"{config.parent_input_dir}/before_fit_para/{transition_or_emissions}_with_description.csv"
        after = f"{config.parent_input_dir}/after_fit_para/{transition_or_emissions}_with_description.csv"

        # if not os.path.exists(before):
        #     convert_kernel_files_to_matrices_files(config, os.path.dirname(before))
        # if not os.path.exists(after):
        #     convert_kernel_files_to_matrices_files(config, os.path.dirname(after))

        if not os.path.exists(before):
            call_for_every_A_and_B_found_in_subdirs(config.model, config.parent_input_dir)

        out_dir_path = f"{config.parent_input_dir}/diff_para"
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)
        out_file_path = f"{out_dir_path}/{transition_or_emissions}_diff.csv"
        diff_of_csv_with_description(before, after, out_file_path)
