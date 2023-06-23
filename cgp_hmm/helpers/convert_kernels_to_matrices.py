#!/usr/bin/env python3
import json
import numpy as np
def convert_kernel_files_to_matrices_files(config, dir_path):
    print("called convert_kernel_files_to_matrices_files. dir_path =", dir_path)

    if not config.model.is_made:
        config.model.make_model()

    # from cell.py
    def read_weights_from_file(kernel_dir):
            with open(f"{kernel_dir}/I_kernel.json") as file:
                I_kernel = np.array(json.load(file))
            with open(f"{kernel_dir}/A_kernel.json") as file:
                A_kernel = np.array(json.load(file))
            with open(f"{kernel_dir}/B_kernel.json") as file:
                B_kernel = np.array(json.load(file))
            return I_kernel, A_kernel, B_kernel

    try:
        I_kernel, A_kernel, B_kernel = read_weights_from_file(dir_path)
    except:
        print("a file not found in convert_kernel_files_to_matrices_files", dir_path)
        return

    print("nCodons =", config.nCodons)
    config.model.I_as_dense_to_json_file(f"{dir_path}/I.json", I_kernel)
    config.model.A_as_dense_to_json_file(f"{dir_path}/A.json", A_kernel)
    config.model.B_as_dense_to_json_file(f"{dir_path}/B.json", B_kernel)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..") # if called from helpers. prior path migth not work,
    # prior path
    sys.path.insert(0, ".") # if called from scr
    from Config import Config
    config = Config()
    config.init_for_convert_kernel()

    for after_or_before in ["after_fit_para", "before_fit_para"]:
        matr_dir = f"{config.current_run_dir}/{after_or_before}"
        convert_kernel_files_to_matrices_files(config, matr_dir)