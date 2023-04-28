#!/usr/bin/env python3
from Utility import append_time_ram_stamp_to_file


def main(config):
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from Training import fit_model
    import time
    import os
    import datetime

    if config.autograph_verbose:
        tf.autograph.set_verbosity(3, True)

    model, history = fit_model(config)
    # model.save("my_saved_model")

    # writng the loss history to file
    with open(f"{config.current_run_dir}/loss.log", "w") as file:
        for loss in history.history['loss']:
            file.write(str(loss) + " " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            file.write("\n")

    plt.plot(history.history['loss'])
    plt.savefig(f"{config.current_run_dir}/loss.png")

    I_kernel, A_kernel, B_kernel = model.get_weights()

    # write_matrices_after_fit:
    start = time.perf_counter()
    append_time_ram_stamp_to_file(f"main_programm after fit export paras start", config.bench_path, start)
    dir_path = f"{config.current_run_dir}/after_fit_para"
    if not os.path.exists(dir_path):
        os.system(f"mkdir -p {dir_path}")
    # in human readalbe format
    # config.model.A_as_dense_to_file(f"{dir_path}/A.csv", A_kernel, with_description = False)
    config.model.A_as_dense_to_file(f"{dir_path}/A.with_description.csv", A_kernel, with_description = True)
    # config.model.B_as_dense_to_file(f"{dir_path}/B.csv", B_kernel, with_description = False)
    config.model.B_as_dense_to_file(f"{dir_path}/B.with_description.csv", B_kernel, with_description = True)

    # json format
    config.model.I_as_dense_to_json_file(f"{dir_path}/I.json", I_kernel)
    config.model.A_as_dense_to_json_file(f"{dir_path}/A.json", A_kernel)
    config.model.B_as_dense_to_json_file(f"{dir_path}/B.json", B_kernel)

    # tf weights
    path = f"{config.current_run_dir}/after_fit_para"
    model.get_layer("cgp_hmm_layer").C.write_weights_to_file(path)

    append_time_ram_stamp_to_file(f"main_programm after fit export paras end", config.bench_path, start)


    # getting parameters diff before and after learning
    start = time.perf_counter()
    append_time_ram_stamp_to_file(f"main_programm export after fit to dot start", config.bench_path, start)
    if config.internal_exon_model:
        from Utility import from_before_and_after_json_matrices_calc_diff_and_write_csv
        from_before_and_after_json_matrices_calc_diff_and_write_csv(config)
    append_time_ram_stamp_to_file(f"main_programm export after fit to dot end", config.bench_path, start)

    # export_to_dot_and_png
    start = time.perf_counter()
    append_time_ram_stamp_to_file(f"main_programm export after fit to dot start", config.bench_path, start)
    config.model.export_to_dot_and_png(A_kernel, B_kernel, name = "after_fit", to_png = config.nCodons < 10)
    append_time_ram_stamp_to_file(f"main_programm export after fit to dot end", config.bench_path, start)


    if config.viterbi:
        # write convert fasta file to json (not one hot)
        # see make_dataset in Training.py
        import Viterbi
        Viterbi.main(config)


    config.write_passed_args_to_file()

    # this may lead to confusion
    # most_recent_call_dir = f"{config.current_run_dir}/../most_recent_call_dir"
    # if not os.path.exists(most_recent_call_dir):
    #     os.makedirs(most_recent_call_dir)
    # os.system(f"cp -r {config.current_run_dir}/* {most_recent_call_dir}")

if __name__ == '__main__':
    from Config import Config
    config = Config()
    config.init_for_training()
    main(config)
