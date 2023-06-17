#!/usr/bin/env python3
from Utility import append_time_ram_stamp_to_file


def main(config):
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from Training import fit_model
    import time
    import json
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

    with open(f"{config.current_run_dir}/history.log", "w") as file:
        file.write(json.dumps(history.history))



    plt.plot(history.history['loss'])
    plt.savefig(f"{config.current_run_dir}/loss.png")

    # write_matrices_after_fit:
    # start = time.perf_counter()
    # append_time_ram_stamp_to_file(f"main_programm after fit export paras start", config.bench_path, start)
    # dir_path = f"{config.current_run_dir}/after_fit_para"
    # if not os.path.exists(dir_path):
    #     os.system(f"mkdir -p {dir_path}")

    # if config.write_matrices_after_fit:
    #     I_kernel, A_kernel, B_kernel = model.get_weights()

    #     config.model.I_as_dense_to_json_file(f"{dir_path}/I.json", I_kernel)
    #     config.model.A_as_dense_to_json_file(f"{dir_path}/A.json", A_kernel)
    #     config.model.B_as_dense_to_json_file(f"{dir_path}/B.json", B_kernel)

    # # tf weights
    # path = f"{config.current_run_dir}/after_fit_para"

    # for i in range(config.epochs-1):
    #     try:
    #         model.get_layer(f"cgp_hmm_layer{'_' + str(i) if config.likelihood_influence_growth_factor else ''}").C.write_weights_to_file(path)
    #         break
    #     except:
    #         pass

    # append_time_ram_stamp_to_file(f"main_programm after fit export paras end", config.bench_path, start)


    # getting parameters diff before and after learning
    if config.calc_parameter_diff:
        start = time.perf_counter()
        append_time_ram_stamp_to_file(f"main_programm from_before_and_after_json_matrices_calc_diff_and_write_csv start", config.bench_path, start)
        if config.internal_exon_model:
            from Utility import from_before_and_after_json_matrices_calc_diff_and_write_csv
            from_before_and_after_json_matrices_calc_diff_and_write_csv(config)
        append_time_ram_stamp_to_file(f"main_programm from_before_and_after_json_matrices_calc_diff_and_write_csv end", config.bench_path, start)

    # export_to_dot_and_png2023-05-09_15-27_ycbm_chr1_1043537_1043732_65
    # if config.after_fit_png:
    #     start = time.perf_counter()
    #     append_time_ram_stamp_to_file(f"main_programm export after fit to dot start", config.bench_path, start)
    #     config.model.export_to_dot_and_png(A_kernel, B_kernel, name = "after_fit", to_png = config.nCodons < 10)
    #     append_time_ram_stamp_to_file(f"main_programm export after fit to dot end", config.bench_path, start)


    if config.viterbi:
        config.force_over_write = True
        # write convert fasta file to json (not one hot)
        # see make_dataset in Training.py
        from Viterbi import main
        config.only_first_seq = True
        config.force_overwrite = True
        config.after_or_before = "a"
        main(config)
        config.after_or_before = "b"
        main(config)

    print("main_programm_done")

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
