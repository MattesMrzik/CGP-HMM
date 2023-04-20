#!/usr/bin/env python3
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
    print("done fit_model()")
    # model.save("my_saved_model")

    # writng the loss history to file
    with open(f"{config.current_run_dir}/loss.log", "w") as file:
        for loss in history.history['loss']:
            file.write(str(loss) + " " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            file.write("\n")

    plt.plot(history.history['loss'])
    plt.savefig(f"{config.current_run_dir}/loss.png")

    I_kernel, A_kernel, B_kernel = model.get_weights()

    # if config.write_matrices_after_fit:

    start = time.perf_counter()
    print("starting to write model")

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
    
    path = f"{config.current_run_dir}/after_fit_para"
    model.get_layer("cgp_hmm_layer").C.write_weights_to_file(path)

    print("done write model. it took ", time.perf_counter() - start)

    if config.internal_exon_model:
        from Utility import from_before_and_after_json_matrices_calc_diff_and_write_csv
        from_before_and_after_json_matrices_calc_diff_and_write_csv(config)

    # if config.write_parameters_after_fit:

    config.model.export_to_dot_and_png(A_kernel, B_kernel, name = "after_fit", to_png = config.nCodons < 10)


    if config.viterbi:
        # write convert fasta file to json (not one hot)
        # see make_dataset in Training.py

        import Viterbi

        Viterbi.run_cc_viterbi(config)
        if config.manual_passed_fasta:
            print("you passed --manual_passed_fasta so viterbi guess isnt checked against generated seqs")
        if not config.manual_passed_fasta:
            viterbi_guess = Viterbi.load_viterbi_guess(config)

            true_state_seqs = Viterbi.get_true_state_seqs_from_true_MSA(config)

            Viterbi.compare_guess_to_true_state_seq(true_state_seqs, viterbi_guess)

            Viterbi.write_viterbi_guess_to_true_MSA(config, true_state_seqs, viterbi_guess)

            Viterbi.eval_start_stop(config, viterbi_guess)

    

    config.write_all_attributes_to_file()

    most_recent_call_dir = f"{config.current_run_dir}/../most_recent_call_dir"
    if not os.path.exists(most_recent_call_dir):
        os.makedirs(most_recent_call_dir)
    os.system(f"cp -r {config.current_run_dir} {most_recent_call_dir}")

if __name__ == '__main__':
    from Config import Config
    config = Config("main_programm")
    main(config)
