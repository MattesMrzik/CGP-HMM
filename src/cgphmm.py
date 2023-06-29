#!/usr/bin/env python3

def main(config):
    import tensorflow as tf
    from Training import fit_model
    import json
    import datetime


    model, history = fit_model(config)

    # writing the loss history and metrics to file
    with open(f"{config.current_run_dir}/loss.log", "w") as file:
        for loss in history.history['loss']:
            file.write(str(loss) + " " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            file.write("\n")

    with open(f"{config.current_run_dir}/history.log", "w") as file:
        file.write(json.dumps(history.history))

    if config.after_fit_png:
        cell = model.get_layer("cgp_hmm_layer").C
        config.model.export_to_dot_and_png(cell.A_kernel, cell.B_kernel, name = "after_fit", to_png = config.nCodons < 10)


    from Viterbi import main
    config.only_first_seq = True
    config.after_or_before = "a"
    main(config)
    config.after_or_before = "b"
    main(config)


if __name__ == '__main__':
    from Config import Config
    config = Config()
    config.init_for_training()
    main(config)
