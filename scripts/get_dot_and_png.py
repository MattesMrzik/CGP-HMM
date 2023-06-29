#!/usr/bin/env python3
import json

if __name__ == "__main__":
    '''
    This doesnt need to be particularly fast since it isnt particularly
    usefull to visualize, i think for evaluation the matices.csv are better
    '''

    import sys
    sys.path.insert(0, "../src")
    from Config import Config
    config = Config()
    config.init_for_get_dot_and_png()

    after_or_before = ['after','before'][config.for_initial_weights]

    path_to_para_dir = f"{config.parent_input_dir}{after_or_before}_fit_para"

    def load_kernel(a_or_b : str):
        kernel_file_path = f"{path_to_para_dir}/{a_or_b.upper()}_kernel.json"
        with open(kernel_file_path, "r") as in_file:
            return json.load(in_file)

    config.model.export_to_dot_and_png(A_weights = load_kernel("A"), \
                                       B_weights = load_kernel("B"), \
                                       to_png = config.png, \
                                       name = after_or_before)
