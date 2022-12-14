import argparse
import Utility
import re

class Config():

    def __init__(self, for_which_program):
        self.parser = argparse.ArgumentParser(description='Config module description')
        self.manuall_arg_lists = {"small_bench" : [], "main_programm" : []}
        if for_which_program == "small_bench":
            self.add_small_bench()
        if for_which_program == "main_programm":
            self.add_main_programm()

        self.parsed_args = self.parser.parse_args()
        # for key in self.parser.__dict__:
        #     print(key, self.parser.__dict__[key])
        #     print()

    def add_arg_small_bench(self, *kwargs, type = None, help ="help", default = None, action = None):
        arg_name = kwargs[-1].strip("-")
        if action == None:
            self.parser.add_argument(*kwargs, type = type, default = default, help = help)
        else:
            self.parser.add_argument(*kwargs, action = action, help = help)
        self.manuall_arg_lists["small_bench"].append(arg_name)
        # print(f"{kwargs}, type = {type}, default = {default}, action = {action}, help = {help}")

    def add_arg_main(self, *kwargs, type = None, help ="help", default = None, action = None):
        arg_name = kwargs[-1].strip("-") , re.match("(-*)", kwargs[-1]).group(1)
        if action == None:
            self.parser.add_argument(*kwargs, type = type, default = default, help = help)
        else:
            self.parser.add_argument(*kwargs, action = action, help = help)
        self.manuall_arg_lists["main_programm"].append(arg_name)
        # print(f"{kwargs}, type = {type}, default = {default}, action = {action}, help = {help}")


    def print(self):
        s = "==========> config <==========\n"
        max_len = max([len(k[0]) for l in self.manuall_arg_lists.values() for k in l ])
        for key in [k for l in self.manuall_arg_lists.values() for k in l ]:
            s += " " * (max_len - len(key[0]))
            s += key[0]
            s += " = "
            s += str(self.parsed_args.__dict__[key[0]]) + "\n"
        s += "==========> config <=========="
        print(s)

    def __getattr__(self, name):
        return self.parsed_args.__dict__[name]

    def add_small_bench(self):
        self.add_arg_small_bench('--repeat', default = 1, type = int, help ="repeat the main_programm [int] times")

        self.add_main_programm()

    def add_main_programm(self):
        self.add_arg_main("-c", "--nCodons",default = 2, type = int)
        self.add_arg_main('--run_viterbi', action='store_true', help ="run_viterbi")

    def get_args_as_str(self, for_what):
        s = ""
        for key in self.manuall_arg_lists[for_what]:
            value = self.parsed_args.__dict__[key[0]]
            if type(value) == bool:
                s += key[1] + key[0] if value else ""
            else:
                s += key[1] + key[0]  + " " + str(value) + " "
        return(s)
