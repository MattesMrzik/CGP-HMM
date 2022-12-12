import argparse
import Utility

class Config():

    def __init__(self, for_which_program):
        self.parser = argparse.ArgumentParser(description='Config module description')
        self.manually_added_for_bench = []
        self.manually_added_for_main = []
        if for_which_program == "small_bench":
            self.add_small_bench()
        if for_which_program == "main_programm":
            self.add_small_bench()

        self.parse()


    def add_arg_small_bench(self, *kwargs, type = None, help ="help", default = None, action = None):
        if action == None:
            self.parser.add_argument(*kwargs, type = type, default = default, help = help)
            self.manually_added_for_bench.append(kwargs[-1])
        else:
            self.parser.add_argument(*kwargs, action = action, help = help)
            self.manually_added_for_bench.append(kwargs[-1])
        print(f"{kwargs}, type = {type}, default = {default}, action = {action}, help = {help}")
    def add_arg_main(self, *kwargs, type = None, help ="help", default = None, action = None):
        if action == None:
            self.parser.add_argument(*kwargs, type = type, default = default, help = help)
            self.manually_added_for_main.append(kwargs[-1])
        else:
            self.parser.add_argument(*kwargs, action = action, help = help)
            self.manually_added_for_main.append(kwargs[-1])
        print(f"{kwargs}, type = {type}, default = {default}, action = {action}, help = {help}")


    def parse(self):
        print("parsed")
        self.parsed_args = self.parser.parse_args()

    def __getattr__(self, name):
        print("name =", name)
        return self.parsed_args.__dict__[name]

    # def print(self):
    #     print(self.parsed_args.__dict__)

    def add_small_bench(self):
        self.add_arg_small_bench('--repeat', default = 1, type = int, help ="repeat the main_programm [int] times")

        self.add_main_programm()

    def add_main_programm(self):
        self.add_arg_main("-c", "--nCodons",default = 2, type = int)
        self.add_arg_main('--run_viterbi', action='store_true', help ="run_viterbi")

    def get_for_main(self):
        s = ""
        for key in self.manually_added_for_main:
            minus = "-" if len(key) == 1 else "--"
            print(f"key = {key}, values = {self.parsed_args.__dict__[key[len(minus):]]}")
            s += key  + " " + str(self.parsed_args.__dict__[key[len(minus):]]) + " "
        return(s)
