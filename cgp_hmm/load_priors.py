#!/usr/bin/env python3
import pandas as pd
import re

class Prior():
    def __init__(self, prios_dir_path):
        self.exon_prior_df = self.load_priors_exon_priors(f"{prios_dir_path}/human_exon_probs.pbl")
        self.ASS_df = self.load_priors_splice_site(f"{prios_dir_path}/human_intron_probs.pbl", description="ASS", left_pattern_len=3, right_patterh_len=2)
        self.DSS_df = self.load_priors_splice_site(f"{prios_dir_path}/human_intron_probs.pbl", description="DSS", left_pattern_len=4, right_patterh_len=3)
        self.intron_prior_df = self.load_intron_site(f"{prios_dir_path}/human_intron_probs.pbl")
        

    def load_priors_exon_priors(self, path, description = "EMISSION", order = 2):
        data = []
        with open(path, "r") as in_file:
            found_description = False
            for line in in_file:
                line = line.strip()
                found_description = found_description or (line == f"[{description}]")
                if found_description and len(line) == 0:
                    break
                if not found_description:
                    continue
                if not (x:= re.match(rf"^\w{{{order+1}}}\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+$", line)):
                    continue
                data.append(line)

        df = pd.DataFrame([line.split() for line in data], columns=['pattern', 'win0', 'win1', 'win2'])
        assert len(df.index) == 4**(order + 1), f"len(df.index) = {len(df.index)} and it should be {4**(order + 1)}"
        # Show the DataFrame
        result_dict = df.set_index('pattern').to_dict('index')
        assert len(result_dict) == df['pattern'].nunique()
        return df, result_dict
                        


    def load_priors_splice_site(self, path,  description = None, left_pattern_len = None, right_patterh_len = None):
        data = []
        with open(path, "r") as in_file:
            found_description = False
            for line in in_file:
                line = line.strip()
                found_description = found_description or (line == f"[{description}]")
                if found_description and len(line) == 0:
                    break
                if not found_description:
                    continue
                if not (x:= re.match(rf"^(\w{{{left_pattern_len}}})(\w{{{right_patterh_len}}})\s+(\d*\.?\d+(?:[Ee][+\-]?\d+)?)$", line)):
                    continue
                di_nuc = ["AG", "GT"][["ASS", "DSS"].index(description)]
                data.append((x.group(1), x.group(2), f"{x.group(1)}{di_nuc}{x.group(2)}",x.group(3)))

        df = pd.DataFrame(data, columns=['left_pattern', 'right_pattern', 'seq', 'prob'])
        # Show the DataFrame
        result_dict = df.set_index('seq').to_dict('index')
        assert len(result_dict) == df['seq'].nunique()
        return df, result_dict

    def load_intron_site(self, path, description = "EMISSION", order = 2):
        data = []
        with open(path, "r") as in_file:
            found_description = False
            for line in in_file:
                line = line.strip()
                found_description = found_description or (line == f"[{description}]")
                if found_description and len(line) == 0:
                    break
                if not found_description:
                    continue
                if not (x:= re.match(rf"^\w{{{order+1}}}\s+\d+\.\d+$", line)):
                    continue
                data.append(line)

        df = pd.DataFrame([line.split() for line in data], columns=['pattern', 'prob'])
        assert len(df.index) == 4**(order + 1), f"len(df.index) = {len(df.index)} and it should be {4**(order + 1)}"
        # Show the DataFrame

        result_dict = df.set_index('pattern').to_dict('index')
        assert len(result_dict) == df['pattern'].nunique()
        return df, result_dict
    
    def get_intron_prob(self, pattern):
        return float(self.intron_prior_df[1][pattern.lower()]["prob"])
    
    def get_exon_prob(self, pattern, window = None):
        return float(self.exon_prior_df[1][pattern.lower()][f"win{window}"])

    