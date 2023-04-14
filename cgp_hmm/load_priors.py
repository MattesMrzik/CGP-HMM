#!/usr/bin/env python3
import pandas as pd
import re

class Prior():
    def __init__(self, config):
        prios_dir_path = config.prior_path
        self.exon_prior_df = self.load_priors_exon_priors(f"{prios_dir_path}/human_exon_probs.pbl")
        self.ASS_df = self.load_priors_splice_site(f"{prios_dir_path}/human_intron_probs.dss_start5_ass_start5.pbl", description="ASS", left_pattern_len=config.ass_start, right_patterh_len=config.ass_end)
        self.DSS_df = self.load_priors_splice_site(f"{prios_dir_path}/human_intron_probs.dss_start5_ass_start5.pbl", description="DSS", left_pattern_len=config.dss_start, right_patterh_len=config.dss_end)
        self.intron_prior_df = self.load_intron(f"{prios_dir_path}/human_intron_probs.pbl")
        

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
        df["win0"] = df["win0"].astype(float)
        df["win1"] = df["win1"].astype(float)
        df["win2"] = df["win2"].astype(float)

        # Show the DataFrame
        result_dict = df.set_index('pattern').to_dict('index')
        assert len(result_dict) == df['pattern'].nunique()

        epsilon = 1e-3
        for i in range(0, len(df), 4):
            chunk = df.iloc[i:i+4, :]
            for col in ["win0", "win1", "win2"]:
                assert abs(chunk[col].sum() - 1) < epsilon, f"sum of row ({i}), row+4 in col {col} is {chunk[col].sum()} and it should be one"
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
        df["prob"] = df["prob"].astype(float)
        df["prob"] = df["prob"]/1000
        # Show the DataFrame
        result_dict = df.set_index('seq').to_dict('index')
        assert len(result_dict) == df['seq'].nunique()
        epsilon = 2e-3
        assert abs(df["prob"].sum() - 1) < epsilon, f"sum of load_priors_splice_site is {df['prob'].sum()} and it should be one"
        return df, result_dict

    def load_intron(self, path, description = "EMISSION", order = 2):
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
        df["prob"] = df["prob"].astype(float)

        assert len(df.index) == 4**(order + 1), f"len(df.index) = {len(df.index)} and it should be {4**(order + 1)}"

        epsilon = 1e-4
        for i in range(0, len(df), 4):
            chunk = df.iloc[i:i+4, :]
            assert abs(chunk["prob"].sum() - 1) < epsilon, f"sum of row ({i}), row+4 is {chunk['prob'].sum()} and it should be one"
        # Show the DataFrame

        result_dict = df.set_index('pattern').to_dict('index')
        assert len(result_dict) == df['pattern'].nunique()
        return df, result_dict
    
    def get_intron_prob(self, pattern):
        return float(self.intron_prior_df[1][pattern.lower()]["prob"])
    
    def get_exon_prob(self, pattern, window = None):
        return float(self.exon_prior_df[1][pattern.lower()][f"win{window}"])
    
    def get_splice_site_matching_pattern_probs(self, description = None, pattern = None):
        assert description in ["ASS", "DSS"], "description in get_splice_site_matching_pattern_probs must be either ASS or DSS"
        df = self.ASS_df[0] if description == "ASS" else self.DSS_df[0]
        try:
            return df.groupby(df['seq'].str.match(pattern))['prob'].sum()[True]
        except:
            return 0
        # return(self.ASS_df[0][self.ASS_df[0].str.contains(pattern)]["probs"].sum())

class Config_impostor():
    def __init__(self):
        self.ass_start = 5
        self.ass_end = 2
        self.dss_start = 5
        self.dss_end = 2

if __name__ == "__main__":
    config = Config_impostor()
    p = Prior(config, "/home/mattes/Documents/cgp_data/priors/human/")
    print(p.ASS_df[0])
    # aaa = p.get_splice_site_matching_pattern_probs("ASS", r".{0}gaa.{2}AG.*")
    # aac = p.get_splice_site_matching_pattern_probs("ASS", r"gac.{2}AG.*")
    # aag = p.get_splice_site_matching_pattern_probs("ASS", r"gag.{2}AG.*")
    # aat = p.get_splice_site_matching_pattern_probs("ASS", r"gat.{2}AG.*")
    # aa = p.get_splice_site_matching_pattern_probs("ASS", r"ga.{3}AG.*")
    # print(aa, aaa+aac+aag+aat)
    # print(f"aaa {aaa}, aa {aa}, aaa/aa {aaa/aa}")
    # print(f"aaa {aaa}, aa {aa}, aaa/aa {aac/aa}")
    # print(f"aaa {aaa}, aa {aa}, aaa/aa {aag/aa}")
    # print(f"aaa {aaa}, aa {aa}, aaa/aa {aat/aa}")

    print(p.get_splice_site_matching_pattern_probs("ASS", ".{0}aaa.{2}AG.*"))
    