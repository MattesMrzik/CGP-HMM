#!/usr/bin/env python3
import pandas as pd
import re
import os
import json

class Prior():
    def __init__(self, config):
        print("getting concentration for prior and initial paramters")
        self.config = config
        prios_dir_path = config.prior_path
        intron_pbl_path = os.path.join(prios_dir_path,"human_intron_probs.pbl")
        exon_pbl_path = os.path.join(prios_dir_path,"human_exon_probs.pbl")

        dir_name = f"n{config.nCodons}_assstart{config.ass_start}_assend{config.ass_end}_dssstart{config.dss_start}_dssend{config.dss_end}"
        path_to_dir_that_to_save_precal = os.path.join(config.out_path, "precalculated_emission_priors", dir_name)

        if not os.path.exists(path_to_dir_that_to_save_precal):
            os.makedirs(path_to_dir_that_to_save_precal)

        ass_aux_path = os.path.join(path_to_dir_that_to_save_precal, "ass_aux.json")
        dss_aux_path = os.path.join(path_to_dir_that_to_save_precal, "dss_aux.json")

        exon_prior_path_df = os.path.join(path_to_dir_that_to_save_precal, "exon.csv")
        exon_prior_path_dict = os.path.join(path_to_dir_that_to_save_precal, "exon.json")

        intron_prior_path_df = os.path.join(path_to_dir_that_to_save_precal, "intron.csv")
        intron_prior_path_dict = os.path.join(path_to_dir_that_to_save_precal, "intron.json")

        ass_prior_path_df = os.path.join(path_to_dir_that_to_save_precal, "ass.csv")
        ass_prior_path_dict = os.path.join(path_to_dir_that_to_save_precal, "ass.json")

        dss_prior_path_df = os.path.join(path_to_dir_that_to_save_precal, "dss.csv")
        dss_prior_path_dict = os.path.join(path_to_dir_that_to_save_precal, "dss.json")

        # aux
        if os.path.exists(ass_aux_path):
            with open(ass_aux_path, "r") as file:
                print("loading", ass_aux_path)
                self.ASS_aux = json.load(file)
        else:
            self.ASS_aux = self.get_auxilary_data(intron_pbl_path, "ASS")
            with open(ass_aux_path, "w") as file:
                 json.dump(self.ASS_aux, file)

        if os.path.exists(dss_aux_path):
            with open(dss_aux_path, "r") as file:
                print("loading", dss_aux_path)
                self.DSS_aux = json.load(file)
        else:
            self.DSS_aux = self.get_auxilary_data(intron_pbl_path, "DSS")
            with open(dss_aux_path, "w") as file:
                 json.dump(self.DSS_aux, file)

        # exon
        if os.path.exists(exon_prior_path_df) and os.path.exists(exon_prior_path_dict):
            print("loading", exon_prior_path_df)
            self.exon_prior_df = pd.read_csv(exon_prior_path_df)
            print("loading", exon_prior_path_dict)
            with open(exon_prior_path_dict, "r") as file:
                self.exon_prior_dict = json.load(file)
        else:
            self.exon_prior_df, self.exon_prior_dict = self.load_priors_exon_priors(exon_pbl_path)
            with open(exon_prior_path_dict, "w") as file:
                 json.dump(self.exon_prior_dict, file)
            self.exon_prior_df.to_csv(exon_prior_path_df)

        # intron
        if os.path.exists(intron_prior_path_df) and os.path.exists(intron_prior_path_dict):
            print("loading", intron_prior_path_df)
            self.intron_prior_df = pd.read_csv(intron_prior_path_df)
            print("loading", intron_prior_path_dict)
            with open(intron_prior_path_dict, "r") as file:
                self.intron_prior_dict = json.load(file)
        else:
            self.intron_prior_df, self.intron_prior_dict = self.load_intron(intron_pbl_path)
            with open(intron_prior_path_dict, "w") as file:
                 json.dump(self.intron_prior_dict, file)
            self.intron_prior_df.to_csv(intron_prior_path_df)

        # ass
        if os.path.exists(ass_prior_path_df) and os.path.exists(ass_prior_path_dict):
            print("loading", ass_prior_path_df)
            self.ASS_df = pd.read_csv(ass_prior_path_df)
            print("loading", ass_prior_path_dict)
            with open(ass_prior_path_dict, "r") as file:
                self.ASS_dict = json.load(file)
        else:
            self.ASS_df, self.ASS_dict = self.load_priors_splice_site(intron_pbl_path, description="ASS", left_pattern_len=config.ass_start, right_patterh_len=config.ass_end)
            with open(ass_prior_path_dict, "w") as file:
                 json.dump(self.ASS_dict, file)
            self.ASS_df.to_csv(ass_prior_path_df)
        # dss
        if os.path.exists(dss_prior_path_df) and os.path.exists(dss_prior_path_dict):
            print("loading", dss_prior_path_df)
            self.DSS_df = pd.read_csv(dss_prior_path_df)
            print("loading", dss_prior_path_dict)
            with open(dss_prior_path_dict, "r") as file:
                self.DSS_dict = json.load(file)
        else:
            self.DSS_df, self.DSS_dict = self.load_priors_splice_site(intron_pbl_path, description="DSS", left_pattern_len=config.dss_start, right_patterh_len=config.dss_end)
            with open(dss_prior_path_dict, "w") as file:
                 json.dump(self.DSS_dict, file)
            self.DSS_df.to_csv(dss_prior_path_df)

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

    def get_auxilary_data(self, path, description) -> dict:
        data = []
        with open(path, "r") as in_file:
            found_description = False
            for line in in_file:
            #    if line.startswith("#"):
            #        print(line.strip())
               if found_description:
                #    print("found_description")
                   if not line.startswith("#"):
                       data.append(float(line.strip()))
                       if len(data) == 3:
                            break
               found_description = found_description or (line.strip() == f"[{description}]")

        d = {"size of vector": data[0], \
             "site count" : data[1], \
             "pseudocount" : data[2]}
        return d

    def get_pseudocount_prob(self, description):
        if description == "ASS":
            return self.ASS_aux["pseudocount"]/self.ASS_aux["site count"]
        elif description == "DSS":
            return self.DSS_aux["pseudocount"]/self.DSS_aux["site count"]

    def get_sum_of_missing_values_added_pseudocount(self, description, len_df) -> float:
        '''
        description: is either ASS or DSS

        the pseudo count is added to all values in the df but
        the df doesnt contain patterns that wherent observed
        so to get prob sum of 1:
        sum df + (4^(ss_start + ss_end) - len(df)) * pseudocount / ss_count
        so this method returns (4^(ss_start + ss_end) - len(df)) * pseudocount / ss_count
        '''
        if description == "ASS":
            return (4**(self.config.ass_start + self.config.ass_end) - len_df) * self.get_pseudocount_prob(description)
        elif description == "DSS":
            return (4**(self.config.dss_start + self.config.dss_end) - len_df) * self.get_pseudocount_prob(description)
        else:
            print("desciption for get_sum_of_missing_values_added_pseudocount must be either ASS or DSS")
            exit()

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
        missing_pseudo_count_prob = self.get_sum_of_missing_values_added_pseudocount(description, len(df))

        assert abs(df["prob"].sum() + missing_pseudo_count_prob - 1) < epsilon, f"sum of load_priors_splice_site is {df['prob'].sum()} + {missing_pseudo_count_prob} = {df['prob'].sum() + missing_pseudo_count_prob} and it should be one, also check config.ass_start / end, config.dss_start /end"
        return df, result_dict

    def load_intron(self, path, description = "EMISSION", order = 2) -> tuple[pd.DataFrame, dict]:
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

        epsilon = 1e-3
        for i in range(0, len(df), 4):
            chunk = df.iloc[i:i+4, :]
            assert abs(chunk["prob"].sum() - 1) < epsilon, f"sum of row ({i}), row+4 is {chunk['prob'].sum()} and it should be one"
        # Show the DataFrame

        result_dict = df.set_index('pattern').to_dict('index')
        assert len(result_dict) == df['pattern'].nunique()
        return df, result_dict

    def get_intron_prob(self, pattern):
        return float(self.intron_prior_dict[pattern.lower()]["prob"])

    def get_exon_prob(self, pattern, window = None):
        return float(self.exon_prior_dict[pattern.lower()][f"win{window}"])

    def get_splice_site_matching_pattern_probs(self, description = None, pattern = None):
        assert description in ["ASS", "DSS"], "description in get_splice_site_matching_pattern_probs must be either ASS or DSS"
        df = self.ASS_df if description == "ASS" else self.DSS_df
        try:
            number_of_matching_entries_in_df = df.groupby(df['seq'].str.match(pattern))["prob"].count()[True]
            how_many_bases_in_pattern = sum([c.lower() in "acgt" for c in pattern])
            how_many_patterns_shoud_match = 4**(len(df.iloc[0]["seq"]) - how_many_bases_in_pattern)
            number_of_matching_patterns_that_are_not_in_df = how_many_patterns_shoud_match - number_of_matching_entries_in_df
            missing_prob = number_of_matching_patterns_that_are_not_in_df*self.get_pseudocount_prob(description)
            return df.groupby(df['seq'].str.match(pattern))['prob'].sum()[True]
        except:
            return 0
