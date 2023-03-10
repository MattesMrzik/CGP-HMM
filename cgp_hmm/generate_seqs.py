#!/usr/bin/env python3

def generate_simple():

    from itertools import product
    codons = []
    for codon in product("ACGT", repeat = 3):
        codon = "".join(codon)
        if codon not in ["TAA", "TGA", "TAG"]:
            codons += [codon]

    if config.forced_gene_structure:
        codons = ["AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC"]
    alphabet = ["A","C","G","T"]

    num_seqs = 100
    seqs = {}
    with open(config.fasta_path, "w") as file:
        max_left_flank_len = (config.seq_len - config.gen_len -6)//2
        max_right_flank_len = config.seq_len - config.gen_len - 6 - max_left_flank_len

        min_left_flank_len = max_left_flank_len if config.dont_strip_flanks else 1
        min_right_flank_len = max_right_flank_len if config.dont_strip_flanks else 1

        for seq_id in range(num_seqs):
            if config.forced_gene_structure:
                alphabet = ["T","G"]
            ig5 = "".join(np.random.choice(alphabet, np.random.randint(min_left_flank_len, max_left_flank_len +1))) # TODO: also check if low = 2
            atg = "ATG"
            # coding = "".join(np.random.choice(["A","C","G","T"], config["nCodons"] * 3))
            gene_codons =
            coding = "".join(np.random.choice(codons, config.nCodons))
            stop = np.random.choice(["TAA","TGA","TAG"])
            ig3 = "".join(np.random.choice(alphabet, np.random.randint(min_right_flank_len, max_right_flank_len +1)))

            seqs[f">use_simple_seq_gen_{seq_id}"] = ig5 + atg + coding + stop + ig3
        for key, value in seqs.items():
            file.write(key + "\n")
            file.write(value + "\n")
