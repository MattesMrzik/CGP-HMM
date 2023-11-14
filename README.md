# CGP-HMM
This is the code of my masters's thesis, with the title "Unsupervised Learning of a Hidden Markov Model of a Family of Gene Structures from Unaligned Genomic Sequences.
## Abstract
In the rapidly advancing field of genomics, an increasing number of genomes are being
sequenced and are awaiting structural annotation. This thesis introduces `cgphmm`, a pioneering tool that utilizes alignment-free comparative analysis across multiple species.
The core innovation of `cgphmm` is the use of a modified Hidden Markov Model combined
with unsupervised learning using gradient descent, a distinctive approach that serves as
an important proof of concept in the field of genomic analysis. A probabilistic model is
constructed that accurately captures the genic structure and sequences of a given exon
family. This is achieved through an unsupervised learning process involving sequences
from different species believed to contain a homologous exon. The model can then be
used for simultaneous exon annotation across all input species. Furthermore, `cgphmm`
is highly accurate in predicting exon-intron boundaries of human coding exons. The
overall runtime scales linearly with the number of species. This feature is particularly
beneficial in the era of big data, where the number of sequenced genomes is growing
exponentially.

## Install

* git clone --recursive https://github.com/MattesMrzik/CGP-HMM
* cd viterbi_cc
* make
* chmod u+x Viterbi

## Run

* cd src
* python3 cgphmm.py

## Dataset creation
* _get\_exons\_df.py_ creates a df containing all exons.
* _get\_internal\_exon.py_ selects suitable exons. Mapps coordinates and extracts fasta sequences.
* _select\_good\_exons\_for\_training.py_ selects exons that can be used for training. Copies them to a new directory.

## Evaluation
* the script _multi\_run.py_ was used for running and evaluation.
