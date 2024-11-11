"""
Keep biological constants here.
"""
import os
import pandas as pd

# Get the directory of the current file (constants.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the 'refs' directory
refs_dir = os.path.join(current_dir, '..', 'refs')

HLA_SEQUENCE_MAP = pd.read_csv(os.path.join(refs_dir, '2field_hla_consensus_seqs.csv'), index_col=0).to_dict()["Full Sequence"]
HLA_PSEUDO_MAP = pd.read_csv(os.path.join(refs_dir, 'hla_pseudo_seqs.csv'), index_col=0).to_dict()["pseudo-sequence"]
AA_VOCABULARY = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

ATCHLEY_TABLE = pd.read_csv(os.path.join(refs_dir, 'atchley_factors.csv'), index_col=0)
KIDERA_TABLE = pd.read_csv(os.path.join(refs_dir, 'kidera_factors.csv'), index_col=0)
PROPERTIES_TABLE = pd.read_csv(os.path.join(refs_dir, 'aa_properties15.csv'), index_col=0)

PEPTIDE_MAX_LEN = 15
MHC_MAX_LEN = 365
PSEUDO_MAX_LEN = 34
TCR_MAX_LEN = 325
CDR3_MAX_LEN = 23
