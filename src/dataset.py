"""
The purpose of this python3 script is to implement the TCRpMHCdataset class.
"""

import torch
import pandas as pd
import numpy as np
import re
from src.constants import *
from src.TCR import TCR
from src.pMHC import pMHC
import warnings
from functools import partial

class TCRpMHCdataset(torch.utils.data.Dataset):
    """
    Main class for the TCRpMHC dataset. This class is designed to be used with the PyTorch DataLoader class. Designed for 
    creation of a dataset designed to capture the many to many nature of TCR and pMHC cross-reactivity. Should accept 
    either TCR -> multiple pMHC mapping or pMHC -> multiple TCR mapping.

    Args:
        source (str): The source of the dataset. Either 'tcr' or 'pmhc'.
        target (str): The target of the dataset. Either 'tcr' or 'pmhc'.
        use_mhc (bool): Whether to use the MHC sequence or the pMHC sequence.
        use_pseudo (bool): Whether to use the pseudo MHC sequence or the full MHC sequence.
        use_cdr3 (bool): Whether to use the CDR3 sequence or the full TCR sequence.
    
    Attributes:
        source (str): The source of the dataset. Either 'tcr' or 'pmhc'.
        target (str): The target of the dataset. Either 'tcr' or 'pmhc'.
        tcrs (list): The list of TCRs in the dataset.
        pMHCs (list): The list of pMHCs in the dataset.
        use_mhc (bool): Whether to use the MHC sequence or the pMHC sequence.
        use_pseudo (bool): Whether to use the pseudo MHC sequence or the full MHC sequence.
        use_cdr3 (bool): Whether to use the CDR3 sequence or the full TCR sequence.

    Returns:
        TCRpMHCdataset: A TCRpMHCdataset object
    """

    def __init__(self, source, target, use_mhc=False, use_pseudo=True, use_cdr3=True):
        self.source = source
        self.target = target
        assert self.source in ['tcr', 'pmhc']
        assert self.target in ['tcr', 'pmhc']
        self.tcrs = []
        self.pMHCs = []
        self.tcr_dict = dict()
        self.pmhc_dict = dict()
        self.use_mhc = use_mhc
        self.use_pseudo = use_pseudo if not use_mhc else False
        self.use_cdr3 = use_cdr3
        
    def __len__(self):
        assert len(self.pMHCs) == len(self.tcrs)
        return len(self.pMHCs)
    
    def __repr__(self):
        return f'TCRpMHCdataset(source="{self.source}", target="{self.target}",use_mhc={self.use_mhc},use_pseudo={self.use_pseudo}, use_cdr3={self.use_cdr3})'
    
    def __str__(self):
        return f'TCR:pMHC Dataset of N={self.__len__()}. Mode:{self.source} -> {self.target}.'
    
    def __getitem__(self, idx):
        """Return a tuple of (TCR object, PMHC) for the given index."""
        tcr = self.tcrs[idx] 
        pmhc = self.pMHCs[idx]
        if self.source == 'pmhc':
            return pmhc, tcr
        else:    
            return tcr, pmhc
        
    def get_srclist(self):
        """Return the list of source objects."""
        return self.tcrs if self.source == 'tcr' else self.pMHCs
    
    def get_trglist(self):
        """Return the list of target objects."""
        return self.pMHCs if self.source == 'tcr' else self.tcrs
    
    def load_data_from_file(self, path_to_csv):
        """
        Load the data from a csv file with the following required columns:
            1. 'CDR3b'
            2. 'TRBV'
            3. 'TRBJ'
            4. 'Epitope'
            5. 'Allele'
            6. 'Reference'
        
        Args:
            path_to_csv (str): The path to the csv file with the following columns:
                1. 'CDR3a': The CDR3a sequence in capital single letter Amino Acid Code format (str, optional)
              * 2. 'CDR3b': The CDR3b sequence in capital single letter Amino Acid Code format (str, required)
                3. 'TRAV': The TRAV gene in IMGT format (str, optional)
              * 4. 'TRBV': The TRBV gene in IMGT format (str, required)
                5. 'TRAJ': The TRAJ gene in IMGT format (str, optional)
              * 6. 'TRBJ': The TRBJ gene in IMGT format (str, required)
                7. 'TRAD': The TRAD gene in IMGT format (str, optional)
                8. 'TRBD': The TRBD gene in IMGT format (str, optional)
                9. 'TRA_stitched': The full TRA sequence in capital single letter Amino Acid Code format (str, optional)
                10. 'TRB_stitched': The full TRB sequence in capital single letter Amino Acid Code format (str, optional [can be imputed])
              * 11. 'Epitope': The peptide sequence in capital single letter Amino Acid Code format (str, required)
              * 12. 'Allele': The HLA allele in IMGT format (str, required)
                13. 'Pseudo': The pseudo MHC sequence in capital single letter Amino Acid Code format (str, optional [can be imputed])
                14. 'MHC': The full MHC sequence in capital single letter Amino Acid Code format (str, optional [can be imputed])
              * 15. 'Reference': The reference for the data point (str, required)
        Raises:
            FileNotFoundError: If the file is not found.                            
        Returns:
            None: This function does not return anything.

        """
        try:
            df = pd.read_csv(path_to_csv)
        except FileNotFoundError:
            print(f'File not found: {path_to_csv}')
            return
        
        assert 'CDR3b' in df.columns
        assert 'TRBV' in df.columns
        assert 'TRBJ' in df.columns
        assert 'Epitope' in df.columns
        assert 'Allele' in df.columns
        assert 'Reference' in df.columns
        
        if 'CDR3a' not in df.columns:
            df['CDR3a'] = ''
        if 'TRAV' not in df.columns:
            df['TRAV'] = ''
        if 'TRAJ' not in df.columns:
            df['TRAJ'] = ''
        if 'TRAD' not in df.columns:
            df['TRAD'] = ''
        if 'TRBD' not in df.columns:
            df['TRBD'] = ''
        if 'TRA_stitched' not in df.columns:
            df['TRA_stitched'] = ''
        if 'TRB_stitched' not in df.columns:
            df['TRB_stitched'] = ''
        if 'Pseudo' not in df.columns:
            df['Pseudo'] = ''
        if 'MHC' not in df.columns:
            df['MHC'] = ''
            
        self._load_data_from_df(df)

    def _load_data_from_df(self, df):
        """
        Load the data from a dataframe with the following required columns:

            1. 'CDR3a'
            2. 'CDR3b'
            3. 'TRAV'
            4. 'TRBV'
            5. 'TRAJ'
            6. 'TRBJ'
            7. 'TRAD'
            8. 'TRBD'
            9. 'TRA_stitched'
            10. 'TRB_stitched'
            11. 'Epitope'
            12. 'Allele'
            13. 'Pseudo'
            14. 'MHC'
            15. 'Reference'

        Args:
            df (pd.DataFrame): The dataframe with the following columns

        Raises:
            None

        Returns:
            None: This function does not return anything.

        """
        og_nrows = len(df)
        df = df.replace({np.nan: None})
        for index, row in df.iterrows():
            try:
                ### 1. Create the TCR and pMHC objects
                tcr_i = TCR(cdr3a=(row['CDR3a'] if isinstance(row['CDR3a'], str) else None), 
                            cdr3b=row['CDR3b'], 
                            trav=row['TRAV'], trbv=row['TRBV'], 
                            traj=row['TRAJ'], trbj=row['TRBJ'],
                            trad=row['TRAD'], trbd=row['TRBD'], 
                            tcra_full=row['TRA_stitched'], tcrb_full=row['TRB_stitched'],
                            reference=row['Reference'], use_cdr3b=self.use_cdr3)
            except:
                warnings.warn(f'Error loading row {index} TCR(cdr3a={row["CDR3a"]}, cdr3b={row["CDR3b"]},'
                              f'trbv={row["TRBV"]},trbj={row["TRBJ"]}). Skipping...', RuntimeWarning)
                continue
            
            try:
                pMHC_i = pMHC(peptide=row['Epitope'], hla_allele=row['Allele'], reference=row['Reference'], use_pseudo=self.use_pseudo, use_mhc=self.use_mhc, eager_impute=True)
            except:
                warnings.warn(f'Error loading row {index} pMHC(peptide={row["Epitope"]}, allele={row["Allele"]}). Skipping...', RuntimeWarning)
                continue
            
            ### 2. Hash the TCR and pMHC objects to get unique keys
            tcr_key = hash(tcr_i)
            pMHC_key = hash(pMHC_i)

            ### 3. If tcr exists then grab the existing tcr object and add the new information to it
            if tcr_key in self.tcr_dict.keys():
                tcr_i = self.tcr_dict[tcr_key]

            # Add reference and cognate pMHC information to that TCR (assumes no duplicates of paired data)
            tcr_i.add_reference(row['Reference'])
            tcr_i.add_pMHC(pMHC_i)
            # Add the updated version back to the dictionary
            self.tcr_dict[tcr_key] = tcr_i

            ### 4. If pmhc exists then grab the existing pmhc object and add the new information to it
            if pMHC_key in self.pmhc_dict.keys():
                pMHC_i = self.pmhc_dict[pMHC_key]
            # Add reference and cognate TCR
            pMHC_i.add_reference(row['Reference'])
            pMHC_i.add_tcr(tcr_i)
            # Add the updated version to the dictionary
            self.pmhc_dict[pMHC_key] = pMHC_i
            
            # Add TCR and PMHC to list **Updates the previous objects in the list thanks to pythons pointers**
            self.tcrs.append(tcr_i)
            self.pMHCs.append(pMHC_i)
            
                
        print(f'Loaded {len(self)} TCR:pMHC pairs from {og_nrows} rows of data.')

    def to_seq2seq_dict(self, stringify_input=False, stringify_output=True):    
        """
        TODO: check in the data the overlap between pMHC-TCRs and pSEUDO-TCRs
        
        Return a de-dpuplicated dictionary representation of the parallel dataset. 
        Keys are the source objects or their string representations, with string representations
        having more condensing of the data (by merging collisions). str() used in conjunction with 
        the use_cdr3b/use_pseudo flags to ensure that the same string representation 
        that will be passed through tokenization is used for the dict.
        
        Args:
            stringify_input (bool): Whether to convert the input to a string representation using __str__.
            stringify_output (bool): Whether to convert the output to a string representation using __str__.

        Raises:
            None

        Returns:
            data_dict (dict): The dictionary representation of the dataset with 
                                k,v pairs of some combination of source, target and
                                the repr function [{TCR: pMHC} repr(TCR):repr(pMHC)].
        """
        data_dict = dict()
        src_dict = self.tcr_dict if self.source == 'tcr' else self.pmhc_dict
        
        for src in src_dict.values():
            ref_trgs = src.get_pMHCs() if self.source == 'tcr' else src.get_tcrs()
            
            key = src if not stringify_input else str(src)
            values = [str(trg) if stringify_output else trg for trg in ref_trgs]
            
            # Update the dictionary
            if key in data_dict:
                data_dict[key].update(values)
            else:
                data_dict[key] = set(values)

        # Covner the sets to lists
        for k, v in data_dict.items():
            data_dict[k] = list(v)

        return data_dict
    
    def to_df(self):
        """
        Return a dataframe representation of the dataset instance.
        
        Args:
            None

        Raises:
            None

        Returns:
            df (pd.DataFrame): The dataframe representation of the dataset where each row is a unique TCR:pMHC pair and the reference is the 
            concatenation of the list of references for that pair. Multiple references are separated by a semicolon.
        """
        df = pd.DataFrame()
        df['CDR3a'] = [tcr.cdr3a if tcr.cdr3a is not None else '' for tcr in self.tcrs]
        df['CDR3b'] = [tcr.cdr3b if tcr.cdr3b is not None else '' for tcr in self.tcrs]
        df['TRAV'] = [tcr.trav if tcr.trav is not None else '' for tcr in self.tcrs]
        df['TRBV'] = [tcr.trbv if tcr.trbv is not None else '' for tcr in self.tcrs]
        df['TRAJ'] = [tcr.traj if tcr.traj is not None else '' for tcr in self.tcrs]
        df['TRBJ'] = [tcr.trbj if tcr.trbj is not None else '' for tcr in self.tcrs]
        df['TRAD'] = [tcr.trad if tcr.trad is not None else '' for tcr in self.tcrs]
        df['TRBD'] = [tcr.trbd if tcr.trbd is not None else '' for tcr in self.tcrs]
        df['TRA_stitched'] = [tcr.tcra_full if tcr.tcra_full is not None else '' for tcr in self.tcrs]
        df['TRB_stitched'] = [tcr.tcrb_full if tcr.tcrb_full is not None else '' for tcr in self.tcrs]
        df['Epitope'] = [pmhc.peptide if pmhc.peptide is not None else '' for pmhc in self.pMHCs]
        df['Allele'] = [pmhc.allele if pmhc.allele is not None else '' for pmhc in self.pMHCs]
        df['Pseudo'] = [pmhc.pseudo if pmhc.pseudo is not None else '' for pmhc in self.pMHCs]
        df['MHC'] = [pmhc.mhc for pmhc in self.pMHCs]
        df['Reference'] = [pmhc.get_references() for pmhc in self.pMHCs]
        return df
    
    def to_csv(self, path_to_csv):
        """
        Write the dataset to a csv file.
        
        Args:
            path_to_csv (str): The path to the csv file to write to.

        Raises:
            None

        Returns:
            None: This function does not return anything.
        """
        df = self.to_df()
        df.to_csv(path_to_csv, index=False)

    def get_dataloader(self, tokenizer, batch_size=1, shuffle=False, num_workers=0):
        """
        Return a dataloader for the dataset.
        
        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for the dataset.
            batch_size (int): The batch size for the dataloader.
            shuffle (bool): Whether to shuffle the dataset.
            num_workers (int): The number of workers to use for the dataloader.

        Raises:
            None

        Returns:
            dataloader (torch.utils.data.DataLoader): The dataloader for the dataset.
        """
        collate_fn = self.collate_fn(tokenizer)
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    
    def collate_fn(self, tokenizer):
        """
        Collate function for the dataloader. This function is used to collate the data into a batch.
        
        Args:
            batch (list): A list of tuples of (TCR, pMHC) objects called using the __getitem__.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use after collating the data.

        Raises:
            None

        Returns:
        
        """
        def _collate_fn(batch):
            source_batch, target_batch = tuple(zip(*batch))
        
            pmhc_list = [(src.peptide, src.pseudo) for src in source_batch] if self.source=='pmhc' else [(trg.peptide, trg.pseudo) for trg in target_batch]
            tcr_list = [src.cdr3b for src in source_batch] if self.source=='tcr' else [trg.cdr3b for trg in target_batch]

            # Use the tokenizer to tokenize the data using the appropriate logic
            batched_tcrs = tokenizer.tokenize_tcr(tcr_list, padding=True, truncation=True, max_length=25, return_tensors='pt')
            batched_pmhcs = tokenizer.tokenize_pmhc(pmhc_list, padding=True, truncation=True, max_length=52, return_tensors='pt')

            # Return the batched tensor data in the form BatchEncoding
            if self.source == 'pmhc':
                # Get the labels (cognate sequence)
                batched_pmhcs['labels'] = batched_tcrs['input_ids']
                return batched_pmhcs
            else:
                batched_tcrs['labels'] = batched_pmhcs['input_ids']
                return batched_tcrs
        
        return _collate_fn