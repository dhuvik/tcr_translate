# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom wrapper around HF tokenizer to push some of the tokenization logic 
to the tokenizer itself. This is to make it easier to switch between TCR
and pMHC tokenization schemes.
"""

from transformers import T5Tokenizer, BartTokenizer
import torch
import numpy as np
import os
from typing import List, Optional
from transformers.tokenization_utils_base import BatchEncoding

from pathlib import Path

# Get current file's directory
CURRENT_DIR = Path(__file__).parent.resolve()

# Define paths relative to current directory
t5_vocab_file = str(CURRENT_DIR / 'tcrt5_tokenizer.model')
bart_vocab_file = str(CURRENT_DIR / 'tcrbart_vocab.json')
bart_merges_file = str(CURRENT_DIR / 'tcrbart_merges.txt')

class TCRT5Tokenizer(T5Tokenizer):
    def __init__(self, 
                 vocab_file=t5_vocab_file,
                 bos_token='[SOS]', 
                 eos_token='[EOS]', 
                 sep_token='[SEP]', 
                 cls_token='[CLS]', 
                 unk_token='[UNK]', 
                 pad_token='[PAD]', 
                 mask_token='[MASK]',
                 *args,
                 **kwargs):
        super().__init__(vocab_file=vocab_file,
                         bos_token=bos_token, 
                         eos_token=eos_token, 
                         sep_token=sep_token, 
                         cls_token=cls_token, 
                         unk_token=unk_token, 
                         pad_token=pad_token, 
                         mask_token=mask_token,
                         *args,
                         **kwargs)
        
    def tokenize_tcr(self, tcr, **kwargs):
        """
        Tokenizes TCR amino acid sequence(s).

        Args:
            tcr (list or str): List of or single TCR amino acid sequence. Can be full TRB/TRA or CDR3 (Single-chain).

        NOTE: Can extend to multiple chains by passing in a list of strings.

        Returns:
            List of tokenized TCRs.
        """
        # Add the TCR token to the beginning of the sequence
        if isinstance(tcr, list):
            tcrs = [f'[TCR]{tcr}' for tcr in tcr]
        else:
            tcrs = [f'[TCR]{tcr}']
        
        tokenized_tcr = self.__call__(tcrs, **kwargs)
        #assert isinstance(tokenized_tcr, BatchEncoding)
        return tokenized_tcr
    
    def tokenize_pmhc(self, pmhc, **kwargs):
        """
        Tokenizes pMHC amino acid sequence tuple(s).

        Args:
            pmhc (list or tuple): List of or single pMHC amino acid sequence tuple. Can be full MHC or psuedosequence.
        """
        # Split the pMHC into its components
        if isinstance(pmhc, list):
            pmhcs = [f'[PMHC]{peptide}{self.sep_token}{mhc}' for (peptide,mhc) in pmhc]    
        else:
            pmhcs = [f'[PMHC]{pmhc[0]}{self.sep_token}{pmhc[1]}']
        tokenized_pmhc = self.__call__(pmhcs, **kwargs)
        #assert isinstance(tokenized_pmhc, BatchEncoding)
        return tokenized_pmhc
    
    def decode(self, token_ids, **kwargs):
        """
        Decodes token ids to string.

        Args:
            token_ids (list or tensor): List of or single token ids.

        Returns:
            Decoded string or list of strings.
        """
        # Determine if there are multiple sequences
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, list):
            pass
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        else:
            raise ValueError("token_ids must be a list, numpy ndarray, or torch tensor.")
        
        if isinstance(token_ids[0], list):
            decoded = [self.decode(t, **kwargs) for t in token_ids]
        else:
            decoded = super().decode(token_ids, **kwargs).replace(' ', '')  # To address space after [SEP] token
        return decoded
        

class TCRBartTokenizer(BartTokenizer):
    def __init__(self, 
                 vocab_file=bart_vocab_file,
                 merges_file=bart_merges_file,
                 bos_token='[SOS]',
                 eos_token='[EOS]',
                 sep_token='[SEP]',
                 cls_token='[CLS]',
                 unk_token='[UNK]',
                 pad_token='[PAD]',
                 mask_token='[MASK]',
                 *args, 
                 **kwargs):
        super().__init__(vocab_file=vocab_file,
                         merges_file=merges_file,
                         bos_token=bos_token,
                         eos_token=eos_token,
                         sep_token=sep_token,
                         cls_token=cls_token,
                         unk_token=unk_token,
                         pad_token=pad_token,
                         mask_token=mask_token,
                         *args,
                         **kwargs)
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
        """
        DXK: This method overrides the `build_inputs_with_special_tokens` method from the `BartTokenizer` class.

        See original documentation here.

        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BART sequence has the following format:

        - single sequence: `[SOS]SEQ1[EOS]`
        - pair of sequences: `[SOS]SEQ1[SEP]SEQ2[EOS]`

        """
        sos = [self.bos_token_id]
        eos  = [self.eos_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return sos + token_ids_0 + eos
        return sos + token_ids_0 + sep + token_ids_1 + eos
    
    def tokenize_tcr(self, tcr, **kwargs):
        """
        Tokenizes TCR amino acid sequence(s).

        Args:
            tcr (list or str): List of or single TCR amino acid sequence. Can be full TRB/TRA or CDR3 (Single-chain).

        NOTE: Can extend to multiple chains by passing in a list of strings.

        Returns:
            List of tokenized TCRs.
        """
        if isinstance(tcr, list):
            pass
        else:
            tcr = [tcr]
        tokenized_tcr = self.__call__(tcr, **kwargs)
        return tokenized_tcr
    
    def tokenize_pmhc(self, pmhc, **kwargs):
        """
        Tokenizes pMHC amino acid sequence tuple(s).

        Args:
            pmhc (list or tuple): List of or single pMHC amino acid sequence tuple. Can be full MHC or psuedosequence.

        Returns:
            List of tokenized pMHCs.
            ["[CLS]PEPTIDE[SEP][SEP]MHCSEQUENCE[SEP]"...]
        """
        # Split the pMHC into its components
        if isinstance(pmhc, list):
            pass
        else:
            pmhc = [pmhc]

        tokenized_pmhc = self.__call__(pmhc, **kwargs)
        return tokenized_pmhc
    
    def decode(self, token_ids, **kwargs):
        """
        Decodes token ids to string.

        Args:
            token_ids (list or tensor): List of or single token ids.

        Returns:
            Decoded string or list of strings.
        """
        # Determine if there are multiple sequences
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, list):
            pass
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        else:
            raise ValueError("token_ids must be a list, numpy ndarray, or torch tensor.")
        
        if isinstance(token_ids[0], list):
            # Check to see if the tokens are a list of lists
            decoded = [self.decode(t, **kwargs) for t in token_ids]
        else:
            decoded = super().decode(token_ids, **kwargs).replace(' ', '')  # To address space after [SEP] token
        return decoded
            
        
        
