#######################################################
######## Helper Class/Methods for Translation #########
#######################################################

import numpy as np
import einops
from collections import Counter
import math
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from src.constants import *
import re
import torch
import random
import torch.nn.functional as F
import Levenshtein as levenshtein
from transformers import BartTokenizer, T5Tokenizer, BatchEncoding
from transformers import LogitsWarper
import warnings



class HuggingFaceModelAdapter:
    """
    Class to qualitatively and quantitatively evaluate the performance
    of a HuggingFace model on any TCRpMHC dataset. Model adapter is
    functionally a wrapper around the HuggingFace model and tokenizer.
    """
    def __init__(self, hf_tokenizer, hf_model, **kwargs):
        self.tokenizer = hf_tokenizer
        self.model = hf_model
        self.use_task_prefix = kwargs.get('use_task_prefix', False)
        self.device = kwargs.get('device', 'cpu')

    def format_input(self, source):
        """
        Prepare the input for the model. Interfaces with the TCRpMHC dataset __getitem__ method.

        Args:
            source (TCR or pMHC): A TCR or pMHC object from the TCRpMHC dataset designed to map to the opposite target.

        Returns:
            tokenized_src (BatchEncoding): The tokenized source input for the model.
        """
        # Source is a pMHC and target is a TCR (pMHC -> TCR)
        if hasattr(source, 'peptide'):
            # Tokenize the source depending on the tokenizer
            
            # If Bart tokenizer: '[CLS]PEPTIDE[SEP][SEP]PSEUDO/MHC[SEP]'  <- Double SEP is BART Format
            if isinstance(self.tokenizer, BartTokenizer):
                # seq = f'{source.peptide}{self.tokenizer.sep_token}{source.pseudo}'  <- If using single SEP token
                # tokenized_src = self.tokenizer(seq, return_tensors='pt').to(self.device)
                tokenized_src = self.tokenizer(source.peptide, source.pseudo, return_tensors='pt').to(self.device)  # Currently adds two [SEP] tokens in the middle
            # If T5 tokenizer: '[PMHC]PEPTIDE[SEP]PSEUDO/MHC[EOS]'
            elif isinstance(self.tokenizer, T5Tokenizer):
                seq = f'[PMHC]{source.peptide}{self.tokenizer.sep_token}{source.pseudo}'
                tokenized_src = self.tokenizer(seq, return_tensors='pt').to(self.device)
            else:
                raise NotImplementedError("This tokenizer has not been implemented or used in training.")
            # Return the tokenized src
            return tokenized_src
        
        # Source is a TCR and target is a pMHC (TCR -> pMHC)
        elif hasattr(source, 'cdr3b'):
            # If Bart tokenizer: 'PEPTIDE[SEP]PSEUDO'
            if isinstance(self.tokenizer, BartTokenizer):
                seq = source.cdr3b
            # If T5 tokenizer: '[TCR]CDR3b/TRB[EOS]'
            elif isinstance(self.tokenizer, T5Tokenizer):
                seq = f'[TCR]{source.cdr3b}'
            else:
                raise NotImplementedError("This tokenizer has not been implemented or used in training.")
            tokenized_src = self.tokenizer(seq, return_tensors='pt').to(self.device)
            return tokenized_src
        else:
            raise ValueError("This adapter must be used with a TCRpMHCDataset object yielding TCR and pMHC.")
        
    def format_output(self, trg):
        """
        Format the output of the model to be human readable.
        """
        pattern = r'\[.*?\]'
        # Use re.sub to remove the matched text (including brackets)
        result = re.sub(pattern, '', trg)
        return result
    
    @staticmethod
    def rearrange_logits(model_output, softmax=True):
        """
        Convenience function for massaging HF model output scores (logits) 
        into a more interpretable format.

        Nested structure of the Logits (Scores):
        
        Layer 1 (Tuple of length max_output_len):
            Layer 2 (Tensor of shape [bsz, vocab_size]):
        
        Args:
            model_output - Dictionary with keys:
                                            'sequences' - output tokens
                                            'scores' - logits 
                                            'encoder_attentions' - self_attn from encoders
                                            'encoder_hidden_states' - self-explanatory
                                            'decoder_attentions' - self_attn from decoders
                                            'cross_attentions' - cross attention decoder:encoder
                                            'decoder_hidden_states' - self_explanatory
                                            
        Returns:
            rearranged_logits - torch.Tensor of shape [bsz, ouputput_size, vocab_size]
                                For beam search and its variants it is [num_beams, output_size, vocab_size]
        """
        output_collated_tensors = torch.stack(model_output.scores)
        output_collated_tensors = einops.rearrange(output_collated_tensors, 'seq_len bsz vocab_size -> bsz seq_len vocab_size')
        if softmax:
            return F.softmax(output_collated_tensors, dim=-1)
        return output_collated_tensors

    @staticmethod
    def rearrange_xattn(model_output):
        """
        Convenience function for massaging HF model cross attention output

        
        Nested structure of the Cross Attention:
        
        Layer 1 (Tuple of length max_output_len):
            Layer 2 (Tuple of len num_decoders):
                Layer 3 (Tensor of shape [bsz, num_attn_heads, 1, max_output_len])
        
        
        Args:
            model_output - Dictionary with keys:
                                            'sequences' - output tokens
                                            'scores' - logits 
                                            'encoder_attentions' - self_attn from encoders
                                            'encoder_hidden_states' - self-explanatory
                                            'decoder_attentions' - self_attn from decoders
                                            'cross_attentions' - cross attention decoder:encoder
                                            'decoder_hidden_states' - self_explanatory
                                            
        Returns:
            x_attn - torch.Tensor of shape [num_decoders, bsz, num_attn_heads, output_size, input_size]
        """
        decoder_collated_tensors = [torch.stack(model_output.cross_attentions[i]) for i in range(len(model_output.cross_attentions))]
        # decoder_collated_tensors should now be of shape [num_decoders, bsz, num_attn_heads, 1, input_len]
        # Now we want to concatenate across the output len at axis=3
        x_attn = torch.cat(decoder_collated_tensors, dim=3)
        return x_attn
    
    def _greedy_decoding(self, tokenized_input, max_len, n=1, return_dict=False):
        """
        Implements greedy decoding (deterministic, agrmax over auto-regressively sampled logits) 
        according to the HF definition found here: https://huggingface.co/docs/transformers/generation_strategies#greedy-search

        Args: 
            tokenized_input (BatchEncoding): The tokenized input containing ['input_ids', 'attention_mask'].
            max_len (int): The maximum length of the generated sequence.
            n (int): The number of sequences to return. [Required to be 1 for greedy decoding]
            return_dict (bool): Whether to return the dictionary of outputs.

        Returns:
            output (ModelOutput): The output of the model (logits, attnentions, hidden states, etc.)
        """
        return self.model.generate(**tokenized_input, max_new_tokens=max_len, do_sample=False, num_beams=1, num_return_sequences=n, return_dict_in_generate=return_dict)
    
    def _multinomial_sampling(self, tokenized_input, max_len, n, temperature, return_dict=False):
        """
        Implements (ancestral) multinomial sampling (stochastic sampling from the multinomial distribution of the logits) 
        according to the HF definition found here: https://huggingface.co/docs/transformers/generation_strategies
        
        Args:
            tokenized_input (BatchEncoding): The tokenized input containing the ['input_ids', 'attention_mask'].
            max_len (int): The maximum length of the generated sequence.
            n (int): The number of sequences to return.
            temperature (float): The temperature of the softmax function.
            return_dict (bool): Whether to return the dictionary of outputs.
        """
        return self.model.generate(**tokenized_input, max_new_tokens=max_len, do_sample=True, temperature=temperature, num_return_sequences=n, return_dict_in_generate=return_dict)
    
    def _top_k_sampling(self, tokenized_input, max_len, n, top_k, temperature, return_dict=False):
        """
        Implements top-k sampling (stochastic sampling from the top-k logits) 
        according to the HF blog post: https://huggingface.co/blog/how-to-generate
        
        Args:
            tokenized_input (BatchEncoding): The tokenized input containing the ['input_ids', 'attention_mask'].
            max_len (int): The maximum length of the generated sequence.
            top_k (int): The number of top tokens to sample from.
            temperature (float): The temperature of the softmax function.
            return_dict (bool): Whether to return the dictionary of outputs.
        """
        return self.model.generate(**tokenized_input, max_new_tokens=max_len, do_sample=True, top_k=top_k, temperature=temperature, num_return_sequences=n, return_dict_in_generate=return_dict)

    def _top_p_sampling(self, tokenized_input, max_len, n, top_p, temperature=None, return_dict=False):
        """
        Implements top-p sampling (stochastic sampling from the top-p logits) 
        according to the HF blogpost found here: https://huggingface.co/blog/how-to-generate
        
        Args:
            tokenized_input (BatchEncoding): The tokenized input containing the ['input_ids', 'attention_mask'].
            max_len (int): The maximum length of the generated sequence.
            top_p (float): The cumulative probability threshold.
            temperature (float): The temperature of the softmax function.
            return_dict (bool): Whether to return the dictionary of outputs.
        """
        return self.model.generate(**tokenized_input, max_new_tokens=max_len, do_sample=True, top_p=top_p, temperature=temperature, top_k=0, num_return_sequences=n, return_dict_in_generate=return_dict)
    
    def _beam_search(self, tokenized_input, max_len, n, num_beams, no_repeat_ngram_size, do_sample=False, return_dict=False):
        """
        Implements deterministic and stochastic (multinomial) beam search according to the HF definition found here: 
        https://huggingface.co/docs/transformers/generation_strategies
        
        Beam search has been shown to work well when the output len is predictable (Murray et al., 2018) and 
        (Yang et al 2018). 

        Args:
            tokenized_input (BatchEncoding): The tokenized input containing ['input_ids', 'attention_mask'].
            max_len (int): The maximum length of the generated sequence.
            n (int): The number of sequences to return. Must be less than or equal to num_beams.
            num_beams (int): The number of beams to use.
            no_repeat_ngram_size (int): The size of the n-gram to avoid repeating.
            do_sample (bool): Whether to sample from the logits.
            return_dict (bool): Whether to return the dictionary of outputs.
        """
        assert n <= num_beams, "num_return_sequences must be less than or equal to num_beams"
        return self.model.generate(**tokenized_input, max_new_tokens=max_len, num_return_sequences=n, 
                            num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, do_sample=do_sample, return_dict_in_generate=return_dict)
    
    def _diverse_beam_search(self, tokenized_input, max_len, n, num_beams, num_beam_groups, diversity_penalty, no_repeat_ngram_size, return_dict=False):
        """
        NOTE: In practice this method works poorly for our task.
        Implements diverse beam search according to the HF definition found here: 
        https://huggingface.co/docs/transformers/generation_strategies
        From the paper described here: https://arxiv.org/pdf/1610.02424.pdf
        
        Args:
            tokenized_input (BatchEncoding): The tokenized input containing the ['input_ids', 'attention_mask'].
            max_len (int): The maximum length of the generated sequence.
            n (int): The number of sequences to return.
            num_beams (int): The number of beams to use.
            num_beam_groups (int): The number of groups to use for diverse beam search.
            diversity_penalty (float): The penalty to apply for diversity.
            no_repeat_ngram_size (int): The size of the n-gram to avoid repeating.
            return_dict (bool): Whether to return the dictionary of outputs.    
        """
        assert n <= num_beams, "num_return_sequences must be less than or equal to num_beams"
        return self.model.generate(**tokenized_input, max_new_tokens=max_len, num_return_sequences=n,  
                                    num_beams=num_beams, num_beam_groups=num_beam_groups, no_repeat_ngram_size=no_repeat_ngram_size, 
                                    diversity_penalty=diversity_penalty, do_sample=False, return_dict_in_generate=return_dict)

    def _contrastive_decoding(self, tokenized_input, max_len, n, penalty_alpha, top_k, return_dict=False):
        """
        Implements contrastive decoding according to the HF definition found here: 
        https://huggingface.co/docs/transformers/generation_strategies
        
        Args:
            tokenized_input (BatchEncoding): The tokenized input containing the ['input_ids', 'attention_mask']
            max_len (int): The maximum length of the generated sequence.
            n (int): The number of sequences to return.
            penalty_alpha (float): The penalty alpha to apply.
            top_k (int): The number of top tokens to sample from.
            return_dict (bool): Whether to return the dictionary of outputs.
        """
        return self.model.generate(**tokenized_input, max_new_tokens=max_len, num_return_sequences=n, top_k=top_k, penalty_alpha=penalty_alpha, do_sample=True, return_dict_in_generate=return_dict)
    
    def _typical_sampling(self, tokenized_input, max_len, n, typical_mass, min_tokens_to_keep, return_dict=False):
        """
        Implements typical sampling according to the HF definition found here: 
        https://huggingface.co/docs/transformers/generation_strategies
        
        Args:
            tokenized_input (BatchEncoding): The tokenized input containing the ['input_ids', 'attention_mask']
            max_len (int): The maximum length of the generated sequence.
            n (int): The number of sequences to return.
            typical_mass (float): The typical mass to apply.
            min_tokens_to_keep (int): The minimum tokens to keep.
            return_dict (bool): Whether to return the dictionary of outputs.
        """
        logits_warper = [TypicalLogitsWarper(mass=typical_mass, filter_value=-float("Inf"), min_tokens_to_keep=min_tokens_to_keep)]
        return self.model.generate(**tokenized_input, max_new_tokens=max_len, num_return_sequences=n, do_sample=True, logits_processor=logits_warper, return_dict_in_generate=return_dict)
    
    def translate(self, source, max_len=25, n=1, mode='greedy', return_logits=False, return_xattn=False, **kwargs):
        """ample_
        Implements various decoding strategies for the model to perform translation from source to target.

        Args:
            source (TCR or pMHC): The source input for the model.
            max_len (int): The maximum length of the generated sequence.
            n (int): The number of sequences to return.
            mode (str): The decoding strategy to use.
            return_logits (bool): Whether to return the logits of the output.
            return_xattn (bool): Whether to return the cross attention of the output.
            **kwargs: Additional keyword arguments for generation according to the HF API.

        """
        # Set the model to evaluation mode
        self.model.eval()
        # Format the input and return the tokenized source (TCR or pMHC object)
        tokenized_src = self.format_input(source)
        
        # Get the kwargs
        temperature = kwargs.get('temperature', 1.0)
        top_k = kwargs.get('top_k', None)
        top_p = kwargs.get('top_p', None)
        num_beams = kwargs.get('num_beams', None)
        no_repeat_ngram_size = kwargs.get('no_repeat_ngram_size', 5)
        num_beam_groups = kwargs.get('num_beam_groups', None)
        diversity_penalty = kwargs.get('diversity_penalty', 1.0)
        penalty_alpha =  kwargs.get('penalty_alpha', None)
        typical_mass = kwargs.get('typical_mass', None)
        min_tokens_to_keep = kwargs.get('min_tokens_to_keep', 1)
        skip_special_tokens = kwargs.get('skip_special_tokens', True)
        
        ### If the number of sequences to return is greater than 1, dont hold scores/xattn due to larger memory footprint
        return_dict = True if (return_xattn or return_logits) else False
        if return_dict and n > 1:
            warnings.warn("Returning logits for n > 1 is not recommended due to memory constraints.")
            return_dict = False

        if mode=='greedy':
            # Based on HF Definition of Greedy Decoding
            outputs = self._greedy_decoding(tokenized_src, max_len=max_len, n=n, return_dict=return_dict)
        elif mode=='ancestral':
            # Based on HF Definition of Multinomial Sampling
            outputs = self._multinomial_sampling(tokenized_src, max_len=max_len, n=n, temperature=temperature, return_dict=return_dict)
        elif mode=='top_k':
            # Top-k with temperature sampling
            assert top_k is not None
            outputs = self._top_k_sampling(tokenized_src, max_len=max_len, n=n, top_k=top_k, temperature=temperature, return_dict=return_dict)
        elif mode=='top_p':
            # Top-p with temperature sampling
            assert top_p is not None
            outputs = self._top_p_sampling(tokenized_src, max_len=max_len, n=n, top_p=top_p, temperature=temperature, return_dict=return_dict)
        elif mode=='beam':
            assert num_beams is not None
            # Do Deterministic Beam Search
            outputs = self._beam_search(tokenized_src, max_len=max_len, n=n, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, 
                                        return_dict=return_dict)                
        elif mode=='stochastic_beam':
            assert num_beams is not None
            # Do Beam Search with Multinomial Sampling
            outputs = self._beam_search(tokenized_src, max_len=max_len, n=n, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, 
                                        do_sample=True, return_dict=return_dict)
        elif mode=='diverse_beam':
            # Do Diverse Beam Search
            assert num_beams is not None
            assert num_beam_groups is not None
            outputs = self._diverse_beam_search(tokenized_src, max_len=max_len, n=n, num_beams=num_beams,
                                                no_repeat_ngram_size=no_repeat_ngram_size, diversity_penalty=diversity_penalty, 
                                                num_beam_groups=num_beam_groups, return_dict=return_dict)
        elif mode=='contrastive':
            assert penalty_alpha is not None
            assert top_k is not None
            outputs = self._contrastive_decoding(tokenized_src, max_len=max_len, n=n, penalty_alpha=penalty_alpha, top_k=top_k, return_dict=return_dict)
            
        elif mode=='typical':
            assert typical_mass is not None
            outputs = self._typical_sampling(tokenized_src, max_len=max_len, n=n, typical_mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep, return_dict=return_dict)
        else:
            raise NotImplementedError
        
        if not (return_xattn or return_logits):
            # Outputs are the sequences themselves so just perform the decoding on them directly
            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)
        else:
            # Outputs are the dictionary of outputs index into the sequences
            translations = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=skip_special_tokens)
        
        outs = {}
        outs['translations'] = [self.format_output(translation) for translation in translations]
        if return_xattn:
            outs['cross_attentions'] = self.rearrange_xattn(outputs)
        if return_logits:
            outs['logits'] = self.rearrange_logits(outputs)
        return outs
        
    def translate_plus(self, source, mode='greedy', max_len=25, **kwargs):
        """
        Translate a src into a single output sequence using greedy decoding
        and get the target along with the logits and cross attention.
        """
        return self.translate(source, max_len=max_len, n=1, mode=mode, return_logits=True, return_xattn=True, **kwargs)

    def sample_translations(self, source, max_len=25, n=5, mode='greedy', **kwargs):
        """
        Code to sample n translations from the model and return the translations only.

        Args:
            source: The input source data.
            max_len (int): The maximum length of the generated translations.
            n (int): The number of translations to consider.
            mode (str): The mode of generation.
            **kwargs: Additional keyword arguments for generation specific to the Huggingface Generation API.

        Returns:
            translations (List[str]): A list of generated translations.
        """
        outs = self.translate(source=source, max_len=max_len, n=n, mode=mode, return_logits=False, return_xattn=False,  **kwargs)
        return outs['translations']
    
    
class TypicalLogitsWarper(LogitsWarper):
    """
    Code taken directly from the Typical Sampling Codebase. 
    https://github.com/cimeister/typical-sampling/src/transformers/generation_logits_process.py
    """
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
    
    
    
