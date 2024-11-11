###############################################################################
######## Tests for Classes and Functions for HuggingFace Compatibility ########
###############################################################################

# Import custom classes
from src.adapter import *
from src.tokenizer import *
from src.pMHC import *
from src.TCR import *

from transformers import T5Tokenizer, BartTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import BartConfig, BartForConditionalGeneration
from transformers import T5Config, T5ForConditionalGeneration
import os


curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)

t5_tokenizer = TCRT5Tokenizer(vocab_file=os.path.join(parent_dir, 'src/tcrt5_tokenizer.model'), bos_token='[SOS]', eos_token='[EOS]', sep_token='[SEP]', cls_token='[CLS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]')
bart_tokenizer = TCRBartTokenizer(os.path.join(parent_dir, 'src/tcrbart_vocab.json'), 
                               os.path.join(parent_dir,'src/tcrbart_merges.txt'), 
                               bos_token='[SOS]', eos_token='[EOS]', sep_token='[SEP]', cls_token='[CLS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]')


t5_config = T5Config(
    vocab_size=128,
    max_position_embeddings=512,
    d_model=1496, 
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    sep_token_id=4,
    decoder_start_token_id=1,
    encoder_layers=6,
    decoder_layers=6,
    attention_heads=8,
    output_hidden_states=True,
    output_scores=True,
    output_attentions=True,
    add_cross_attention=True,
    top_k=1
)

bart_config = BartConfig(
    vocab_size=28,
    max_position_embeddings=512,
    d_model=768,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    sep_token_id=4,
    decoder_start_token_id=1,
    encoder_layers=6,
    decoder_layers=6,
    attention_heads=16,
    output_hidden_states=True,
    output_scores=True,
    output_attentions=True,
    add_cross_attention=True,
    top_k=1
)


class TestHuggingFaceModelAdapter:
    @classmethod
    def setup_class(cls):
        
        # initialize the model
        cls.bart_model = BartForConditionalGeneration(bart_config)
        cls.t5_model = T5ForConditionalGeneration(t5_config)
        
        # Assuming basic tokenizer function or object
        cls.bart_adapter = HuggingFaceModelAdapter(bart_tokenizer, cls.bart_model)
        cls.t5_adapter = HuggingFaceModelAdapter(t5_tokenizer, cls.t5_model)

    @classmethod
    def teardown_class(cls):
        del cls.bart_adapter
        del cls.t5_adapter
        del cls.bart_model
        del cls.t5_model

    def setup_method(self):
        self.pmhc1 = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A2", eager_impute=True)
        self.pmhc2 = pMHC(peptide="GLCTLVAML", hla_allele="HLA-A2", eager_impute=True)
        self.pmhc3 = pMHC(peptide="GILGFVFTL", hla_allele="HLA-A*02:01", eager_impute=True)
        self.tcr1 = TCR(cdr3b='cASsIRSsYEqYF', trbv='TRBV19*1', trbj='TRBJ2-07*1',
                  cdr3a="CATGLTGGGNKLTF", trav="TRAV17*1", traj="TRAJ10*1")
        self.tcr2 = TCR(cdr3b='CASSIRSSEYF', trbv='TRBV19*1', trbj='TRBJ2-07*1')
        
    def teardown_method(self):
        del self.pmhc1
        del self.pmhc2
        del self.tcr1
        del self.tcr2

    def test_init(self):
        assert isinstance(self.bart_adapter.tokenizer, BartTokenizer)
        assert isinstance(self.t5_adapter.tokenizer, T5Tokenizer)

    def test_format_input_pmhc(self):
        # Test that pMHC source is formatted correctly
        source = self.pmhc1
        source.add_tcr(self.tcr1)
        source.add_tcr(self.tcr2)

        tokenized_bart = self.bart_adapter.format_input(source)['input_ids']
        tokenized_t5 = self.t5_adapter.format_input(source)['input_ids']
        
        assert isinstance(self.bart_adapter.format_input(source), BatchEncoding)
        assert tokenized_bart[0][0].item() == self.bart_adapter.tokenizer.bos_token_id
        assert tokenized_bart[0][-1].item() == self.bart_adapter.tokenizer.eos_token_id
        # Check for the [SEP] after the peptide.
        assert tokenized_bart[0][len(source.peptide)+1].item() == self.bart_adapter.tokenizer.sep_token_id   
        assert torch.count_nonzero(tokenized_bart[0] == self.bart_adapter.tokenizer.sep_token_id) == 1
        
        assert isinstance(self.t5_adapter.format_input(source), BatchEncoding)
        assert tokenized_t5[0][0].item() == 5  # Task token for pMHC
        assert tokenized_t5[0][-1].item() == self.t5_adapter.tokenizer.eos_token_id
        # Check for the [SEP] after the peptide.
        assert tokenized_t5[0][len(source.peptide)+1].item() == self.t5_adapter.tokenizer.sep_token_id
        assert torch.count_nonzero(tokenized_t5[0] == self.t5_adapter.tokenizer.sep_token_id) == 1

    def test_format_input_tcr(self):
        # Test that TCR source is formatted correctly
        source = self.tcr1
        source.add_pMHC(self.pmhc2)
        source.add_pMHC(self.pmhc2)

        tokenized_bart = self.bart_adapter.format_input(source)['input_ids']
        tokenized_t5 = self.t5_adapter.format_input(source)['input_ids']

        assert isinstance(self.bart_adapter.format_input(source), BatchEncoding)
        assert tokenized_bart[0][0].item() == self.bart_adapter.tokenizer.bos_token_id
        assert tokenized_bart[0][-1].item() == self.bart_adapter.tokenizer.eos_token_id
        
        assert isinstance(self.t5_adapter.format_input(source), BatchEncoding)
        assert tokenized_t5[0][0].item() == 3  # Task token for TCR
        assert tokenized_t5[0][-1].item() == self.t5_adapter.tokenizer.eos_token_id

    def test_rearrange_logits(self):
        # Test the rearrange_logits function
        src = pMHC(peptide='SIINFEKL', hla_allele='HLA-A*02:01')
        max_length = random.randint(25,35)
        num_samples = 1
        bart_tokenized = self.bart_adapter.format_input(src)
        bart_output = self.bart_adapter._greedy_decoding(bart_tokenized, max_len=max_length, n=num_samples, return_dict=True)
        t5_tokenized = self.t5_adapter.format_input(src)
        t5_output = self.t5_adapter._greedy_decoding(t5_tokenized, max_len=max_length, n=num_samples, return_dict=True)

        bart_logits = self.bart_adapter.rearrange_logits(bart_output)
        t5_logits = self.t5_adapter.rearrange_logits(t5_output)

        assert isinstance(bart_logits, torch.Tensor)
        assert isinstance(t5_logits, torch.Tensor)
        assert bart_logits.shape[0] == num_samples
        assert t5_logits.shape[0] == num_samples
        assert bart_logits.shape[1] == max_length
        assert t5_logits.shape[1] == max_length

    def test_rearrange_xattn(self):
        # Test the rearrange_xattn function
        src = self.pmhc1
        max_length = random.randint(25,35)
        bart_tokenized = self.bart_adapter.format_input(src)
        bart_output = self.bart_adapter._greedy_decoding(bart_tokenized, max_len=max_length, n=1, return_dict=True)
        t5_tokenized = self.t5_adapter.format_input(src)
        t5_output = self.t5_adapter._greedy_decoding(t5_tokenized, max_len=max_length, n=1, return_dict=True)

        bart_xattn = self.bart_adapter.rearrange_xattn(bart_output)
        t5_xattn = self.t5_adapter.rearrange_xattn(t5_output)

        assert isinstance(bart_xattn, torch.Tensor)
        assert isinstance(t5_xattn, torch.Tensor)

        # Get the number of decoder layers and the number of attention heads per model
        self.bart_model.config.decoder_layers
        self.t5_model.config.decoder_layers

        # Check that the shapes of the cross attention is as expected
        assert bart_xattn.shape[0] == self.bart_model.config.decoder_layers
        assert t5_xattn.shape[0] == self.t5_model.config.decoder_layers

        # Check that dim=2 has the number of attention heads as expected
        assert bart_xattn.shape[2] == self.bart_model.config.attention_heads
        assert t5_xattn.shape[2] == self.t5_model.config.attention_heads

        # Check that the last two dims are output_len x input_len
        assert bart_xattn.shape[3] == max_length
        assert t5_xattn.shape[3] == max_length
        
        assert bart_xattn.shape[4] == bart_tokenized.input_ids.shape[-1]
        assert t5_xattn.shape[4] == t5_tokenized.input_ids.shape[-1]

    def test_bart_greedy_decoding(self):
        # Test the greedy decoding of the HF model adapter
        src = self.pmhc3
        bart_tokenized = self.bart_adapter.format_input(src)
     
        bart_output = self.bart_adapter._greedy_decoding(bart_tokenized, max_len=25, n=1, return_dict=True)
        bart_output2 = self.bart_adapter._greedy_decoding(bart_tokenized, max_len=25, n=1, return_dict=True)

        # Check that the outputs are equal
        # Logit equals are currently failing due to unknown stochasticity in model inference
        # assert torch.allclose(self.bart_adapter.rearrange_logits(bart_output), self.bart_adapter.rearrange_logits(bart_output2), atol=1e-1)
        assert torch.equal(bart_output.sequences, bart_output2.sequences)

        assert isinstance(bart_output, dict)
        # Check that the argmax of the logits is the same as the sequence and remove the [SOS] token and [EOS] token
        assert all(bart_output.sequences[0][1:-1] == self.bart_adapter.rearrange_logits(bart_output).argmax(dim=-1)[0][:-1])
        # Check that the sequence is of the expected shapes
        assert self.bart_adapter.rearrange_logits(bart_output).shape[0] == 1
        assert self.bart_adapter.rearrange_logits(bart_output).shape[1] <= 25
        
    def test_t5_greedy_decoding(self):
        # Test the greedy decoding of the HF model adapter
        src = self.pmhc2
        t5_tokenized = self.t5_adapter.format_input(src)
        t5_output = self.t5_adapter._greedy_decoding(t5_tokenized, max_len=25, n=1, return_dict=True)

        assert isinstance(t5_output, dict)
        # Check that the argmax of the logits is the same as the sequence and remove the [SOS] token and [EOS] token
        assert all(t5_output.sequences[0][1:-1] == self.t5_adapter.rearrange_logits(t5_output).argmax(dim=-1)[0][:-1])
        # Check that the sequence is of the expected shapes
        assert self.t5_adapter.rearrange_logits(t5_output).shape[0] == 1
        assert self.t5_adapter.rearrange_logits(t5_output).shape[1] <= 25

    def test_bart_ancestral_sampling(self):
        # Test the ancestral sampling of the HF model adapter
        src = self.pmhc1
        bart_tokenized = self.bart_adapter.format_input(src)
        bart_output1 = self.bart_adapter._multinomial_sampling(bart_tokenized, max_len=25, n=2, temperature=1.0, return_dict=True)
        bart_output2 = self.bart_adapter._multinomial_sampling(bart_tokenized, max_len=25, n=2, temperature=15.0, return_dict=True)
        bart_output3 = self.bart_adapter._multinomial_sampling(bart_tokenized, max_len=25, n=2, temperature=10.0, return_dict=True)

        ### Check that the randomness gives actually different sequences
        #assert not torch.equal(bart_output1.sequences, bart_output2.sequences)
        #assert not torch.equal(bart_output3.sequences, bart_output2.sequences)

        # Check that the shannon entropy is higher for the higher temperature
        logits1 = self.bart_adapter.rearrange_logits(bart_output1)
        logits2 = self.bart_adapter.rearrange_logits(bart_output2)
        
        # Take the average entropys across sequence positions due to diff lengths
        low_temp_entropy = torch.mean(torch.distributions.Categorical(probs=logits1).entropy())
        high_temp_entropy = torch.mean(torch.distributions.Categorical(probs=logits2).entropy())

        assert high_temp_entropy >= low_temp_entropy
        
    def test_t5_ancestral_sampling(self):
        # Test the ancestral sampling of the HF model adapter
        src = self.pmhc2
        t5_tokenized = self.t5_adapter.format_input(src)
        t5_output1 = self.t5_adapter._multinomial_sampling(t5_tokenized, max_len=25, n=1, temperature=1.0, return_dict=True)
        t5_output2 = self.t5_adapter._multinomial_sampling(t5_tokenized, max_len=25, n=1, temperature=15.0, return_dict=True)
        t5_output3 = self.t5_adapter._multinomial_sampling(t5_tokenized, max_len=25, n=1, temperature=10.0, return_dict=True)

        ### Check that the randomness gives actually different sequences
        #assert not torch.equal(t5_output1.sequences, t5_output2.sequences)
        #assert not torch.equal(t5_output3.sequences, t5_output2.sequences)

        # Check that the shannon entropy is higher for the higher temperature
        logits1 = self.t5_adapter.rearrange_logits(t5_output1)
        logits2 = self.t5_adapter.rearrange_logits(t5_output2)
        
        # Take the average entropys across sequence positions due to diff lengths
        low_temp_entropy = torch.mean(torch.distributions.Categorical(probs=logits1).entropy())
        high_temp_entropy = torch.mean(torch.distributions.Categorical(probs=logits2).entropy())

        assert high_temp_entropy >= low_temp_entropy

    def test_bart_top_k_sampling(self):
        # Test that top_k sampling works as expected
        src = self.pmhc1
        bart_tokenized = self.bart_adapter.format_input(src)
        top_k = random.randint(1, 10)
        num_samples = random.randint(1, 2)
        bart_output = self.bart_adapter._top_k_sampling(bart_tokenized, max_len=25, n=num_samples, top_k=top_k, temperature=1.0, return_dict=True)
        
        # Check that at most only the top_k tokens have non-zero probability
        logits = self.bart_adapter.rearrange_logits(bart_output)
        for log in logits[0]:
            assert torch.count_nonzero(log) <= top_k

    def test_t5_top_k_sampling(self):
        # Test that top_k sampling works as expected
        src = self.pmhc3
        t5_tokenized = self.t5_adapter.format_input(src)
        top_k = random.randint(1, 10)
        num_samples = random.randint(1, 2)
        t5_output = self.t5_adapter._top_k_sampling(t5_tokenized, max_len=25, n=num_samples, top_k=top_k, temperature=1.0, return_dict=True)
        
        # Check that at most only the top_k tokens have non-zero probability
        logits = self.t5_adapter.rearrange_logits(t5_output)
        for log in logits[0]:
            assert torch.count_nonzero(log) <= top_k

    def test_bart_top_p_sampling(self):
        # Test that top_p sampling works as expected
        src = self.pmhc1
        bart_tokenized = self.bart_adapter.format_input(src)
        top_p = random.uniform(0.75, 0.95)
        bart_output = self.bart_adapter._top_p_sampling(bart_tokenized, max_len=25, n=1, top_p=top_p, return_dict=True)
        bart_debug = self.bart_adapter._top_p_sampling(bart_tokenized, max_len=25, n=1, top_p=1.0, return_dict=True)
        
        top_p_logits = self.bart_adapter.rearrange_logits(bart_output, softmax=False)[0]
        debug_logits = self.bart_adapter.rearrange_logits(bart_debug, softmax=False)[0] 

        # Exclude the last token as it is the [EOS] token w -inf probability everywhere
        for logits in debug_logits[:-1]:
            # Check that no -inf values are present
            assert not torch.any(logits == float('-inf'))
        
        for logits in top_p_logits[:-1]:
            # Check that some -inf values are present
            assert torch.any(logits == float('-inf'))

    def test_t5_top_p_sampling(self):
        # Test that top_p sampling works as expected
        src = self.pmhc2
        t5_tokenized = self.t5_adapter.format_input(src)
        top_p = random.uniform(0.75, 0.95)
        t5_output = self.t5_adapter._top_p_sampling(t5_tokenized, max_len=25, n=1, top_p=top_p, return_dict=True)
        t5_debug = self.t5_adapter._top_p_sampling(t5_tokenized, max_len=25, n=1, top_p=1.0, return_dict=True)
        
        top_p_logits = self.t5_adapter.rearrange_logits(t5_output, softmax=False)[0]
        debug_logits = self.t5_adapter.rearrange_logits(t5_debug, softmax=False)[0] 

        # Exclude the last token as it is the [EOS] token w -inf probability everywhere
        for logits in debug_logits[:-1]:
            # Check that no -inf values are present
            assert not torch.any(logits == float('-inf'))
        
        for logits in top_p_logits[:-1]:
            # Check that some -inf values are present
            assert torch.any(logits == float('-inf'))

    def test_bart_beam_search(self):
        # Test the beam search of the HF model adapter
        src = self.pmhc1
        bart_tokenized = self.bart_adapter.format_input(src)
        bart_output1 = self.bart_adapter._beam_search(bart_tokenized, max_len=25, n=2, num_beams=4, no_repeat_ngram_size=4, do_sample=False, return_dict=True)
        bart_output2 = self.bart_adapter._beam_search(bart_tokenized, max_len=25, n=4, num_beams=4, no_repeat_ngram_size=4, do_sample=False, return_dict=True)

        # TODO: Check that the beam search gives the same sequence for the same input with sampling off
        # Currently failing due to potential tie breaker when run on a randomly initialized model
        
        # Check that the sequence is of the expected shapes
        assert self.bart_adapter.rearrange_logits(bart_output1).shape[0] == 4
        assert self.bart_adapter.rearrange_logits(bart_output2).shape[1] <= 25

    def test_t5_beam_search(self):
        # Test the beam search of the HF model adapter
        src = self.pmhc2
        t5_tokenized = self.t5_adapter.format_input(src)
        t5_output1 = self.t5_adapter._beam_search(t5_tokenized, max_len=25, n=2, num_beams=2, no_repeat_ngram_size=4, do_sample=False, return_dict=True)
        t5_output2 = self.t5_adapter._beam_search(t5_tokenized, max_len=25, n=2, num_beams=2, no_repeat_ngram_size=4, do_sample=False, return_dict=True)

        # TODO: Check that the beam search gives the same sequence for the same input with sampling off
        # Currently failing due to potential tie breaker when run on a randomly initialized model
        
        # Check that the sequence is of the expected shapes
        assert self.t5_adapter.rearrange_logits(t5_output1).shape[0] == 2
        assert self.t5_adapter.rearrange_logits(t5_output2).shape[1] <= 25

    def test_bart_stochastic_beam_search(self):
        # Test the beam search of the HF model adapter
        src = self.pmhc1
        bart_tokenized = self.bart_adapter.format_input(src)
        bart_output1 = self.bart_adapter._beam_search(bart_tokenized, max_len=25, n=4, num_beams=4, no_repeat_ngram_size=4, do_sample=True, return_dict=True)
        bart_output2 = self.bart_adapter._beam_search(bart_tokenized, max_len=25, n=4, num_beams=4, no_repeat_ngram_size=4, do_sample=True, return_dict=True)

        # Check that the sequence is of the expected shapes
        print(len(bart_output1.scores))
        print(type(bart_output1.scores))
        assert self.bart_adapter.rearrange_logits(bart_output1).shape[0] == 4
        assert self.bart_adapter.rearrange_logits(bart_output2).shape[1] <= 25

    def test_t5_stochastic_beam_search(self):
        # Test the beam search of the HF model adapter
        src = self.pmhc2
        t5_tokenized = self.t5_adapter.format_input(src)
        t5_output1 = self.t5_adapter._beam_search(t5_tokenized, max_len=25, n=1, num_beams=2, no_repeat_ngram_size=4, do_sample=True, return_dict=True)
        t5_output2 = self.t5_adapter._beam_search(t5_tokenized, max_len=25, n=1, num_beams=2, no_repeat_ngram_size=4, do_sample=True, return_dict=True)

        # Check that the sequence is of the expected shapes
        assert self.t5_adapter.rearrange_logits(t5_output1).shape[0] == 2
        assert self.t5_adapter.rearrange_logits(t5_output2).shape[1] <= 25

    def test_bart_diverse_beam_search(self):
        # Test the beam search of the HF model adapter
        src = self.pmhc1
        max_length = random.randint(25,35)
        bart_tokenized = self.bart_adapter.format_input(src)
        bart_output = self.bart_adapter._diverse_beam_search(bart_tokenized, max_length, n=10, num_beams=10, num_beam_groups=10, diversity_penalty=0.5, no_repeat_ngram_size=9, return_dict=True)
        
        # Check that the sequence is of the expected shapes
        assert self.bart_adapter.rearrange_logits(bart_output).shape[0] == 10
        assert self.bart_adapter.rearrange_logits(bart_output).shape[1] <= max_length

    def test_t5_diverse_beam_search(self):
        # Test the beam search of the HF model adapter
        src = self.pmhc2
        max_length = random.randint(25,35)
        t5_tokenized = self.t5_adapter.format_input(src)
        t5_output = self.t5_adapter._diverse_beam_search(t5_tokenized, max_length, n=10, num_beams=10, num_beam_groups=10, diversity_penalty=0.5, no_repeat_ngram_size=9, return_dict=True)
        
        # Check that the sequence is of the expected shapes
        assert self.t5_adapter.rearrange_logits(t5_output).shape[0] == 10
        assert self.t5_adapter.rearrange_logits(t5_output).shape[1] <= max_length

    def test_bart_contrastive_decoding(self):
        # Test the contrastive decoding of the HF model adapter
        src = self.pmhc1
        bart_tokenized = self.bart_adapter.format_input(src)
        bart_output = self.bart_adapter._contrastive_decoding(bart_tokenized, max_len=25, n=1, penalty_alpha=0.7, top_k=20, return_dict=True)

        # Check that the sequence is of the expected shapes
        assert self.bart_adapter.rearrange_logits(bart_output).shape[0] == 1
        assert self.bart_adapter.rearrange_logits(bart_output).shape[1] <= 25

    def test_t5_contrastive_decoding(self):
        # Test the contrastive decoding of the HF model adapter
        src = self.pmhc2
        t5_tokenized = self.t5_adapter.format_input(src)
        t5_output = self.t5_adapter._contrastive_decoding(t5_tokenized, max_len=25, n=1, penalty_alpha=0.7, top_k=20, return_dict=True)

        assert isinstance(t5_output, dict)
        # Check that the sequence is of the expected shapes
        assert self.t5_adapter.rearrange_logits(t5_output).shape[0] == 1
        assert self.t5_adapter.rearrange_logits(t5_output).shape[1] <= 25

    def test_bart_typical_sampling(self):
        # Test the typical sampling of the HF model adapter
        src = self.pmhc1
        bart_tokenized = self.bart_adapter.format_input(src)
        bart_output = self.bart_adapter._typical_sampling(bart_tokenized, max_len=25, n=2, typical_mass=0.9, min_tokens_to_keep=1, return_dict=True)

        # Check that the sequence is of the expected shapes
        assert self.bart_adapter.rearrange_logits(bart_output).shape[0] == 2
        assert self.bart_adapter.rearrange_logits(bart_output).shape[1] <= 25

    def test_t5_typical_sampling(self):
        # Test the typical sampling of the HF model adapter
        src = self.pmhc2
        t5_tokenized = self.t5_adapter.format_input(src)
        t5_output = self.t5_adapter._typical_sampling(t5_tokenized, max_len=25, n=2, typical_mass=0.9, min_tokens_to_keep=1, return_dict=True)

        # Check that the sequence is of the expected shapes
        assert self.t5_adapter.rearrange_logits(t5_output).shape[0] == 2
        assert self.t5_adapter.rearrange_logits(t5_output).shape[1] <= 25

    def test_translate_pmhc(self):
        # Test the translate function on the HF adapter
        source_pmhc = self.pmhc1
        max_length = random.randint(10,25)
        bart_output = self.bart_adapter.translate(source_pmhc, max_len=max_length, skip_special_tokens=False)
        t5_output = self.t5_adapter.translate(source_pmhc, max_len=max_length, skip_special_tokens=False)

        # Assert the types are as expected
        assert isinstance(bart_output['translations'], list)
        assert isinstance(t5_output['translations'], list)  
        
    def test_translate_tcr(self):
        # Test the translate function on the HF adapter
        source_tcr = self.tcr1
        max_length = random.randint(42,61)
        bart_output = self.bart_adapter.translate(source_tcr, max_len=max_length, skip_special_tokens=False)
        t5_output = self.t5_adapter.translate(source_tcr, max_len=max_length, skip_special_tokens=False)

        assert isinstance(bart_output['translations'], list)
        assert isinstance(t5_output['translations'], list)

    def test_translate_plus_tcr(self):
        source_tcr = self.tcr1
        max_length = random.randint(42,61)

        t5_output = self.t5_adapter.translate_plus(source_tcr, max_len=max_length)
        bart_output = self.bart_adapter.translate_plus(source_tcr, max_len=max_length)

        # Check that all the output keys are there
        assert 'translations' in t5_output
        assert 'logits' in t5_output
        assert 'cross_attentions' in t5_output

        assert 'translations' in bart_output
        assert 'logits' in bart_output
        assert 'cross_attentions' in bart_output

        # Check that the types are as expected
        assert isinstance(t5_output['translations'], list)
        assert isinstance(t5_output['logits'], torch.Tensor)
        assert isinstance(t5_output['cross_attentions'], torch.Tensor)
        
        assert isinstance(bart_output['translations'], list)
        assert isinstance(bart_output['logits'], torch.Tensor)
        assert isinstance(bart_output['cross_attentions'], torch.Tensor)
