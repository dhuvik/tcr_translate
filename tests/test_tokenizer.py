###############################################################################
######## Tests for Classes and Functions for TCRpMHCDataset Object ############
###############################################################################

# Import custom classes
from src.tokenizer import *
from transformers.tokenization_utils_base import BatchEncoding
from transformers import T5Tokenizer, BartTokenizer


class TestTCRT5Tokenizer:

    @classmethod
    def setup_class(cls):
        cls.tokenizer = TCRT5Tokenizer()

    @classmethod
    def teardown_class(cls):
        del cls.tokenizer

    def test_init(self):
        assert self.tokenizer.bos_token == '[SOS]'
        assert self.tokenizer.eos_token == '[EOS]'
        assert self.tokenizer.sep_token == '[SEP]'
        assert self.tokenizer.cls_token == '[CLS]'
        assert self.tokenizer.unk_token == '[UNK]'
        assert self.tokenizer.pad_token == '[PAD]'
        assert self.tokenizer.mask_token == '[MASK]'

    def test_tokenize_single_tcr(self):
        # Test that the tokenizer returns the correct tokens
        tokenized_tcr = self.tokenizer.tokenize_tcr('CASSFLY', return_tensors='pt', padding=True)
        assert isinstance(tokenized_tcr, BatchEncoding)
        encoded_tcr = tokenized_tcr['input_ids'][0]
        assert self.tokenizer.decode(encoded_tcr) == '[TCR]CASSFLY[EOS]'

    def test_tokenize_multiple_tcrs(self):
        # Test that the tokenizer returns the correct tokens
        tokenized_tcrs = self.tokenizer.tokenize_tcr(['CASSFLY', 'CASSIRSSEQYF'], return_tensors='pt', padding=True)
        assert isinstance(tokenized_tcrs, BatchEncoding)
        encoded_tcrs = tokenized_tcrs['input_ids']
        assert encoded_tcrs.shape == (2, len('CASSIRSSEQYF')+2)
        assert self.tokenizer.decode(encoded_tcrs[0]) == '[TCR]CASSFLY[EOS][PAD][PAD][PAD][PAD][PAD]'
        assert self.tokenizer.decode(encoded_tcrs)[1] == '[TCR]CASSIRSSEQYF[EOS]'
        
    def test_tokenize_single_pmhc(self):
        # Test that the tokenizer returns the correct tokens
        tokenized_pmhc = self.tokenizer.tokenize_pmhc(('SIINFEKL', 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY'), return_tensors='pt', padding=True)
        assert isinstance(tokenized_pmhc, BatchEncoding)
        encoded_pmhc = tokenized_pmhc['input_ids'][0]
        assert self.tokenizer.decode(encoded_pmhc) == '[PMHC]SIINFEKL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY[EOS]'

    def test_tokenize_multiple_pmhcs(self):
        # Test that the tokenizer returns the correct tokens
        tokenized_pmhcs = self.tokenizer.tokenize_pmhc([('SIINFEKL', 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY'), ('GILGFVFTL', 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY')], return_tensors='pt', padding=True)
        assert isinstance(tokenized_pmhcs, BatchEncoding)
        encoded_pmhcs = tokenized_pmhcs['input_ids']
        assert encoded_pmhcs.shape == (2, 43+3)
        assert self.tokenizer.decode(encoded_pmhcs)[0] == '[PMHC]SIINFEKL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY[EOS][PAD]'
        assert self.tokenizer.decode(encoded_pmhcs)[1] == '[PMHC]GILGFVFTL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY[EOS]'
        assert self.tokenizer.batch_decode(encoded_pmhcs) == ['[PMHC]SIINFEKL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY[EOS][PAD]', '[PMHC]GILGFVFTL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY[EOS]']


class TestTCRBartTokenizer:
    @classmethod
    def setup_class(cls):
        cls.tokenizer = TCRBartTokenizer()

    @classmethod
    def teardown_class(cls):
        del cls.tokenizer

    def test_init(self):
        assert self.tokenizer.bos_token == '[SOS]'
        assert self.tokenizer.eos_token == '[EOS]'
        assert self.tokenizer.sep_token == '[SEP]'
        assert self.tokenizer.cls_token == '[CLS]'
        assert self.tokenizer.unk_token == '[UNK]'
        assert self.tokenizer.pad_token == '[PAD]'
        assert self.tokenizer.mask_token == '[MASK]'

    def test_tokenize_single_tcr(self):
        # Test that the tokenizer returns the correct tokens
        tokenized_tcr = self.tokenizer.tokenize_tcr('CASSFLY', return_tensors='pt', padding=True)
        assert isinstance(tokenized_tcr, BatchEncoding)
        encoded_tcr = tokenized_tcr['input_ids'][0]
        assert self.tokenizer.decode(encoded_tcr) == '[SOS]CASSFLY[EOS]'

    def test_tokenize_multiple_tcrs(self):
        # Test that the tokenizer returns the correct tokens
        tokenized_tcrs = self.tokenizer.tokenize_tcr(['CASSFLY', 'CASSIRSSEQYF'], return_tensors='pt', padding=True)
        assert isinstance(tokenized_tcrs, BatchEncoding)
        encoded_tcrs = tokenized_tcrs['input_ids']
        assert encoded_tcrs.shape == (2, len('CASSIRSSEQYF')+2)
        assert self.tokenizer.decode(encoded_tcrs[0]) == '[SOS]CASSFLY[EOS][PAD][PAD][PAD][PAD][PAD]'
        assert self.tokenizer.decode(encoded_tcrs)[1] == '[SOS]CASSIRSSEQYF[EOS]'
        
    def test_tokenize_single_pmhc(self):
        # Test that the tokenizer returns the correct tokens
        tokenized_pmhc = self.tokenizer.tokenize_pmhc(('SIINFEKL', 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY'), return_tensors='pt', padding=True)
        assert isinstance(tokenized_pmhc, BatchEncoding)
        encoded_pmhc = tokenized_pmhc['input_ids'][0]
        assert self.tokenizer.decode(encoded_pmhc) == '[SOS]SIINFEKL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY[EOS]'

    def test_tokenize_multiple_pmhcs(self):
        # Test that the tokenizer returns the correct tokens
        tokenized_pmhcs = self.tokenizer.tokenize_pmhc([('SIINFEKL', 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY'), ('GILGFVFTL', 'YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY')], return_tensors='pt', padding=True)
        assert isinstance(tokenized_pmhcs, BatchEncoding)
        encoded_pmhcs = tokenized_pmhcs['input_ids']
        assert encoded_pmhcs.shape == (2, 43+3)
        assert self.tokenizer.decode(encoded_pmhcs)[0] == '[SOS]SIINFEKL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY[EOS][PAD]'
        assert self.tokenizer.decode(encoded_pmhcs)[1] == '[SOS]GILGFVFTL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY[EOS]'
        assert self.tokenizer.batch_decode(encoded_pmhcs) == ['[SOS]SIINFEKL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY[EOS][PAD]', '[SOS]GILGFVFTL[SEP]YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY[EOS]']