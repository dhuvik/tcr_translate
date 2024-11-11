###############################################################################
######## Tests for Classes and Functions for TCRpMHCDataset Object ############
###############################################################################

# Import custom classes
from src.dataset import *
from src.tokenizer import *
import random
import pytest
import os
import time

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)

class TestTCRpMHCdataset:
    @classmethod
    def setup_class(cls):
        """
        Called before class initialization.
        """
        pass

    @classmethod
    def teardown_class(cls):
        """
        Called after every class initialization.
        """
        pass

    def setup_method(self):
        """
        Called before every method.
        """
        self.tcr2pmhc_dataset = TCRpMHCdataset(
                                source='tcr',
                                target='pmhc',
                                use_mhc=False,
                                use_pseudo=True,
                                use_cdr3=True,
        )
        self.pmhc2tcr_dataset = TCRpMHCdataset(
                                source='pmhc',
                                target='tcr',
                                use_mhc=False,
                                use_pseudo=True,
                                use_cdr3=True,
        )
        self.sample_data_path = os.path.join(curr_dir, 'test_data/sampled_paired_data_cleaned.csv')
        self.sample_data_df = pd.read_csv(self.sample_data_path)

    def teardown_method(self):
        """
        Called after every method.
        """
        del self.tcr2pmhc_dataset
        del self.pmhc2tcr_dataset
        del self.sample_data_df
        del self.sample_data_path

    def test_tcr2pmhc_init(self):
        """
        Test initialization of TCRpMHCdataset object.
        """
        assert self.tcr2pmhc_dataset.source == 'tcr'
        assert self.tcr2pmhc_dataset.target == 'pmhc'
        assert self.tcr2pmhc_dataset.use_mhc == False
        assert self.tcr2pmhc_dataset.use_pseudo == True
        assert self.tcr2pmhc_dataset.use_cdr3 == True
        assert len(self.tcr2pmhc_dataset) == 0
        assert str(self.tcr2pmhc_dataset) == 'TCR:pMHC Dataset of N=0. Mode:tcr -> pmhc.'

    def test_pmhc2tcr_init(self):
        """
        Test initialization of TCRpMHCdataset object.
        """
        assert self.pmhc2tcr_dataset.source == 'pmhc'
        assert self.pmhc2tcr_dataset.target == 'tcr'
        assert self.pmhc2tcr_dataset.use_mhc == False
        assert self.pmhc2tcr_dataset.use_pseudo == True
        assert self.pmhc2tcr_dataset.use_cdr3 == True
        assert len(self.pmhc2tcr_dataset) == 0
        assert str(self.pmhc2tcr_dataset) == 'TCR:pMHC Dataset of N=0. Mode:pmhc -> tcr.'

    def test_get_srclist(self):
        """
        Test getting the source list of the dataset.
        """
        self.tcr2pmhc_dataset._load_data_from_df(self.sample_data_df)
        self.pmhc2tcr_dataset._load_data_from_df(self.sample_data_df)
        for src in self.tcr2pmhc_dataset.get_srclist():
            assert isinstance(src, TCR)
        for src in self.pmhc2tcr_dataset.get_srclist():
            assert isinstance(src, pMHC)

    def test_get_trglist(self):
        # Test getting the source list of the dataset.
        self.tcr2pmhc_dataset.load_data_from_file(self.sample_data_path)
        self.pmhc2tcr_dataset.load_data_from_file(self.sample_data_path)
        for trg in self.tcr2pmhc_dataset.get_trglist():
            assert isinstance(trg, pMHC)
        for trg in self.pmhc2tcr_dataset.get_trglist():
            assert isinstance(trg, TCR)

    def test_seq2seq_dict(self):
        # Test the sequence to sequence mapping dict generated from the dataset
        self.tcr2pmhc_dataset.load_data_from_file(self.sample_data_path)
        self.pmhc2tcr_dataset.load_data_from_file(self.sample_data_path)
        
        tcr2pmhc_dict = self.tcr2pmhc_dataset.to_seq2seq_dict()
        pmhc2tcr_dict = self.pmhc2tcr_dataset.to_seq2seq_dict()
        

        # Test TCR to pMHC mapping
        for src, trg_set in tcr2pmhc_dict.items():
            # Check TCR expectations
            assert isinstance(src, TCR)
            assert '[SEP]' not in str(src)
            assert str(src).startswith('C')

            # Check pMHC expectations
            assert isinstance(trg_set, list)
            for trg in trg_set:
                assert isinstance(trg, str)
                assert '[SEP]' in trg

    
        # Test pMHC to TCR mapping
        for src, trg_set in pmhc2tcr_dict.items():
            # Check pMHC expectations
            assert isinstance(src, pMHC)
            assert '[SEP]' in str(src)

            # Check pMHC expectations
            assert isinstance(trg_set, list)
            for trg in trg_set:
                assert isinstance(trg, str)
                assert '[SEP]' not in trg
                assert trg.startswith('C')
    
    def test_load_data_from_file(self):
        """
        Test loading of data from a File by first making it into a suitable DF.
        """
        self.tcr2pmhc_dataset.load_data_from_file(self.sample_data_path)
        assert len(self.tcr2pmhc_dataset) == 6833
        assert str(self.tcr2pmhc_dataset) == 'TCR:pMHC Dataset of N=6833. Mode:tcr -> pmhc.'
        self.pmhc2tcr_dataset.load_data_from_file(self.sample_data_path)
        assert len(self.pmhc2tcr_dataset) == 6833
        assert str(self.pmhc2tcr_dataset) == 'TCR:pMHC Dataset of N=6833. Mode:pmhc -> tcr.'

    def test_load_data_from_df(self):
        """
        Test loading of data from a suitable DF.
        """
        self.tcr2pmhc_dataset._load_data_from_df(self.sample_data_df)
        assert len(self.tcr2pmhc_dataset) == 6833
        assert str(self.tcr2pmhc_dataset) == 'TCR:pMHC Dataset of N=6833. Mode:tcr -> pmhc.'
        self.pmhc2tcr_dataset._load_data_from_df(self.sample_data_df)
        assert len(self.pmhc2tcr_dataset) == 6833
        assert str(self.pmhc2tcr_dataset) == 'TCR:pMHC Dataset of N=6833. Mode:pmhc -> tcr.'

    def test_to_df(self):
        """
        Test conversion of TCRpMHCdataset to pandas dataframe.
        """
        self.tcr2pmhc_dataset.load_data_from_file(self.sample_data_path)
    
        assert isinstance(self.tcr2pmhc_dataset.to_df(), pd.DataFrame)
        assert len(self.tcr2pmhc_dataset.to_df()) == len(self.tcr2pmhc_dataset)
        assert isinstance(self.tcr2pmhc_dataset, TCRpMHCdataset)

        tcr_cdr3bs = [tcr.cdr3b for tcr in self.tcr2pmhc_dataset.tcrs]
        tcr_trbvs = [tcr.trbv for tcr in self.tcr2pmhc_dataset.tcrs]
        tcr_trbjs = [tcr.trbj for tcr in self.tcr2pmhc_dataset.tcrs]

        pmhc_peptides = [pmhc.peptide for pmhc in self.tcr2pmhc_dataset.pMHCs]
        pmhc_alleles = [pmhc.allele for pmhc in self.tcr2pmhc_dataset.pMHCs]

        # Assert that all TCR and pMHC sequences are in the dataset and in the same spot
        assert list(self.tcr2pmhc_dataset.to_df()['CDR3b']) == tcr_cdr3bs
        assert list(self.tcr2pmhc_dataset.to_df()['Epitope']) == pmhc_peptides
        assert list(self.tcr2pmhc_dataset.to_df()['Allele']) == pmhc_alleles
        assert list(self.tcr2pmhc_dataset.to_df()['TRBV']) == tcr_trbvs
        assert list(self.tcr2pmhc_dataset.to_df()['TRBJ']) == tcr_trbjs

    def test_get_dataloader(self):
        """
        Test dataloader creation for TCRpMHCdataset.
        """
        self.tcr2pmhc_dataset.load_data_from_file(self.sample_data_path)
        self.pmhc2tcr_dataset.load_data_from_file(self.sample_data_path)

        # Test dataloader creation with default batch_size
        tcr2pmhc_dataloader = self.tcr2pmhc_dataset.get_dataloader(TCRT5Tokenizer(), batch_size=1)
        pmhc2tcr_dataloader = self.pmhc2tcr_dataset.get_dataloader(TCRBartTokenizer(), batch_size=1)

        assert isinstance(tcr2pmhc_dataloader, torch.utils.data.DataLoader)
        assert isinstance(pmhc2tcr_dataloader, torch.utils.data.DataLoader)

        # Test dataloader creation with custom batch_size
        batch_size = 4
        tcr2pmhc_dataloader = self.tcr2pmhc_dataset.get_dataloader(TCRT5Tokenizer(), batch_size=batch_size)
        pmhc2tcr_dataloader = self.pmhc2tcr_dataset.get_dataloader(TCRBartTokenizer(), batch_size=batch_size)

        assert isinstance(tcr2pmhc_dataloader, torch.utils.data.DataLoader)
        assert isinstance(pmhc2tcr_dataloader, torch.utils.data.DataLoader)

        # Assert that the batch size is correct
        assert tcr2pmhc_dataloader.batch_size == batch_size
        assert pmhc2tcr_dataloader.batch_size == batch_size

    def test_collate_fn(self):
        """
        Test collate function for TCRpMHCdataset.
        """
        self.tcr2pmhc_dataset.load_data_from_file(self.sample_data_path)
        self.pmhc2tcr_dataset.load_data_from_file(self.sample_data_path)

        # Sample a batch of 4 from each dataset
        batch_size = 4
        tcr2pmhc_batch = []
        pmhc2tcr_batch = []
        for i in range(batch_size):
            tcr2pmhc_batch += [self.tcr2pmhc_dataset.__getitem__(random.randint(0, len(self.tcr2pmhc_dataset)))]
            pmhc2tcr_batch += [self.pmhc2tcr_dataset.__getitem__(random.randint(0, len(self.pmhc2tcr_dataset)))]

        # Collate the batch
        tcr2pmhc_collated = self.tcr2pmhc_dataset.collate_fn(TCRT5Tokenizer())(tcr2pmhc_batch)
        pmhc2tcr_collated = self.pmhc2tcr_dataset.collate_fn(TCRBartTokenizer())(pmhc2tcr_batch)

        assert isinstance(tcr2pmhc_collated, BatchEncoding)
        assert isinstance(pmhc2tcr_collated, BatchEncoding)

        # Assert that the batch size is correct
        assert len(tcr2pmhc_collated['input_ids']) == batch_size
        assert len(pmhc2tcr_collated['input_ids']) == batch_size

        # Assert that the input_ids are of the correct type
        assert isinstance(tcr2pmhc_collated['input_ids'], torch.Tensor)
        assert isinstance(pmhc2tcr_collated['input_ids'], torch.Tensor)

        # Assert that the input_ids are of the correct shape
        assert tcr2pmhc_collated['input_ids'].shape[1] <= 25  # Ensure that TCR is smaller
        assert tcr2pmhc_collated['labels'].shape[1] >= 34  # Ensure that pMHC is larger
        assert pmhc2tcr_collated['input_ids'].shape[1] >= 34  # Same as above
        assert pmhc2tcr_collated['labels'].shape[1] <= 25

    def test_dataloader_with_collating(self):
        """
        Test dataloader creation for TCRpMHCdataset with collating.
        """
        self.tcr2pmhc_dataset.load_data_from_file(self.sample_data_path)
        self.pmhc2tcr_dataset.load_data_from_file(self.sample_data_path)

        # Get the dataloader with two different tokenizers
        tcr2pmhc_dataloader = self.tcr2pmhc_dataset.get_dataloader(TCRT5Tokenizer(), batch_size=16)
        pmhc2tcr_dataloader = self.pmhc2tcr_dataset.get_dataloader(TCRBartTokenizer(), batch_size=16)

        # Assert that the batch size is correct
        assert tcr2pmhc_dataloader.batch_size == 16
        assert pmhc2tcr_dataloader.batch_size == 16

        # Assert that the correct number of batches are created
        assert len(tcr2pmhc_dataloader) <= len(self.tcr2pmhc_dataset) // 16 + (1 if len(self.tcr2pmhc_dataset) % 16 != 0 else 0)
        assert len(pmhc2tcr_dataloader) <= len(self.pmhc2tcr_dataset) // 16 + (1 if len(self.pmhc2tcr_dataset) % 16 != 0 else 0)

        # Check that we get the batch encodings
        batch_count = 0

        for batch in tcr2pmhc_dataloader:
            batch_count += 1
            assert isinstance(batch, BatchEncoding)

        assert batch_count == len(tcr2pmhc_dataloader)
        batch_count = 0

        for batch in pmhc2tcr_dataloader:
            batch_count += 1
            assert isinstance(batch, BatchEncoding)
        
        assert batch_count == len(pmhc2tcr_dataloader)
        