#######################################################
######## Helper Functions for Model Eval ##############
#######################################################

import time 
# Import custom classes
from src.adapter import *
from src.tokenizer import *
from src.adapter import *
from src.evaluation import ModelEvaluator
from src.dataset import TCRpMHCdataset
from transformers import T5Tokenizer, BartTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import BartConfig, BartForConditionalGeneration
from transformers import T5Config, T5ForConditionalGeneration
import os


curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)

t5_tokenizer = TCRT5Tokenizer()
bart_tokenizer = TCRBartTokenizer()


t5_config = T5Config(
    vocab_size=128,
    max_position_embeddings=65,
    d_model=64, 
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    sep_token_id=4,
    decoder_start_token_id=1,
    encoder_layers=2,
    decoder_layers=2,
    attention_heads=1,
    output_hidden_states=True,
    output_scores=True,
    output_attentions=True,
    add_cross_attention=True,
    top_k=1
)

bart_config = BartConfig(
    vocab_size=28,
    max_position_embeddings=60,
    d_model=64,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    sep_token_id=4,
    decoder_start_token_id=1,
    encoder_layers=2,
    decoder_layers=2,
    attention_heads=1,
    output_hidden_states=True,
    output_scores=True,
    output_attentions=True,
    add_cross_attention=True,
    top_k=1
)


class TestModelEvaluator:
    @classmethod
    def setup_class(cls):
        
        # initialize the model
        cls.bart_model = BartForConditionalGeneration(bart_config)
        cls.t5_model = T5ForConditionalGeneration(t5_config)
        
        # Assuming basic tokenizer function or object
        cls.bart_evaluator = ModelEvaluator(bart_tokenizer, cls.bart_model)
        cls.t5_evaluator = ModelEvaluator(t5_tokenizer, cls.t5_model)

        cls.pmhc2tcr_dataset = TCRpMHCdataset(
                                source='pmhc',
                                target='tcr',
                                use_mhc=False,
                                use_pseudo=True,
                                use_cdr3=True,
        )
        sample_data_path = os.path.join(curr_dir, 'test_data/sampled_paired_data_cleaned.csv')
        sample_data = pd.read_csv(sample_data_path).sample(n=20).reset_index(drop=True)
        cls.pmhc2tcr_dataset._load_data_from_df(sample_data)

    @classmethod
    def teardown_class(cls):
        del cls.bart_evaluator
        del cls.t5_evaluator
        del cls.bart_model
        del cls.t5_model
        del cls.pmhc2tcr_dataset
            
    def setup_method(cls):
        pass

    def teardown_method(cls):
        pass

    def test_evaluate_loss(self):
        # Test to see if it works in the I/O sense
        dataset = self.pmhc2tcr_dataset
        # Evaluate functionality (does it take I/O properly)
        bart_loss = self.bart_evaluator.evaluate_loss(dataset)
        assert isinstance(bart_loss, float)
        assert bart_loss > 0.0

        t5_loss = self.t5_evaluator.evaluate_loss(dataset)
        assert isinstance(t5_loss, float)
        assert t5_loss > 0.0

    def test_sequence_bleu(self):
        # Test to see if it works in the I/O sense
        translation = "exempt"
        references = ["exemplar", "exemplary", "exemplify", "exemplification"]
        b3 = self.bart_evaluator._sequence_bleu(translation, references, max_references=3, max_ngram=3)
        b4 = self.bart_evaluator._sequence_bleu(translation, references, max_references=3, max_ngram=4)
        assert b3 != b4
        assert b3 >= b4

    def test_dataset_bleu(self):
        # Test to see if it works in the I/O sense
        dataset = self.pmhc2tcr_dataset
        # Evaluate functionality (does it take I/O properly)
        bart_bleu = self.bart_evaluator.dataset_bleu(dataset)
        assert isinstance(bart_bleu, float)
        assert bart_bleu >= 0.0

        t5_bleu = self.t5_evaluator.dataset_bleu(dataset)
        assert isinstance(t5_bleu, float)
        assert t5_bleu >= 0.0

    def test_find_n_closest_matches(self):
        # Define some example data
        query = "example"
        references = ["exampel", "xzampls", "exapmle", "exmplear"]
        n = 2
        # Expected result
        expected_result = ["exampel", "exapmle"]
        # Call the function to get the actual result
        bart_actual_result = self.bart_evaluator.find_n_closest_matches(query, references, n)
        t5_actual_result = self.t5_evaluator.find_n_closest_matches(query, references, n)
        # Assert that the actual result matches the expected result
        assert bart_actual_result == expected_result
        assert t5_actual_result == expected_result

    def test_precision_at_k(self):
        # Test the precision at k function
        sample_translations = ["example", "counterfactual", "exemplar"]
        references = ["example", "exemplar", "prototype", "paradigm"]
        assert self.bart_evaluator._precision_at_k(sample_translations, references) == float(2/3)

    def test_recall_at_k(self):
        # Test the recall at k function with less than n_ref translations
        sample_translations = ["example", "counterfactual", "exemplar"]
        references = ["example", "exemplar", "prototype", "paradigm"]
        assert self.t5_evaluator._recall_at_k(sample_translations, references, k=3) == float(2/3)
        
        # Test the recall at k function with more than n_ref translations
        sample_translations2 = ["example", "counterfactual", "exemplar", "antithesis", "paragon"]
        assert self.bart_evaluator._recall_at_k(sample_translations2, references, k=5) == float(2/4)

    def test_f1_at_k(self):
        # Test the f1 at k function on a case where pr and rec are not 0
        sample_translations = ["example", "counterfactual", "exemplar"]
        references = ["example", "exemplar", "prototype", "paradigm"]
        assert self.bart_evaluator._f1_at_k(sample_translations, references, k=3) == 2/3

        # Test the f1 at k function on a case where pr and rec are 0
        sample_translations2 = ["counterfactual", "antithesis", "paragon"]
        assert self.t5_evaluator._f1_at_k(sample_translations2, references, k=3) == 0.0

    def test_mean_edit_distance(self):
        # Test the mean edit distance function
        sample_translations = ["example", "exampel", "exmplar"]
        references = ["example", "exemplar", "prototype", "paradigm"]
        assert self.t5_evaluator._mean_edit_distance(sample_translations, references) == np.mean([0, 2, 1])
    
    def test_mean_sequence_recovery(self):
        # Test the mean sequence recovery function
        sample_translations = ["example", "exampel", "parakeet"]
        references = ["example", "exemplar", "prototype", "paradigm"]
        assert self.bart_evaluator._mean_sequence_recovery(sample_translations, references) == np.mean([1.0, 5/7, 1/2])

    def test_atomic_metrics_at_k(self):
        # Test the get metrics at k function
        dataset = self.pmhc2tcr_dataset
        # Evaluate functionality (does it take I/O properly)
        translation_metrics = self.bart_evaluator.atomic_metrics_at_k(dataset, k=10, top_k=20, return_translations=True)
        assert isinstance(translation_metrics, dict)
        for pmhc in dataset.pMHCs:
            assert isinstance(translation_metrics[pmhc], dict)
            assert -100 not in list(translation_metrics[pmhc].values())
            assert isinstance(translation_metrics[pmhc]['precision'], float)
            assert isinstance(translation_metrics[pmhc]['recall'], float)
            assert isinstance(translation_metrics[pmhc]['f1'], float)
            assert isinstance(translation_metrics[pmhc]['d_edit'], float)
            assert isinstance(translation_metrics[pmhc]['char-bleu'], float)
            assert isinstance(translation_metrics[pmhc]['translations'], list)
            assert len(translation_metrics[pmhc]['translations']) == 10
            assert isinstance(translation_metrics[pmhc]['reference_translations'], list)

    def test_dataset_metrics_at_k(self):
        # Test the whole dataset-level metrics at k
        dataset = self.pmhc2tcr_dataset
        # Evaluate functionality (does it take I/O properly)
        dataset_metrics = self.bart_evaluator.dataset_metrics_at_k(dataset, k=10, top_k=20)
        assert isinstance(dataset_metrics, dict)
        
        
        assert -100 not in list(dataset_metrics.values())
        assert isinstance(dataset_metrics['precision'], float)
        assert isinstance(dataset_metrics['recall'], float)
        assert isinstance(dataset_metrics['f1'], float)
        assert isinstance(dataset_metrics['d_edit'], float)
        assert isinstance(dataset_metrics['char-bleu'], float)
        assert isinstance(dataset_metrics['perplexity'], float)

    def test_stratified_metrics_at_k(self):
        # Test the different stratified metrics at k
        dataset = self.pmhc2tcr_dataset
        stratify_on = 'Epitope'  # Allele or Epitope
        # Evaluate functionality (does it take I/O properly)
        fine_grained_metrics = self.bart_evaluator.stratified_metrics_at_k(dataset, k=10, top_k=20, stratify_on=stratify_on, return_translations=True)
        assert isinstance(fine_grained_metrics, dict)
        for group in self.pmhc2tcr_dataset.to_df()[stratify_on].unique():
            assert isinstance(fine_grained_metrics[group], dict)
            assert -100 not in list(fine_grained_metrics[group].values())
            assert isinstance(fine_grained_metrics[group]['precision'], float)
            assert isinstance(fine_grained_metrics[group]['recall'], float)
            assert isinstance(fine_grained_metrics[group]['f1'], float)
            assert isinstance(fine_grained_metrics[group]['d_edit'], float)
            assert isinstance(fine_grained_metrics[group]['char-bleu'], float)
            
            
