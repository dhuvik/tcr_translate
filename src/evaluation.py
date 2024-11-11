#######################################################
######## Helper Functions for Model Eval ##############
#######################################################

import numpy as np
from tqdm import tqdm
from src.constants import *
from src.dataset import TCRpMHCdataset
import torch
import Levenshtein as levenshtein
import copy
from src.adapter import *
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction


class ModelEvaluator(HuggingFaceModelAdapter):
    """
    # TODO: MODIFY SO THAT IT MINIMIZES MODEL INFERENCES and doesn't duplicate computation

    Class to qualitatively and quantitatively evaluate the performance
    of a HuggingFace model on any TCRpMHC dataset. Model Evaluator takes
    as input an adapter module which is functionally a wrapper around the
    HuggingFace model and tokenizer.

    Attributes:
        hf_model_adapter: A HuggingFaceModelAdapter object.
    """

    def __init__(self, hf_tokenizer, hf_model, **kwargs):
        super().__init__(hf_tokenizer=hf_tokenizer, hf_model=hf_model, **kwargs)

    @staticmethod
    def find_n_closest_matches(query, references, n):
        # Find the n closest matches to the query in the reference list
        distances = [(ref, levenshtein.distance(query, ref)) for ref in references]
        distances.sort(key=lambda x: x[1])  # Sort in ascending order
        return [d[0] for d in distances[:n]]

    @staticmethod
    def _sequence_bleu(translation, references, max_references=20, max_ngram=4):
        """
        Calculate the sequence level Char-BLEU score for a single TCR or pMHC using the NLTK Sentence-Bleu function. Since
        the BLEU score is defined using a single hypothesis "translation" and multiple references, we use greedy
        decoding to generate the translations and compare them to the reference translations. The BLEU score is
        calculated by treating each character as a word and computing the BLEU score using the NLTK sequence-bleu.

        See documentation for the NLTK Corpus-Bleu function for more information:
        https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.sentence_bleu

        Args:
            translation (str): The hypothesis translation to be compared against the references.
            references (list): A list of reference translations.
            max_references (int): The maximum number of reference translations to consider.
            max_ngram (int): The maximum n-gram to consider Defaults to standard BLEU-4.

        Returns:
            bleu (float): The (sentence) sequence-level Char-BLEU score.
        """
        # Split up the sequences into a list of characters
        references = [
            list(x)
            for x in ModelEvaluator.find_n_closest_matches(
                translation, references, n=max_references
            )
        ]
        translation = list(translation)
        return float(
            sentence_bleu(
                references,
                translation,
                weights=tuple([1 / max_ngram] * max_ngram),
                smoothing_function=None,
            )
        )

    @staticmethod
    def _dataset_bleu(
        translations, references, max_references=20, max_ngram=4, verbose=False
    ):
        """
        Helper function to caluculate the BLEU score for a list of hyoptheses or references.

        Args:
            translations (list): A list of generated translations.
            references (list): A list of lists of reference translations.
            max_references (int): The maximum number of reference translations to consider.
            max_ngram (int): The maximum n-gram to consider Defaults to standard BLEU-4.
        """
        # Use a smoothing function to handle potential zero counts in n-gram matches
        chencherry = SmoothingFunction()

        expanded_references = []
        expanded_translations = []

        for idx, translation in enumerate(translations):
            # Fetching the n closest matches which will be used as the references for the translation
            expanded_references.append(
                [
                    list(x)
                    for x in ModelEvaluator.find_n_closest_matches(
                        translation, references[idx], n=max_references
                    )
                ]
            )
            expanded_translations.append(list(translation))

        # For debugging purposes
        if verbose == True:
            print(f"Expanded References:{expanded_references}")
            print(f"Expanded Translations:{expanded_translations}")

        return float(
            corpus_bleu(
                expanded_references,
                expanded_translations,
                weights=tuple([1 / max_ngram] * max_ngram),
                smoothing_function=chencherry.method1,
            )
        )

    def dataset_bleu(self, dataset, max_references=20, max_len=25, max_ngram=4):
        """
        Calculate the Dataset level Char-BLEU score for a TCRpMHC dataset using the NLTK Corpus-Bleu function. Since
        the BLEU score is defined using a single hypothesis "translation" and multiple references, we use greedy
        decoding to generate the translations and compare them to the reference translations. The BLEU score is
        calculated by treating each character as a word and computing the BLEU score using the NLTK Corpus-Bleu.

        See documentation for the NLTK Corpus-Bleu function for more information:
        https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.corpus_bleu

        Args:
            dataset (TCRpMHCDataset): A TCRpMHCDataset object.
            max_references (int): The maximum number of reference translations to consider.
                                  Too many references can slow down the calculation and lead to
                                  a less meaningful score. Defaults to 20.
            max_ngram (int): The maximum n-gram to consider Defaults to standard BLEU-4.

        Returns:
            bleu (float): The character level BLEU score.
        """
        translations = []
        references = []

        # Get the seq2seq mapping
        seq2seq_mapping = dataset.to_seq2seq_dict()
        for src in tqdm(seq2seq_mapping.keys(), desc="Char-BLEU"):
            # Get the ref trgs and append to references
            references.append(seq2seq_mapping[src])
            translations.append(
                self.sample_translations(
                    source=src, n=1, max_len=max_len, mode="greedy"
                )[0]
            )

        return self._dataset_bleu(
            translations, references, max_references=max_references, max_ngram=max_ngram
        )

    @staticmethod
    def _precision_at_k(translations, reference_translations, k=None):
        # Calculate the precision at k
        correct = [
            translation
            for translation in translations
            if translation in reference_translations
        ]
        return len(correct) / len(translations)

    @staticmethod
    def _recall_at_k(translations, reference_translations, k):
        # Calculate the recall at k
        correct = [
            translation
            for translation in translations
            if translation in reference_translations
        ]
        return len(set(correct)) / min(k, len(reference_translations))

    @staticmethod
    def _f1_at_k(translations, reference_translations, k=None):
        # Calculate the F1 score at k
        precision = ModelEvaluator._precision_at_k(
            translations, reference_translations, k
        )
        recall = ModelEvaluator._recall_at_k(translations, reference_translations, k)
        return (
            0.0
            if precision + recall == 0
            else (2 * precision * recall / (precision + recall))
        )

    @staticmethod
    def _mean_edit_distance(translations, reference_translations):
        # Calculate the mean edit distance
        edit_distances = []
        for translation in translations:
            closest_match = ModelEvaluator.find_n_closest_matches(
                translation, reference_translations, 1
            )[0]
            edit_distances += [levenshtein.distance(translation, closest_match)]
        return sum(edit_distances) / len(translations)

    @staticmethod
    def _mean_sequence_recovery(translations, reference_translations):
        # Calculate the mean sequence recovery as a percent for each translation in a set of translations
        per_sequence_percents = []
        for translation in translations:
            # Comparisons should only be made with references of the same length
            same_len_references = [
                ref for ref in reference_translations if len(ref) == len(translation)
            ]
            if len(same_len_references) == 0:
                closest_match = ModelEvaluator.find_n_closest_matches(
                    translation, reference_translations, 1
                )[0]
                per_sequence_percents.append(
                    1
                    - levenshtein.distance(translation, closest_match)
                    / len(closest_match)
                )
                continue
            closest_match = ModelEvaluator.find_n_closest_matches(
                translation, same_len_references, 1
            )[0]
            idx_recovery = [
                1 if char == closest_match[idx] else 0
                for idx, char in enumerate(translation)
            ]
            per_sequence_percents.append(sum(idx_recovery) / len(translation))
        return np.mean(per_sequence_percents)

    def get_batch_size(
        self,
        dataset,
        max_memory_usage: float = 0.97,
    ) -> int:
        # Set model to evaluation mode
        self.model.eval()

        # Initialize batch size
        bsz = 1
        while bsz < min(len(dataset), 2048):
            try:
                if bsz * 2 >= len(dataset):
                    return bsz
                dloader = dataset.get_dataloader(self.tokenizer, batch_size=bsz)

                # Perform forward pass
                with torch.no_grad():
                    batch = next(iter(dloader))
                    _ = self.model(**batch.to(self.device))

                # Check memory usage
                memory_allocated = torch.cuda.memory_allocated(self.device)
                memory_cached = torch.cuda.memory_reserved(self.device)
                memory_usage = memory_allocated / (memory_cached + 1)

                # Increase batch size if memory usage is within limit
                if memory_usage <= max_memory_usage:
                    bsz *= 2
                else:
                    break
            except RuntimeError:
                # Reduce batch size if out-of-memory error occurs
                bsz //= 2
                break

        # Cleanup
        torch.cuda.empty_cache()
        return bsz // 2

    def evaluate_loss(self, dataset, cumulative=False):
        """
        Evaluate the loss of the model on a dataset. The HF model loss function is used. CrossEntropyLoss
        for BART and T5 (and other LLMs).

        Args:
            dataset (TCRpMHCDataset): A TCRpMHCDataset object.
            bsz (int): The batch size to use for evaluation.
            cumulative (bool): Return the cumulative loss. Defaults to false (Avg loss will be returned)

        Returns:
            loss (float): The average/cumulative loss across the dataset
        """
        bsz = min(512, len(dataset))
        if self.device != "cpu":
            # Adjust if running on GPU
            bsz = self.get_batch_size(dataset)
        self.model.eval()
        dloader = dataset.get_dataloader(self.tokenizer, batch_size=bsz)
        num_batches = len(dloader)

        # Instantiate the cumulative loss
        cum_loss = 0

        with torch.no_grad():
            for batch in tqdm(dloader, desc="XEntropy Loss"):
                batch.to(self.device)
                # Perform the forward pass
                outs = self.model(**batch)
                loss = outs.loss
                cum_loss += loss
                torch.cuda.empty_cache()

        if cumulative:
            return cum_loss.item()

        return cum_loss.item() / num_batches

    def instance_metrics_at_k(
        self, source, k=100, max_len=25, mode="top_k", **translation_kwargs
    ):
        """
        Calculate the various performance metrics for a single source object. Currently implements
        precision, recall, F1 score, and mean edit distance at k for a single source object.

        Args:
            source: The input source data.
            k (int): The number of translations to consider.
            max_len (int): The maximum length of the generated translations.
            mode (str): The mode of generation. Can be 'top_k' or 'greedy'.
            **translation_kwargs: Additional keyword arguments for generation.
        """
        pass

    def atomic_metrics_at_k(
        self,
        dataset,
        k=100,
        max_len=25,
        return_translations=False,
        mode="top_k",
        **kwargs,
    ):
        """
        Calculate the various performance metrics for a dataset object. Currently implements
        precision, recall, F1 score, and mean edit distance at k for a dataset object. Exploring
        the use of Wasserstein distance as a metric for comparing the model's output distribution
        to the empirical distribution of the cognate -tope.

        Args:
            dataset (TCRpMHCDataset): A TCRpMHCDataset object.
            k (int): The number of translations to consider.
            max_len (int): The maximum length of the generated translations.
            summary (str): The summary statistic to use for the metrics. Can be 'mean', 'median', or 'geommean'.
            mode (str): The mode of generation. Can be 'top_k' or 'greedy'.
            return_translation (bool): Return the translations as well as the metrics. Defaults to False.
            **kwargs: Additional keyword arguments for generation.
        """

        metrics = {
            "char-bleu": -100,
            "precision": -100,
            "recall": -100,
            "f1": -100,
            "d_edit": -100,
            "seq_recovery": -100,
            "translations": None,
            "reference_translations": None,
        }

        # Get the reference translations (dictionary of source: set[translations])
        reference_dict = dataset.to_seq2seq_dict()
        translation_metrics = {}
        for source in tqdm(
            list(set(dataset.get_srclist())), desc="Calculating Atomic Metrics"
        ):
            # Instantiate a new metrics dictionary for each source
            translation_metrics[source] = copy.deepcopy(metrics)

            # Generate the translations and get the references
            src_translations = self.sample_translations(
                source, n=k, max_len=max_len, mode=mode, **kwargs
            )
            trg_sequences = reference_dict[source]

            if return_translations:
                translation_metrics[source]["translations"] = src_translations
                translation_metrics[source]["reference_translations"] = trg_sequences

            # Calculate the sentence-level BLEU score using the NLTK sentence_bleu function
            translation_metrics[source]["char-bleu"] = self._sequence_bleu(
                self.sample_translations(source, n=1, max_len=max_len, mode="greedy")[
                    0
                ],
                trg_sequences,
                max_ngram=4,
                max_references=20,
            )
            # translation_metrics[source]['char-bleu'] = self._dataset_bleu(translation_metrics[source]['translations'], trg_sequences)

            # Calculate the precision, recall, F1 score, and mean edit distance at k
            precision, recall, f1 = self._precision_recall_f1_at_k(
                src_translations, trg_sequences, k=k
            )
            translation_metrics[source]["precision"] = precision
            translation_metrics[source]["recall"] = recall
            translation_metrics[source]["f1"] = f1
            translation_metrics[source]["d_edit"] = self._mean_edit_distance(
                src_translations, trg_sequences
            )
            translation_metrics[source]["seq_recovery"] = self._mean_sequence_recovery(
                src_translations, trg_sequences
            )

        return translation_metrics

    def dataset_metrics_at_k(self, dataset, k=100, max_len=25, mode="top_k", **kwargs):
        """
        Calculate the performance metrics at dataset-level granularity TCRpMHC dataset object.

        Args:
            dataset (TCRpMHCDataset): A TCRpMHCDataset object.
            k (int): The number of translations to consider.
            max_len (int): The maximum length of the generated translations.
            summary (str): The summary statistic to use for the metrics. Can be 'mean', 'median', or 'geommean'.
            mode (str): The mode of generation. Can be 'top_k' or 'greedy'.
            return_translation (bool): Return the translations as well as the metrics. Defaults to False.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            metrics (dict): A dictionary containing the following performance metrics:
                            - char-bleu: The [micro-avg] dataset level Char-BLEU score.
                            - precision: The mean precision score.
                            - recall: The mean recall score.
                            - f1: The mean F1 score.
                            - d_edit: The mean edit distance.
                            - diversity: The number of unique sequences vs total sequences across all pMHC.
                            - perplexity: The perplexity score.
        """

        metrics = {
            "char-bleu": -100,
            "precision": [],
            "recall": [],
            "f1": [],
            "d_edit": [],
            "seq_recovery": [],
            "diversity": [],
            "perplexity": -100,
        }

        # Compute the corpus level char-bleu score of the dataset via greedy decoding.
        metrics["char-bleu"] = self.dataset_bleu(dataset)
        metrics["perplexity"] = np.exp(self.evaluate_loss(dataset, cumulative=False))

        # Get the reference translations (dictionary of source: list[translations])
        reference_dict = dataset.to_seq2seq_dict()

        for source in tqdm(
            list(set(dataset.get_srclist())), desc="Calculating @K Metrics"
        ):
            # Generate the translations
            src_translations = self.sample_translations(
                source, n=k, max_len=max_len, mode=mode, **kwargs
            )
            trg_sequences = reference_dict[source]

            # Calculate the precision, recall, F1 score, and mean edit distance at k
            precision, recall, f1 = self._precision_recall_f1_at_k(
                src_translations, trg_sequences, k=k
            )
            metrics["precision"] += [precision]
            metrics["recall"] += [recall]
            metrics["f1"] += [f1]
            metrics["d_edit"] += [
                self._mean_edit_distance(src_translations, trg_sequences)
            ]
            metrics["diversity"] += src_translations
            metrics["seq_recovery"] += [
                self._mean_sequence_recovery(src_translations, trg_sequences)
            ]

        # Calculate the mean precision, recall, F1 score, and mean edit distance at k
        metrics["precision"] = np.mean(metrics["precision"])
        metrics["recall"] = np.mean(metrics["recall"])
        metrics["f1"] = np.mean(metrics["f1"])
        metrics["d_edit"] = np.mean(metrics["d_edit"])
        metrics["diversity"] = len(set(metrics["diversity"])) / len(
            metrics["diversity"]
        )
        metrics["seq_recovery"] = np.mean(metrics["seq_recovery"])

        return metrics

    def stratified_metrics_at_k(
        self, dataset, stratify_on="Allele", k=100, max_len=25, mode="top_k", **kwargs
    ):
        df = dataset.to_df()
        fine_grained_metrics = {}
        # Get the different groups
        groups = df[stratify_on].unique()
        # Slice the dataframe into groups
        df_list = [
            df[df[stratify_on] == group].reset_index(drop=True) for group in groups
        ]

        # Create dataset objects for each group
        dset_list = [
            TCRpMHCdataset(
                source=dataset.source,
                target=dataset.target,
                use_pseudo=dataset.use_pseudo,
                use_cdr3=dataset.use_cdr3,
                use_mhc=dataset.use_mhc,
            )
            for _ in df_list
        ]

        # Load the data into the dataset objects
        for i, daf in enumerate(df_list):
            dset_list[i]._load_data_from_df(daf)

        # Get the metrics for each group
        for i, group in enumerate(groups):
            fine_grained_metrics[group] = self.dataset_metrics_at_k(
                dset_list[i], k=k, max_len=max_len, mode="top_k", **kwargs
            )
            fine_grained_metrics[group]["size"] = (
                len(set(dset_list[i].pMHCs))
                if dataset.source == "pmhc"
                else len(set(dset_list[i].tcrs))
            )
        return fine_grained_metrics

    def _precision_recall_f1_at_k(self, translations, ref_trgs, k=100):
        """
        Calculate precision, recall, F1 score at k for generated translations.

        Args:
            translations (list): A list of generated translations.
            ref_trgs (list): A list of reference translations.
            k (int): The number of translations to consider.

        Returns:
            precision (float): The precision score.
            recall (float): The recall score.
            f1 (float): The F1 score.
        """
        precision = self._precision_at_k(translations, ref_trgs, k=k)
        recall = self._recall_at_k(translations, ref_trgs, k=k)
        f1 = self._f1_at_k(translations, ref_trgs, k=k)
        return precision, recall, f1
