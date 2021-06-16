import os
from enum import Enum
from pathlib import Path
from typing import List, Union, Dict
import ast

from filelock import FileLock
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, InputFeatures, logging
import pandas as pd
from .data_reader import Sentence

logger = logging.get_logger()

# original labels:
# LABELS = ["O", "OS_PRES", "PERS", "POC", "P_ZK", "REF_P", "UV_OBT", "VY_IN", "VY_VY"]

LABELS = ["Other", "Belief", "Perspective", "Feeling", "Experience",
          "Reflection", "Difficulty", "Intention", "Learning"]


class Split(Enum):
    train = "train"
    eval = "eval"
    test = "test"


class ReflexiveDataset(Dataset):
    
    def __init__(self, sentences_dir: str, dataset_type: str, cache_dir: str, label_list: List[str],
                 tokenizer: Union[AutoTokenizer, PreTrainedTokenizer],
                 use_context=True, mean_confidence_threshold: int = 5):
        self.sentences_dir = sentences_dir
        self.confidence_thrd = mean_confidence_threshold
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.use_context = use_context

        if not Path(cache_dir).exists():
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        cached_features_file = os.path.join(
            cache_dir,
            "cached_{}_{}_{}".format(dataset_type, tokenizer.__class__.__name__, str(self.tokenizer.model_max_length)),
        )
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            logger.info(f"Creating features from reflexive diaries")
            self.sentences = self.sentences_from_tsv(sentences_dir, dataset_type, self.confidence_thrd, self.use_context)
            self.features = self.convert_examples_to_features(self.sentences)

    @staticmethod
    def sentences_from_tsv(sentences_dir: str, dataset_type: str,
                           confidence_thrd: int, use_context: bool) -> List[Sentence]:
        """Creates sentences for the training, eval and test sets."""
        tsv_path = os.path.join(sentences_dir, dataset_type, "sentences.tsv")
        df = pd.read_csv(tsv_path, sep='\t')
        df.sentence = df.sentence.fillna("")
        df.context = df.context.fillna("")
        sentences = []
        # group by sources, iterate every group separately, to avoid context overlays
        for idx, row in enumerate(df.itertuples()):
            confidences = ast.literal_eval(row.confidence)
            if sum(confidences) / len(confidences) >= confidence_thrd:
                sentences.append(Sentence(id=row.idx, text=row.sentence,
                                          context=row.context if use_context else None,
                                          label=row.y))
        logger.info("Retrieving %s of all %s %s sentences, over threshold %s" %
                    (len(sentences), len(df), dataset_type, confidence_thrd))
        return sentences
    
    def convert_examples_to_features(self, examples: List[Sentence]) -> List[Dict[str, List[int]]]:

        batch_encoding = self.tokenizer(
            text=[example.text.strip() for example in examples],
            text_pair=[e.context.strip() for e in examples] if self.use_context else None,
            padding="max_length",
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding if k != "token_type_ids"}
            inputs["label"] = self.label_map[examples[i].label]
            features.append(inputs)

        for i, example in enumerate(examples[:5]):
            logger.info("*** Example ***")
            logger.info("id: %s" % (example.id))
            logger.info("features: %s" % features[i])

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
