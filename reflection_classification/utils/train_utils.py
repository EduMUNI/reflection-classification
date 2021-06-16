from .dataset import ReflexiveDataset
from dataclasses import dataclass, field
from transformers import EvalPrediction, AutoTokenizer
from sklearn.metrics import f1_score
import numpy as np
from typing import Dict, List


def eval_fscore_acc(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).mean()
    f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def get_datasets(tokenizer: AutoTokenizer, sentences_dir: str, label_list: List[str],
                 cache_dir: str, use_context: bool, confidence_thrd: int) -> List[ReflexiveDataset]:
    return [ReflexiveDataset(sentences_dir, tokenizer=tokenizer, dataset_type=dataset_type, use_context=use_context,
                             cache_dir=cache_dir, label_list=label_list, mean_confidence_threshold=confidence_thrd)
            for dataset_type in ["train", "val", "test"]]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/transformer_config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
