import itertools
import dataclasses
from typing import Iterable, Optional, Union, Dict, Tuple, List


@dataclasses.dataclass
class Sentence:
    id: int
    text: str
    context: str
    label: str

    def to_dict(self) -> Dict[str, Union[str, int]]:
        return dataclasses.asdict(self)


def balance_targets(sentences: Iterable[Sentence], method: str = "downsample_o_cat", shuffle=True) \
        -> Iterable[Sentence]:
    """
    Oversamples and/or undersamples training sentences by a number of targets.
    This is useful for linear shallow classifiers, that are prone to simply overfit the most-occurring category.
    See the source code for a documentation of resample methods logic
    :param shuffle: whether to shuffle the output
    :param sentences: sentences to resample
    :param method: resample method, one of {downsample_o_cat, downsample_o_pzk_cats, all_upsampled, remove_o_cat}
    :return: resampled, possibly shuffled input sentences
    """
    import random
    # take the second-top count from categories apart from "Other"
    targets = [s.label for s in sentences]
    second_top_count = sorted([sum([target == cat for target in targets]) for cat in set(targets) - {"O"}])[-2]
    if method == "downsample_o_cat":
        # downsample "other" category to second-most-occurring category count
        out_sentences = list((random.sample([s for s in sentences if s.label == "O"], second_top_count) +
                         [s for s in sentences if s.label != "O"]))
    elif method == "downsample_o_pzk_cats":
        # downsample "other" + "P_ZK" (experience description) category to third-most-occurring category count
        out_sentences = list((random.sample([s for s in sentences if s.label == "O"], second_top_count) +
                         [s for s in sentences if s.label != "O"]))
        out_sentences = list((random.sample([s for s in out_sentences if s.label == "P_ZK"], second_top_count) +
                         [s for s in out_sentences if s.label != "P_ZK"]))
    elif method == "all_upsampled":
        # upsample all categories to a count of most-occurring one (presumably "other" category)
        from itertools import chain
        out_sentences = list(itertools.chain(*[random.choices([s for s in sentences if s.label == cat],
                                                              k=second_top_count) for cat in set(targets)]))
    elif method == "remove_o_cat":
        # completely remove sentences of "other" category
        out_sentences = [s for s in sentences if s.label != "O"]
    else:
        out_sentences = sentences
    if shuffle:
        # random shuffle output sentences
        random.shuffle(out_sentences)
    return out_sentences


def get_sentence_vertical(sentences_dir: str, confidence_thrd: Optional[int] = 0) -> 'DataFrame':
    """
    Creates a tab-separated csv table with sentences_text, tags, users and sources, in out_table_path
    :param sentences_dir: directory of input sentences, divided to [train, val, test] subdirectories
    :param confidence_thrd: minimal mean confidence threshold of the retrieved sentences
    :return: Dataframe with attributes of retrieved sentences
    """
    from itertools import chain
    import pandas as pd  # if you need this, run 'pip install pandas==1.2.1'
    from utils.dataset import ReflexiveDataset

    sentences_splits = [ReflexiveDataset.sentences_from_tsv(sentences_dir, dataset_type, confidence_thrd,
                                                            use_context=True)
                        for dataset_type in ["train", "val", "test"]]
    out_vertical = pd.DataFrame.from_records([s.to_dict() for s in chain(*sentences_splits)])
    return out_vertical


def split_text_to_sentence_context(text: str, sep_chars: Tuple[str] = (".", "?", "!")) -> List[Tuple[str, str]]:
    """
    Splits the input text to sentences with the corresponding context,
    in the format compliant with the training of NeuralClassifier
    :param text: Full input paragraph, e.g. whole reflective diary, to extract the sentences to classify
    :param sep_chars: characters separating potential sentences
    """
    out_sentences = []
    current_sent = []
    words = text.split()

    for w_i, word in enumerate(words):
        current_sent.append(word)
        is_last_or_is_upper = (w_i == len(words)-1 or words[w_i+1][0].isupper())
        if any(word.endswith(mark) for mark in sep_chars) and is_last_or_is_upper:
            out_sentences.append(" ".join(current_sent))
            current_sent = []

    for sent_i, sent in enumerate(out_sentences):
        context = " ".join(out_sentences[sent_i-2:sent_i+2])
        yield sent, context
