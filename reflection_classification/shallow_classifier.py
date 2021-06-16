from typing import List, Optional
from gensim import corpora
from gensim import matutils

import numpy as np

from .utils.dataset import ReflexiveDataset
from .utils.data_reader import Sentence


class ShallowClassifier:
    word_dictionary = None

    def __init__(self, classifier, use_context: bool, bow_size: int,
                 sentences_dir: Optional[str] = None, lang: str = "cze"):
        self.classifier = classifier
        self.sentences_dir = sentences_dir
        self.use_context = use_context
        self.bow_size = bow_size
        self.lang = lang

    def _preprocess_string(self, text: str) -> List[str]:
        from gensim.parsing import preprocess_string
        if self.lang == "cze":
            from utils.cs_stemmer import cz_stem
            return [cz_stem(word) for word in preprocess_string(text)]
        else:
            return preprocess_string(text)

    def _initialize_bow_model(self, sents: List[Sentence]):
        text_preprocessed = [self._preprocess_string(str(s.text)) for s in sents]
        contexts_preprocessed = [self._preprocess_string(str(s.context)) for s in sents]

        self.word_dictionary = corpora.Dictionary(text_preprocessed + contexts_preprocessed)
        # keep most-occurring 10k words
        # we need to check this with Ullmann
        self.word_dictionary.filter_extremes(keep_n=self.bow_size)

    def _vectorize_sentences(self, sents: List[Sentence]):
        text_preprocessed = [self._preprocess_string(s.text) for s in sents]
        # sparse matrix contains just pairs of co-occurrences
        sparse_matrix = [self.word_dictionary.doc2bow(t) for t in text_preprocessed]
        # we want to get natural, dense vectors for each document, containing the most-frequent num_terms
        dense_matrix = matutils.corpus2dense(sparse_matrix, num_terms=self.bow_size).transpose()
        if not self.use_context:
            return dense_matrix
        else:
            # the same for contextual vectors
            text_preprocessed_c = [self._preprocess_string(str(s.context)) for s in sents]
            # sparse matrix contains just pairs of co-occurrences
            sparse_matrix_c = [self.word_dictionary.doc2bow(t) for t in text_preprocessed_c]
            # we want to get natural, dense vectors for each document, containing the most-frequent num_terms
            dense_matrix_c = matutils.corpus2dense(sparse_matrix_c, num_terms=self.bow_size).transpose()

            # concat textual and contextual vectors horizontally
            return np.hstack([dense_matrix, dense_matrix_c])

    def train(self, in_sentences: List[Sentence] = None, confidence_thrd: int = 5):
        if in_sentences is None:
            # get the dataset from outside
            sentences = ReflexiveDataset.sentences_from_tsv(self.sentences_dir, "train",
                                                            confidence_thrd, self.use_context)
        else:
            # user gets the dataset himself
            sentences = in_sentences

        self._initialize_bow_model(sentences)
        vectors = self._vectorize_sentences(sentences)
        self.classifier.fit(vectors, [s.label for s in in_sentences])

    def predict(self, sentences: List[Sentence]):
        vectors = self._vectorize_sentences(sentences)
        targets = self.classifier.predict(vectors)
        return targets


