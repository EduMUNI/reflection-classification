# Czech-English Reflective Dataset (CEReD)

### [Update 14th Sep 2024]: CEReD is now also [available](https://huggingface.co/datasets/MU-NLPC/CEReD) in 🤗 HuggingFace datasets

1. Install the Datasets library with pip:

```shell
pip install datasets
```

2. Use the dataset directly in your Python scripts:

```python
from datasets import load_dataset

ds = load_dataset("MU-NLPC/CEReD", "cs")
print(print(ds["train"]))
```
```python
Dataset({
    features: ['idx', 'context', 'sentence', 'y', 'confidence', 'y_requires_context'],
    num_rows: 6096
})
```
Other ways to access the dataset can also be found on https://huggingface.co/datasets/MU-NLPC/CEReD 

### Access the data from this repository (deprecated)

This directory contains an anonymized data set of reflective diaries/journals
in two formats:

1. In `data/diaries/cs/diaries.tsv`, you can find the anonymized subset 
   of czech diaries collected from the candidate student teachers. 
   The tab-separated file `diaries.tsv` contain following attributes:
   * `id`: unique reflective diary id
   * `person_id`: synthetic id of a creator of the diary
   * `subject`: subject that the reflective diary concern
   * `ordering`: relative rank of the diary relative to other diaries of the same author
   * `Q1`: Teacher evaluation: "Student treated the leading teacher with respect."
   * `Q2`: Teacher evaluation: "Student took responsibility in a preparation for practice."
   * `Q3`: Teacher evaluation: "Student discussed specific means of their further development."
   * `Q4`: Teacher evaluation: "Student actively asked me for a support, feedback, reflection."
   * `Q5`: Teacher evaluation: "Student actively reflected on their activity on practice."
   * `Q6`: Teacher evaluation: "Student recognized the situation of the class and reacted to it with selected stragegy."
   * `Q7`: Teacher evaluation: "Student shown interest in a situation in school, in general."
   * `diary`: Text of the reflective diary
    
   All questions `Q[1-7]` are part of the questionnaire
   filled by the supervising teacher on the relevant practice. 
   The questionnaire concerned the performance evaluation of
   the candidate teacher student, that authored the reflective diary.
   
2. In `sentences/{cs\en}/{train\val\test}/sentences.tsv`, you can
   find sentences that can be used for training a classifier, in
   selected language: original: Czech or translated: English.
   Sentences are divided into train, validation (val) and test set.
   This split can be used to evaluate the classifier on the same
   data, as we did, hence it allows for comparability of 
   the results.
   Again, the tab-separated `sentences.tsv` files contain following 
   attributes:
    * `idx`: unique sentence id
    * `context`: textual context surrounding the classified sentence
    * `sentence`: text of the classified sentence
    * `y`: target category of the sentence, that annotators agreed upon
    * `confidence`: confidence, or typicality of the sentence in its assigned category. Annotators were asked: "How typical is this sentence for the picked category?"
    * `y_requires_context`: whether annotators needed to look at the context, when selecting a category.

# Citation and license of database

APA style: Štefánik, M. & Nehyba, J. (2021). Czech-English Reflective Dataset (CEReD). (Version V1) [Data set]. GitHub. doi:11372/LRT-3573.

The database license is defined by the repository license.
