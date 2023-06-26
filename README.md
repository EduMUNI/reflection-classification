# Reflectivity classification

This repository contains reproducible experiments of our article [Applications of deep language models for reflective writings](https://rdcu.be/cUWGY),
with experiments using classification of reflective types in reflective writings.

We hope that the great ease of reproducibility of our results
will allow other researchers a head start on further research
in related topics.

## Steps to reproduce

For each step, we provide a Google Colab notebook that will set everything ready
for you to use.

However, the core package 'reflection_classification', as well as evaluation scripts will as well work locally. Here are the instructions to run the package in the new environment, for example, locally:

We presume you have linux-like system with python3.8 installed.

```bash
git clone {this repository}
cd reflection-classification

# do not forget to activate, or create an appropriate environment here:
# here's how you create it (presuming you have python3.8 installed, 
# you can check with `which python`, or on Windows `where python`):
python -m venv reflection
# on Windows `py -m venv reflection`
source reflection/bin/activate
# on Windows: `.\reflection\Scripts\activate`

# install the package with dependences:
python -m pip install -e .
```

As the functionality of simple, shallow classifiers is very distinct from
the neural ones, we provide a separate functionality for experimenting with each.


### 1. Shallow classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wyl93yACMiOvlEzTEXhK1JLHIxS8u4RT#scrollTo=vjBYBZym3lGc)

You can train and evaluate selected shallow classifier
using `train_eval_shallow_classifier.py`, 

as a standalone python application:

```bash
python scripts/train_eval_shallow_classifier.py [--args]
```

with following arguments:

```bash
usage: train_eval_shallow_classifier.py [-h] [--classifier CLASSIFIER] --sentences_dir SENTENCES_DIR [--train_confidence_threshold TRAIN_CONFIDENCE_THRESHOLD]
                                        [--test_confidence_threshold TEST_CONFIDENCE_THRESHOLD] [--use_context USE_CONTEXT] [--vocabulary_size VOCABULARY_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --classifier CLASSIFIER
                        Classifier to use. One of: {random_forrest, logistic_regression, naive_bayes, support_vector_classifier}
  --sentences_dir SENTENCES_DIR
                        Directory with {split}/sentence.tsv of annotated sentences
  --train_confidence_threshold TRAIN_CONFIDENCE_THRESHOLD
                        Minimal confidence threshold for sentences to train on.
  --test_confidence_threshold TEST_CONFIDENCE_THRESHOLD
                        Minimal confidence threshold for sentences to test on.
  --use_context USE_CONTEXT
                        Whether the model was trainer using context.
  --vocabulary_size VOCABULARY_SIZE
                        Number of top-n most-occurring words used to create Bag of Words representation for classification
```

### 2. Neural classifier

#### 2.1 Training the model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wyl93yACMiOvlEzTEXhK1JLHIxS8u4RT#scrollTo=Hqb32g3-82ms)

Our trained models **are available for download: see the instructions in 
`classifiers/models`**, you should be able to reproduce the published results without a new training.

Note that for reproducing a training of neural classifier,
you might need an access to at least one GPU with at least 15 GB of GRAM. We used Nvidia Tesla T4 for training the referenced models. 
Compared to CPU training (40 cores), his will cut the training time from 3-4 days to 6-10 hours, depending on configuration.

You can train neural classifiers
using `classifiers/train_neural_classifier.py`, 

as a standalone python application:

```bash
python scripts/train_neural_classifier.py [--args]
```

with following arguments

```bash
usage: train_neural_classifier.py [-h] --model_name MODEL_NAME --sentences_dir SENTENCES_DIR [--train_confidence_threshold TRAIN_CONFIDENCE_THRESHOLD] --trained_model_dir
                                  TRAINED_MODEL_DIR --device DEVICE [--eval_on_test_set EVAL_ON_TEST_SET] [--use_context USE_CONTEXT]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model name, or local path to finetune.
  --sentences_dir SENTENCES_DIR
                        Directory with .tsvs of annotated sentences
  --train_confidence_threshold TRAIN_CONFIDENCE_THRESHOLD
                        Minimal confidence threshold for sentences to train on.
  --trained_model_dir TRAINED_MODEL_DIR
                        Directory to be filled with trained model
  --device DEVICE       Device used for training. One of {cpu, cuda, cuda:[idx]}
  --eval_on_test_set EVAL_ON_TEST_SET
                        Whether to evaluate model (having lowest eval loss) on test set
  --use_context USE_CONTEXT
                        Whether the model will be trained using context.
```

Note that the training process produces training logs with evaluations
on validation set, that is used for picking the best model on output.
These logs are saved in
`runs` directory and can be accessed using tensorboard:
`tensorboard --logdir=runs`

#### 2.2 Evaluate the model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wyl93yACMiOvlEzTEXhK1JLHIxS8u4RT#scrollTo=2QtE3ipt9qWe&line=1&uniqifier=1)

Following command evaluates the accuracy of your model. The model can be 
1. Downloaded automatically from [HuggingFace](https://huggingface.co/MU-NLPC/XLM-R-large-reflective-conf4) by setting `--TRAINED_MODEL_DIR MU-NLPC/XLM-R-large-reflective-conf4`. Use this option to reproduce our results.
2. Downloaded manually - see [models](models) directory.
3. Trained using the script above and picked from `--trained_model_dir` you chose before.

```bash
python scripts/eval_neural_classifier.py [--args]
```

with following arguments

```bash
usage: eval_neural_classifier.py [-h] --trained_model_dir TRAINED_MODEL_DIR --sentences_dir SENTENCES_DIR [--device DEVICE]
                                 [--test_confidence_threshold TEST_CONFIDENCE_THRESHOLD] [--use_context USE_CONTEXT]

optional arguments:
  -h, --help            show this help message and exit
  --trained_model_dir TRAINED_MODEL_DIR
                        Local path containing pre-trained model, filled on training, or downloaded separately
  --sentences_dir SENTENCES_DIR
                        Directory with {split}/sentence.tsv of annotated sentences
  --device DEVICE       Device used to infer. One of {cpu, cuda, cuda:[idx]}
  --test_confidence_threshold TEST_CONFIDENCE_THRESHOLD
                        Minimal confidence threshold for sentences to test on.
  --use_context USE_CONTEXT
                        Whether the model was trainer using context.
```

## Measured results

A table of evaluation best-performing shallow (Random Forrest) and neural (XLM-RoBERTa) models, trained and tested on 
sentences having mean category confidence over the threshold in table. 
See the manuscript for details. 

#### Czech sentences

**XLM-RoBERTa** | Test >= 3 | Test >= 4 | Test >= 5 | Test >= 6  
--- | --- | --- | --- |--- 
Train >= 3 | 76.562% | 80.608% | 85.906% | 92.682%
Train >= 4 | 75.937% | 79.467% | 89.261% | 97.560% 
Train >= 5 | 74.062% | 77.566% | 85.906% | 95.121%  
Train >= 6 | 63.437% | 68.061% | 83.892% | 92.682% 
**Random Forrest** | **Test >= 3** | **Test >= 4** | **Test >= 5** | **Test >= 6**
Train >= 3 | 73.154% | 72.483% | 73.154% | 74.496% 
Train >= 4 | 71.812% | 71.812% | 72.483% | 72.483% 
Train >= 5 | 73.825% | 73.825% | 72.483% | 71.812% 
Train >= 6 | 73.825% | 73.154% | 73.154% | 73.825%
**Baseline*** | 39.597% | 28.137% | 39.597% | 48.780% 

**proportion of most-common category in test dataset*

#### English sentences

\ | Test >= 3 | Test >= 4 | Test >= 5 | Test >= 6  
--- | --- | --- | --- |---  
Train >= 3 | 79.375% | 82.706% | 90.506% | 95.454% 
Train >= 4 | 75.937% | 79.323% | 87.341% | 93.181% 
Train >= 5 | 79.375% | 82.331% | 92.405% | 100.00%  
Train >= 6 | 67.812% | 73.684% | 87.974% | 97.727% 
**Baseline*** | 39.597% | 28.137% | 39.597% | 48.780% 

**proportion of most-common category in test dataset*

#### Czech sentences using English model (cross-lingual)

\ | Test >= 3 | Test >= 4 | Test >= 5 | Test >= 6  
--- | --- | --- | --- |---  
Train >= 3 | 75.936% | 79.087% | 85.235% | 90.244% 
Train >= 4 | 73.125% | 77.566% | 83.221% | 90.244% 
Train >= 5 | 68.438% | 73.004% | 84.563% | 95.122%  
Train >= 6 | 59.063% | 64.638% | 81.208% | 95.122% 
**Baseline*** | 39.597% | 28.137% | 39.597% | 48.780% 

**proportion of most-common category in test dataset*


## 3. Hypotheses evaluation & further use

Notebooks in folder `analyses` contains reproducible evaluations of hypotheses introduced in the manuscript.
Each of them utilize our pre-trained `NeuralClassifier` in order to identify reflectivity in the original, anonymized
reflective diaries. 

Notebooks also demonstrate how the classifier can be used in practice for your own research.

If you have read the evaluation scripts and hypotheses notebooks, but still need a support with reproduction, or
use of classifier for your own research, please create an issue, or contact us at <stefanik.m@mail.muni.cz>.


## Citing

If you use or extend our **results or software** in your research, it would be great if you cite us as follows:
```bibtex
@Article{Nehyba2022applications,
  author={Nehyba, Jan and {\v{S}}tef{\'a}nik, Michal},
  title={Applications of deep language models for reflective writings},
  journal={Education and Information Technologies},
  year={2022},
  month={Sep},
  day={05},
  issn={1573-7608},
  doi={10.1007/s10639-022-11254-7},
  url={https://doi.org/10.1007/s10639-022-11254-7}
}
```

If you use **CEReD dataset**, please use the following citation:
```bibtex
 @misc{Stefanik2021CEReD,
   title = {Czech and English Reflective Dataset ({CEReD})},
   author = {{\v S}tef{\'a}nik, Michal and Nehyba, Jan},
   url = {http://hdl.handle.net/11372/LRT-3573},
   copyright = {Creative Commons - Attribution 4.0 International ({CC} {BY} 4.0)},
   year = {2021} 
 }
```
