You can use a library `gdown` to download pretrained models, just like in `\analyses` notebooks, 
or download the folder manually: 

```bash
pip install gdown==4.7.1
gdown https://drive.google.com/uc?id=1mMzlBz1Sipg8eOYMTF6DaqyzlBtjHOuo -O xlm-roberta-large-confidence4+-cs.zip
mkdir models
unzip xlm-roberta-large-confidence4+-cs.zip -d models
```

## English RoBERTa-XLM model:

The pretrained model (trained on train_conf=4) is available on: https://drive.google.com/uc?id=1mMzlBz1Sipg8eOYMTF6DaqyzlBtjHOuo

## Czech RoBERTa-XLM model:

The pretrained model (trained on train_conf=4) is available on: https://drive.google.com/uc?id=1X-bdeh8qwUQ3GPx3eXqb06HPHQfBwb0q

----

If you'd like to experiment with other configurations reported in the paper, feel free to get back to us!
