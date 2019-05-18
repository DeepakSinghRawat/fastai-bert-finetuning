# fastai-bert-finetuning

### Overview:

We will finetune pre-trained BERT model on The Microsoft Research Paraphrase Corpus (MRPC). MRPC is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.

### Steps:

1. Clone the repo and install dependencies by running:

```pip install -r requirements.txt```

2. Execute "Finetuning Bert on MRPC Corpus using FastAI" notebook

### Results:

We achieve high accuracy of ~0.83 and f1 score of ~0.88 by running for only 3 epochs.

Thanks to [Keita Kurita](https://github.com/keitakurita) for this excellent starter: [A Tutorial to Fine-Tuning BERT with Fast AI](http://mlexplained.com/2019/05/13/a-tutorial-to-fine-tuning-bert-with-fast-ai/)