import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import pandas as pd

from pathlib import Path
from typing import *
from sklearn.metrics import f1_score

import torch
import torch.optim as optim
from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from fastai.metrics import *

class BertMaskedLM(BaseTokenizer):

    def __init__(self, model_name:str='bert-base-uncased', do_lower_case: bool=True):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        # Load pre-trained model (weights)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.eval()

    def predict_token(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Predict all tokens
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)

        masked_index = text.split(' ').index("[MASK]")
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
        return predicted_token

    def predict_tokens(self, text):
        preds = []
        sentences = [f"{s} [SEP]"  for s in text.split("[SEP]")]
        sentences.pop() # remove last element whihch is just [SEP]
        for sentence in sentences:
            token = self.predict_token(sentence)
            sentence = sentence.replace("[MASK]", token)
            preds.append(sentence)
        return preds
