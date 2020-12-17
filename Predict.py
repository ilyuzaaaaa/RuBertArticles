#!/usr/bin/env python
# coding: utf-8

import tokenization
import torch
import sys
import time
import numpy as np
import pandas as pd
from IPython.display import clear_output
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from modeling import BertConfig, BertForSequenceClassification
from optimization import BERTAdam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef

import matplotlib.pyplot as plt
import tkinter as tk

df = pd.read_csv('/Users/iluzaangirova/Ilyuza/dataset_5.csv', encoding='utf8')
df = df[['topics', 'title']]
maps = pd.factorize(df.topics)[1]

device = 'cpu'
bert_config = BertConfig.from_json_file('/Users/iluzaangirova/Ilyuza/bert_config.json')
model = BertForSequenceClassification(bert_config, 5)
model.to(device)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('/Users/iluzaangirova/Ilyuza/weights5000.pth', map_location='cpu'))

tokenizer = tokenization.FullTokenizer(vocab_file='/Users/iluzaangirova/Ilyuza/vocab.txt', do_lower_case=False)

def predict(sentence, tokenizer, max_len):
#     print(sentence)
    tokens_a = tokenizer.tokenize(sentence)
    if len(tokens_a) > max_len - 2:
        tokens_a = tokens_a[0:(max_len - 2)]

    tokens = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_len:
        input_ids.append(0)
        input_mask.append(0)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0)
    input_mask = torch.LongTensor(input_mask).unsqueeze(0)
    a = time.time()

    logits = model(input_ids.cpu(), None, input_mask.cpu())
    logits = logits.squeeze(0)

    b = time.time()
    logits = F.softmax(logits, dim=-1)
    clas = logits.argmax().item()
    return clas

if __name__=="__main__":
	print('Hello')
	window = tk.Tk()
	window.title('Classifier')
	label = tk.Label(text="Введите заголовок статьи")
	entry = tk.Entry()
	label.pack()
	entry.pack()
	title1 = entry.get()
	def button():
	    label = tk.Label(window, text=maps[predict(entry.get(), tokenizer, 512)])
	    label.pack()
	btn1 = tk.Button(window, text = 'классифицировать', command = button)
	btn1.pack()
	window.mainloop()
