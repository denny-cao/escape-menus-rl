# Generate BERT embeddings for the given text data

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import os
import sys
import time

# Load the pre-trained BERT 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Function to generate BERT embeddings for the given text data
def generate_embeddings(text_data):
    # Tokenize the input text
    tokenized_text = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt')
    # Generate the BERT embeddings
    with torch.no_grad():
        outputs = model(**tokenized_text)
        embeddings = outputs.last_hidden_state
    return embeddings

